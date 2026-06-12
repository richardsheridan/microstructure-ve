"""DOLFINx/FEniCSx backend for microstructure-ve.

Solves a `microstructure_ve.Simulation` with FEniCSx instead of ABAQUS, reusing the
same solver-neutral dataclasses (mesh, materials, periodic boundary, drive). Runs a
per-frequency complex linear-elasticity solve and writes a tsv of the homogenized
response mirroring `readODB.py`'s columns, so `verify_pbc.compare` can diff it against
an ABAQUS run.

Scope: serial, 2D, periodic homogenization via the split ``u = E_macro . x + u_per``:
a pure-periodic 2-term MPC on the fluctuation u_per (built from our disjoint nsets), the
macro strain imposed as a RHS source ``-∫ σ(E_macro):ε(v) dx``, one interior node pinned
for rigid translation. (The corner-driven ABAQUS scheme can't be reused directly because
dolfinx_mpc silently drops Dirichlet BCs on MPC master dofs.)

Features:
- ``lateral="confined"`` (default): E_yy = 0 prescribed.
  ``lateral="free"``: E_yy floats so σ̄_yy = 0, by superposition of the E_xx-only and
  E_yy-only unit solves (same stiffness -> the second solve is a cheap back-substitution).
- B-bar (selective reduced integration on the volumetric term) to avoid Q1 volumetric
  locking, matching ABAQUS's CPE4 B-bar formulation.
- One MUMPS factorization reused across all frequencies and both RHS (symbolic reuse;
  only the numeric values change per frequency).

`dolfinx` is imported lazily so importing this module under the msve env (numpy only)
does not fail until `run()` is called.
"""
import numpy as np

from microstructure_ve import Dynamic, DisplacementBoundaryCondition, PeriodicBoundaryCondition

READODB_HEADER = ["frequency", "RF_Real1", "RF_Real2", "RF_Imag1", "RF_Imag2", "U1", "U2"]


def _find(items, cls):
    for it in items:
        if isinstance(it, cls):
            return it
    return None


def default_frequencies(sim):
    """Log-spaced excitation frequencies from the Dynamic subsection of the first step."""
    dyn = _find(sim.steps[0].subsections, Dynamic)
    return np.logspace(np.log10(dyn.f_initial), np.log10(dyn.f_final), dyn.f_count)


def _drive_displacement(sim):
    for step in sim.steps:
        sub = _find(step.subsections, DisplacementBoundaryCondition)
        if sub is not None:
            return float(np.real(sub.displacement))
    raise ValueError("no drive DisplacementBoundaryCondition found in any step")


def _material_cell_maps(model):
    ncells = int(np.prod(model.nodes.shape - 1))
    mat_of_cell = np.empty(ncells, dtype=int)
    poissons, modulus_fns = [], []
    for mi, mat in enumerate(model.materials):
        mat_of_cell[np.asarray(mat.elset.elements) - 1] = mi
        poissons.append(mat.poisson)
        modulus_fns.append(mat.complex_modulus)
    return mat_of_cell, np.array(poissons), modulus_fns


def run(sim, freqs=None, output_path=None, lateral="confined", bbar=True):
    """Solve `sim` over `freqs` and return an (len(freqs), 7) readODB-style array.

    lateral: "confined" (E_yy=0) or "free" (E_yy floats so σ̄_yy=0).
    bbar:    selective reduced integration on the volumetric term (recommended).
    Writes a tsv if output_path is given.
    """
    from itertools import product

    import ufl
    import basix.ufl
    from mpi4py import MPI
    from petsc4py import PETSc
    import dolfinx
    from dolfinx import fem
    import dolfinx.fem.petsc as fempetsc
    import dolfinx_mpc

    if lateral not in ("confined", "free"):
        raise ValueError("lateral must be 'confined' or 'free'")
    if freqs is None:
        freqs = default_frequencies(sim)
    freqs = np.asarray(freqs, dtype=float)

    model = sim.model
    nodes = model.nodes
    ns = nodes.nsets
    assert nodes.dim == 2, "dolfinx backend is 2D-only"
    assert _find(model.bcs, PeriodicBoundaryCondition) is not None, "need a PeriodicBoundaryCondition"
    scale = nodes.scale
    shape = nodes.shape  # (ny+1, nx+1)
    Lx = (shape[1] - 1) * scale
    Ly = (shape[0] - 1) * scale
    exx = _drive_displacement(sim) / Lx  # macro xx strain; delta == exx * Lx

    # ---- mesh (tensor-product / zigzag quad order, no ccw swap) ----
    coords = np.column_stack([(scale * c).ravel() for c in np.indices(shape)[::-1]])
    all_nodes = 1 + np.ravel_multi_index(np.indices(shape), shape)
    slices = list(product((np.s_[:-1], np.s_[1:]), repeat=2))
    cells = np.column_stack([all_nodes[sl].ravel() for sl in slices]) - 1
    c_el = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)))
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, c_el, coords)  # 0.10: (…, e, x)

    # ---- node <-> dof block map ----
    V = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    dof_xy = V.tabulate_dof_coordinates()[:, :2]
    grid_ij = np.rint(dof_xy / scale).astype(int)
    ours_of_block = 1 + np.ravel_multi_index((grid_ij[:, 1], grid_ij[:, 0]), shape)
    block_of_node = np.empty(int(np.prod(shape)) + 1, dtype=int)
    block_of_node[ours_of_block] = np.arange(len(ours_of_block))

    def coord(node):
        return dof_xy[block_of_node[int(node)]].astype(np.float64)

    # ---- DG0 complex Lame fields (reassigned per frequency via original_cell_index) ----
    oci = mesh.topology.original_cell_index
    mat_of_cell, poissons, modulus_fns = _material_cell_maps(model)
    nu_cell = poissons[mat_of_cell]
    DG0 = fem.functionspace(mesh, ("DG", 0))
    mu_fn, lam_fn = fem.Function(DG0), fem.Function(DG0)

    def set_moduli(f):
        E_cell = np.empty(len(mat_of_cell), dtype=complex)
        for mi, mod in enumerate(modulus_fns):
            E_cell[mat_of_cell == mi] = complex(np.asarray(mod(np.array([f]))).ravel()[0])
        mu = E_cell / (2 * (1 + nu_cell))
        lam = E_cell * nu_cell / ((1 + nu_cell) * (1 - 2 * nu_cell))  # plane strain
        mu_fn.x.array[:] = mu[oci]
        lam_fn.x.array[:] = lam[oci]

    # ---- forms ----
    eye = ufl.Identity(2)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def dev(e):
        return e - ufl.tr(e) / 2 * eye

    def sig(e):  # full pointwise stress (for RHS and stress recovery; B-bar-consistent there)
        return 2 * mu_fn * e + lam_fn * ufl.tr(e) * eye

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    if bbar:
        # deviatoric (full quad) + volumetric kappa*tr*tr (reduced 1-pt quad). kappa = lam + 2mu/d.
        dxf = ufl.dx(metadata={"quadrature_degree": 2})
        dxr = ufl.dx(metadata={"quadrature_degree": 1})
        kappa = lam_fn + mu_fn  # plane strain, d=2 -> lam + 2mu/2
        a = (2 * mu_fn * ufl.inner(dev(eps(u)), dev(eps(v)))) * dxf \
            + kappa * ufl.inner(ufl.tr(eps(u)), ufl.tr(eps(v))) * dxr
    else:
        a = ufl.inner(sig(eps(u)), eps(v)) * ufl.dx
    a_form = fem.form(a)

    Ex = ufl.as_matrix([[PETSc.ScalarType(1.0), 0], [0, 0]])  # unit xx macro strain
    Ey = ufl.as_matrix([[0, 0], [0, PETSc.ScalarType(1.0)]])  # unit yy macro strain
    Lx_form = fem.form(-ufl.inner(sig(Ex), eps(v)) * ufl.dx)
    Ly_form = fem.form(-ufl.inner(sig(Ey), eps(v)) * ufl.dx)

    # ---- pure-periodic MPC (disjoint slaves: edges interior + 3 corners -> origin) ----
    o = ns["X0Y0"].node_inds[0]
    pairs = list(zip(ns["X1"].node_inds, ns["X0"].node_inds))
    pairs += list(zip(ns["Y1"].node_inds, ns["Y0"].node_inds))
    pairs += [(ns["X1Y0"].node_inds[0], o), (ns["X0Y1"].node_inds[0], o), (ns["X1Y1"].node_inds[0], o)]
    sm = {coord(s).tobytes(): {coord(m).tobytes(): 1.0} for s, m in pairs}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    for comp in range(nodes.dim):
        mpc.create_general_constraint(sm, comp, comp)
    mpc.finalize()

    # ---- pin one interior node to remove rigid translation ----
    cx, cy = (shape[1] - 1) // 2, (shape[0] - 1) // 2
    pin = np.array([cx * scale, cy * scale])
    bcs = []
    for comp in range(nodes.dim):
        Vc, _ = V.sub(comp).collapse()
        dofs = dolfinx.fem.locate_dofs_geometrical(
            (V.sub(comp), Vc), lambda x: np.isclose(x[0], pin[0]) & np.isclose(x[1], pin[1]))
        fbc = fem.Function(Vc)
        fbc.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(fbc, dofs, V.sub(comp)))

    # ---- one factorized matrix, reused across frequencies and both RHS ----
    ux, uy = fem.Function(V), fem.Function(V)
    set_moduli(freqs[0])
    A = dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=bcs)
    A.assemble()
    b = dolfinx_mpc.assemble_vector(Lx_form, mpc)
    x = A.createVecRight()
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    # PETSc's built-in serial LU; ~25x faster than MUMPS on this small complex system.

    def solve_rhs(L_form, uout):
        with b.localForm() as bl:
            bl.set(0.0)
        dolfinx_mpc.assemble_vector(L_form, mpc, b)
        dolfinx_mpc.apply_lifting(b, [a_form], [bcs], mpc)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fempetsc.set_bc(b, bcs)
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        ksp.solve(b, x)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        fempetsc.assign(x, uout)
        mpc.homogenize(uout)
        mpc.backsubstitution(uout)

    # stress-recovery forms (full integration; consistent with B-bar since tr(eps) is linear)
    comps = [(0, 0), (1, 1), (0, 1)]
    sx_forms = {ij: fem.form(sig(Ex + eps(ux))[ij[0], ij[1]] * ufl.dx) for ij in comps}
    sy_forms = {ij: fem.form(sig(Ey + eps(uy))[ij[0], ij[1]] * ufl.dx) for ij in comps}
    area = fem.assemble_scalar(fem.form(fem.Constant(mesh, PETSc.ScalarType(1.0)) * ufl.dx)).real

    rows = []
    for f in freqs:
        set_moduli(f)
        A.zeroEntries()
        dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=bcs, A=A)
        A.assemble()
        ksp.setOperators(A)  # same nonzero pattern -> MUMPS reuses the symbolic factorization

        solve_rhs(Lx_form, ux)
        sbx = {ij: fem.assemble_scalar(sx_forms[ij]) / area for ij in comps}
        if lateral == "free":
            solve_rhs(Ly_form, uy)  # cheap: reuses the factorization
            sby = {ij: fem.assemble_scalar(sy_forms[ij]) / area for ij in comps}
            eyy = -exx * sbx[(1, 1)] / sby[(1, 1)]
            sxx = exx * sbx[(0, 0)] + eyy * sby[(0, 0)]
            sxy = exx * sbx[(0, 1)] + eyy * sby[(0, 1)]
        else:
            sxx = exx * sbx[(0, 0)]
            sxy = exx * sbx[(0, 1)]
        rf1, rf2 = sxx * Ly, sxy * Ly  # corner x-reaction == σ̄_xx * Ly (matches readODB at X1Y0)
        rows.append([f, rf1.real, rf2.real, rf1.imag, rf2.imag, exx * Lx, 0.0])

    out = np.array(rows)
    if output_path is not None:
        np.savetxt(output_path, out, fmt="%.8e", delimiter="\t",
                   header="\t".join(READODB_HEADER), comments="")
    return out
