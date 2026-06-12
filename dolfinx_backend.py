"""DOLFINx/FEniCSx backend for microstructure-ve.

Solves a `microstructure_ve.Simulation` with FEniCSx instead of ABAQUS, reusing the
same solver-neutral dataclasses (mesh, materials, periodic boundary, drive). Runs a
per-frequency complex linear-elasticity solve and writes a tsv of the homogenized
response mirroring `readODB.py`'s columns, so `verify_pbc.compare` can diff it against
an ABAQUS run.

Scope: serial, 2D or 3D, periodic homogenization via the split ``u = E_macro . x + u_per``:
a pure-periodic 2-term MPC on the fluctuation u_per (slave = every node on a max face,
master = the same node with its max coordinates mapped to 0 -> disjoint slaves), the macro
strain imposed as a RHS source ``-∫ σ(E_macro):ε(v) dx``, one interior node pinned for
rigid translation. (The corner-driven ABAQUS scheme can't be reused directly because
dolfinx_mpc silently drops Dirichlet BCs on MPC master dofs.)

Features:
- ``lateral="confined"`` (default): the lateral macro normal strains are 0.
  ``lateral="free"``: they float so the lateral σ̄ normal components vanish, by superposition
  of the per-axis unit solves (same stiffness -> the extra solves are cheap back-substitutions).
- B-bar (selective reduced integration on the volumetric term, κ = λ + 2μ/d) to avoid
  volumetric locking, matching ABAQUS's CPE4/C3D8 B-bar formulation.
- One LU factorization reused across all frequencies and all unit-strain RHS. PETSc's native
  serial LU is used (MUMPS is ~25x slower on these small complex systems).

The macro loading is uniaxial along x (coordinate axis 0). `dolfinx` is imported lazily so
importing this module under the msve env (numpy only) does not fail until `run()` is called.
"""
import numpy as np

from microstructure_ve import Dynamic, DisplacementBoundaryCondition, PeriodicBoundaryCondition

_ELEMENT = {2: "quadrilateral", 3: "hexahedron"}


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


def _periodic_pairs(shape):
    """(slave_node, master_node) 1-indexed pairs for fluctuation periodicity.

    Slave = any node with at least one grid index at the max face; master = the same node
    with every max index mapped to 0. Disjoint slaves (each max-face node once); masters
    never sit on a max face, so no master is a slave. Reduces to the 2D edge+corner pairing.
    """
    shape = np.asarray(shape)
    all_nodes = 1 + np.ravel_multi_index(np.indices(shape), shape)
    idx = np.indices(shape)
    is_slave = np.zeros(shape, dtype=bool)
    img_idx = []
    for ax in range(len(shape)):
        at_max = idx[ax] == shape[ax] - 1
        is_slave |= at_max
        img_idx.append(np.where(at_max, 0, idx[ax]))
    master_nodes = all_nodes[tuple(img_idx)]
    return list(zip(all_nodes[is_slave], master_nodes[is_slave]))


def run(sim, freqs=None, output_path=None, lateral="confined", bbar=True):
    """Solve `sim` over `freqs` and return a readODB-style array (len(freqs), 1+3*dim).

    lateral: "confined" (lateral normal strains = 0) or "free" (lateral σ̄ normals = 0).
    bbar:    selective reduced integration on the volumetric term (recommended).
    Writes a tsv if output_path is given. Macro loading is uniaxial along x (axis 0).
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
    dim = nodes.dim
    assert dim in (2, 3), "dolfinx backend supports 2D or 3D"
    assert _find(model.bcs, PeriodicBoundaryCondition) is not None, "need a PeriodicBoundaryCondition"
    scale = nodes.scale
    shape = nodes.shape  # (.., ny+1, nx+1); coord axis i <-> grid axis (dim-1-i)
    L = [(shape[dim - 1 - i] - 1) * scale for i in range(dim)]  # [Lx, Ly, (Lz)]
    Lx = L[0]
    cross_area = float(np.prod(L[1:]))  # area perpendicular to the x drive
    exx = _drive_displacement(sim) / Lx  # macro xx strain; delta == exx * Lx

    # ---- mesh (tensor-product / basix lexicographic cell order, same in 2D and 3D) ----
    coords = np.column_stack([(scale * c).ravel() for c in np.indices(shape)[::-1]])
    all_nodes = 1 + np.ravel_multi_index(np.indices(shape), shape)
    slices = list(product((np.s_[:-1], np.s_[1:]), repeat=dim))
    cells = np.column_stack([all_nodes[sl].ravel() for sl in slices]) - 1
    c_el = ufl.Mesh(basix.ufl.element("Lagrange", _ELEMENT[dim], 1, shape=(dim,)))
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, c_el, coords)  # 0.10: (…, e, x)

    # ---- node <-> dof block map by rounded grid coordinates (exact on structured grid) ----
    V = fem.functionspace(mesh, ("Lagrange", 1, (dim,)))
    dof_x = V.tabulate_dof_coordinates()[:, :dim]
    grid = np.rint(dof_x / scale).astype(int)  # columns: (i_x, i_y, [i_z])
    ravel_idx = tuple(grid[:, dim - 1 - j] for j in range(dim))  # back to grid-axis order
    ours_of_block = 1 + np.ravel_multi_index(ravel_idx, shape)
    block_of_node = np.empty(int(np.prod(shape)) + 1, dtype=int)
    block_of_node[ours_of_block] = np.arange(len(ours_of_block))

    def coord(node):
        return dof_x[block_of_node[int(node)]].astype(np.float64)

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
        lam = E_cell * nu_cell / ((1 + nu_cell) * (1 - 2 * nu_cell))  # 3D / plane-strain Lame
        mu_fn.x.array[:] = mu[oci]
        lam_fn.x.array[:] = lam[oci]

    # ---- forms ----
    eye = ufl.Identity(dim)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def dev(e):
        return e - ufl.tr(e) / dim * eye

    def sig(e):  # full pointwise stress (RHS + recovery; B-bar-consistent since tr(eps) is linear)
        return 2 * mu_fn * e + lam_fn * ufl.tr(e) * eye

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    if bbar:
        dxf = ufl.dx(metadata={"quadrature_degree": 2})
        dxr = ufl.dx(metadata={"quadrature_degree": 1})  # 1-point (centroid) for quad/hex
        kappa = lam_fn + 2 * mu_fn / dim
        a = (2 * mu_fn * ufl.inner(dev(eps(u)), dev(eps(v)))) * dxf \
            + kappa * ufl.inner(ufl.tr(eps(u)), ufl.tr(eps(v))) * dxr
    else:
        a = ufl.inner(sig(eps(u)), eps(v)) * ufl.dx
    a_form = fem.form(a)

    # unit macro normal strains E0=xx, E1=yy, (E2=zz); RHS source per unit strain
    unit_E = []
    L_forms = []
    for i in range(dim):
        Ei = np.zeros((dim, dim))
        Ei[i, i] = 1.0
        Em = ufl.as_matrix(Ei.tolist())
        unit_E.append(Em)
        L_forms.append(fem.form(-ufl.inner(sig(Em), eps(v)) * ufl.dx))

    # ---- pure-periodic MPC (disjoint slaves: every max-face node -> its min image) ----
    sm = {coord(s).tobytes(): {coord(m).tobytes(): 1.0} for s, m in _periodic_pairs(shape)}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    for comp in range(dim):
        mpc.create_general_constraint(sm, comp, comp)
    mpc.finalize()

    # ---- pin one interior node to remove rigid translation ----
    center = np.array([(shape[dim - 1 - i] - 1) // 2 * scale for i in range(dim)])

    def at_center(x):
        ok = np.ones(x.shape[1], dtype=bool)
        for i in range(dim):
            ok &= np.isclose(x[i], center[i])
        return ok

    bcs = []
    for comp in range(dim):
        Vc, _ = V.sub(comp).collapse()
        dofs = dolfinx.fem.locate_dofs_geometrical((V.sub(comp), Vc), at_center)
        fbc = fem.Function(Vc)
        fbc.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(fbc, dofs, V.sub(comp)))

    # ---- one factorized matrix, reused across frequencies and all unit-strain RHS ----
    uh = [fem.Function(V) for _ in range(dim)]  # uh[j] = fluctuation for unit strain Ej
    set_moduli(freqs[0])
    A = dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=bcs)
    A.assemble()
    b = dolfinx_mpc.assemble_vector(L_forms[0], mpc)
    x = A.createVecRight()
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")  # native serial LU (~25x faster than MUMPS here)

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

    # stress-recovery forms: full σ̄^(j) tensor per unit strain (averaged over the cell)
    pairs_ij = [(m, n) for m in range(dim) for n in range(dim)]
    sbar_forms = [{ij: fem.form(sig(unit_E[j] + eps(uh[j]))[ij[0], ij[1]] * ufl.dx) for ij in pairs_ij}
                  for j in range(dim)]
    area = fem.assemble_scalar(fem.form(fem.Constant(mesh, PETSc.ScalarType(1.0)) * ufl.dx)).real

    rows = []
    for f in freqs:
        set_moduli(f)
        A.zeroEntries()
        dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=bcs, A=A)
        A.assemble()
        ksp.setOperators(A)  # same nonzero pattern -> LU re-factors values only

        nsolve = dim if lateral == "free" else 1
        sbar = []  # sbar[j][(m,n)] = unit-strain-j homogenized stress component
        for j in range(nsolve):
            solve_rhs(L_forms[j], uh[j])
            sbar.append({ij: fem.assemble_scalar(sbar_forms[j][ij]) / area for ij in pairs_ij})

        e = np.zeros(dim, dtype=complex)
        e[0] = exx
        if lateral == "free" and dim > 1:
            # choose lateral normal strains so σ̄_ii = 0 for i = 1..dim-1
            M = np.array([[sbar[k][(i, i)] for k in range(1, dim)] for i in range(1, dim)])
            r = np.array([-exx * sbar[0][(i, i)] for i in range(1, dim)])
            e[1:] = np.linalg.solve(M, r)
        # combined homogenized stress row 0 (x-face traction): σ̄_{0,n} = Σ_j e_j sbar[j][(0,n)]
        sig0 = np.array([sum(e[j] * sbar[j][(0, n)] for j in range(len(sbar))) for n in range(dim)])
        RF = sig0 * cross_area  # x-face reaction vector (== readODB RF at the x drive node)
        U = np.zeros(dim)
        U[0] = exx * Lx
        rows.append([f] + list(RF.real) + list(RF.imag) + list(U))

    out = np.array(rows)
    if output_path is not None:
        header = (["frequency"]
                  + [f"RF_Real{i+1}" for i in range(dim)]
                  + [f"RF_Imag{i+1}" for i in range(dim)]
                  + [f"U{i+1}" for i in range(dim)])
        np.savetxt(output_path, out, fmt="%.8e", delimiter="\t",
                   header="\t".join(header), comments="")
    return out
