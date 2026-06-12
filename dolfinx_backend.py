"""DOLFINx/FEniCSx backend for microstructure-ve.

Solves a `microstructure_ve.Simulation` with FEniCSx instead of ABAQUS, reusing the
same solver-neutral dataclasses (mesh, materials, periodic boundary, drive). Runs a
per-frequency complex linear-elasticity solve and writes a tsv of the homogenized
response mirroring `readODB.py`'s columns, so `verify_pbc.compare` can diff it against
an ABAQUS run.

Scope (serial, 2D). **Confined** macro loading (the imposed macro strain is fully
prescribed: uniaxial in x, zero lateral). This differs from the corner-driven ABAQUS
scheme in *how* the macro strain is applied, but yields the same physical observable
E*(f), which is the oracle (see docs/dolfinx_backend_design.md).

Why not the corner-driven reuse the design doc sketched: dolfinx_mpc silently drops
Dirichlet BCs placed on MPC *master* dofs, and the corner-driven scheme drives the
macro strain by Dirichlet-ing reference corners that are exactly those masters. The
standard fix used here is the classic periodic-homogenization split
``u = E_macro . x + u_periodic``: a pure-periodic MPC on the fluctuation u_periodic
(2-term, slave=image; our nsets give disjoint slaves, validated), the macro strain as a
RHS source term ``-∫ σ(E_macro):ε(v) dx``, and one interior node pinned to remove the
rigid translation. Free-lateral loading (E_yy floating so σ̄_yy=0) needs an extra global
unknown and is a documented follow-up.

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
    """Log-spaced excitation frequencies from the Dynamic subsection of the first step.

    For a tight oracle comparison, evaluate at the ABAQUS run's actual frequency column
    instead (pass it as `freqs`); the two need not coincide.
    """
    dyn = _find(sim.steps[0].subsections, Dynamic)
    return np.logspace(np.log10(dyn.f_initial), np.log10(dyn.f_final), dyn.f_count)


def _drive_displacement(sim):
    """The harmonic drive amplitude (delta) prescribed in the step's Displacement BC."""
    for step in sim.steps:
        sub = _find(step.subsections, DisplacementBoundaryCondition)
        if sub is not None:
            return float(np.real(sub.displacement))
    raise ValueError("no drive DisplacementBoundaryCondition found in any step")


def _material_cell_maps(model):
    """Per-(original)-cell material index, plus each material's poisson and modulus fn.

    Cell order is the raveled pixel order (== our 1-indexed element order); the caller
    scatters through mesh.topology.original_cell_index before assigning to a DG0 fn.
    """
    ncells = int(np.prod(model.nodes.shape - 1))
    mat_of_cell = np.empty(ncells, dtype=int)
    poissons, modulus_fns = [], []
    for mi, mat in enumerate(model.materials):
        mat_of_cell[np.asarray(mat.elset.elements) - 1] = mi
        poissons.append(mat.poisson)
        modulus_fns.append(mat.complex_modulus)
    return mat_of_cell, np.array(poissons), modulus_fns


def run(sim, freqs=None, output_path=None):
    """Solve `sim` (confined uniaxial-x macro strain) over `freqs` and return an
    (len(freqs), 7) array with readODB-style columns. Writes a tsv if output_path given.
    """
    from itertools import product

    import ufl
    import basix.ufl
    from mpi4py import MPI
    from petsc4py import PETSc
    import dolfinx
    from dolfinx import fem
    import dolfinx_mpc

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

    # macro strain: confined uniaxial in x; delta at the driven corner == exx * Lx
    exx = _drive_displacement(sim) / Lx
    U1 = exx * Lx  # reported applied displacement (== delta)

    # ---- mesh from our grid connectivity (tensor-product / zigzag order, no ccw swap) ----
    coords = np.column_stack([(scale * c).ravel() for c in np.indices(shape)[::-1]])
    all_nodes = 1 + np.ravel_multi_index(np.indices(shape), shape)
    slices = list(product((np.s_[:-1], np.s_[1:]), repeat=2))
    cells = np.column_stack([all_nodes[sl].ravel() for sl in slices]) - 1
    c_el = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)))
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, c_el, coords)  # 0.10: (…, e, x)

    # ---- node <-> dof block map by rounded grid coordinates (exact on structured grid) ----
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
    mu_fn = fem.Function(DG0)
    lam_fn = fem.Function(DG0)

    def set_moduli(f):
        E_cell = np.empty(len(mat_of_cell), dtype=complex)
        for mi, mod in enumerate(modulus_fns):
            E_cell[mat_of_cell == mi] = complex(np.asarray(mod(np.array([f]))).ravel()[0])
        mu = E_cell / (2 * (1 + nu_cell))
        lam = E_cell * nu_cell / ((1 + nu_cell) * (1 - 2 * nu_cell))  # plane strain
        mu_fn.x.array[:] = mu[oci]
        lam_fn.x.array[:] = lam[oci]

    # ---- forms: a(u_per, v) = -∫ σ(E_macro):ε(v); total stress uses E_macro + ε(u_per) ----
    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sig(e):
        return 2 * mu_fn * e + lam_fn * ufl.tr(e) * ufl.Identity(2)

    Emac = ufl.as_matrix([[PETSc.ScalarType(exx), 0], [0, 0]])  # confined
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sig(eps(u)), eps(v)) * ufl.dx  # complex inner conjugates v itself
    L = -ufl.inner(sig(Emac), eps(v)) * ufl.dx

    # ---- pure-periodic MPC on the fluctuation: u_per(slave) = u_per(image), disjoint ----
    o = ns["X0Y0"].node_inds[0]
    pairs = list(zip(ns["X1"].node_inds, ns["X0"].node_inds))
    pairs += list(zip(ns["Y1"].node_inds, ns["Y0"].node_inds))
    pairs += [(ns["X1Y0"].node_inds[0], o), (ns["X0Y1"].node_inds[0], o), (ns["X1Y1"].node_inds[0], o)]
    sm = {coord(s).tobytes(): {coord(m).tobytes(): 1.0} for s, m in pairs}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    for comp in range(nodes.dim):
        mpc.create_general_constraint(sm, comp, comp)
    mpc.finalize()

    # ---- pin one interior node (not in the MPC) to remove rigid translation ----
    cx, cy = (shape[1] - 1) // 2, (shape[0] - 1) // 2
    pin = np.array([cx * scale, cy * scale])

    def at_pin(x):
        return np.isclose(x[0], pin[0]) & np.isclose(x[1], pin[1])

    bcs = []
    for comp in range(nodes.dim):
        Vc, _ = V.sub(comp).collapse()
        dofs = dolfinx.fem.locate_dofs_geometrical((V.sub(comp), Vc), at_pin)
        fbc = fem.Function(Vc)
        fbc.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(fbc, dofs, V.sub(comp)))

    # ---- frequency sweep ----
    uh = fem.Function(V)
    etot = Emac + eps(uh)
    sxx_form = fem.form(sig(etot)[0, 0] * ufl.dx)
    sxy_form = fem.form(sig(etot)[0, 1] * ufl.dx)
    area = fem.assemble_scalar(fem.form(fem.Constant(mesh, PETSc.ScalarType(1.0)) * ufl.dx)).real

    rows = []
    for f in freqs:
        set_moduli(f)
        # dolfinx 0.10: petsc_options only apply under a prefix; direct complex LU (MUMPS)
        problem = dolfinx_mpc.LinearProblem(
            a, L, mpc, bcs=bcs,
            petsc_options_prefix=f"msve_dolfinx_{id(mesh)}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps"},
        )
        sol = problem.solve()
        uh.x.array[:] = sol.x.array
        # homogenized stresses; corner x-reaction == σ̄_xx * Ly (matches readODB at X1Y0)
        sxx = fem.assemble_scalar(sxx_form) / area
        sxy = fem.assemble_scalar(sxy_form) / area
        rf1, rf2 = sxx * Ly, sxy * Ly
        rows.append([f, rf1.real, rf2.real, rf1.imag, rf2.imag, U1, 0.0])

    out = np.array(rows)
    if output_path is not None:
        np.savetxt(output_path, out, fmt="%.8e", delimiter="\t",
                   header="\t".join(READODB_HEADER), comments="")
    return out
