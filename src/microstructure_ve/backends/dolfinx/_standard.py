"""Direct-Dirichlet solve path for standard (non-periodic) boundary conditions.

Assembles the B-bar stiffness with complex moduli, imposes Dirichlet BCs from the simulation's
model-level holds (zero, ``_loading.model_holds``) and step ``Prescribed`` drives (prescribed
displacement), solves with PETSc serial LU, then reports the readODB-style row:
``[f, RF_Real..., RF_Imag..., U_Real...]`` where RF and U are summed over the primary drive nodeset
nodes for each component.

Reaction-force computation: after solving for ``u``, assemble the internal-force vector
``f_int = K_full * u`` as the *action* of the unreduced bilinear form on the solution
(one vector assembly -- no second stiffness matrix). The reaction at each constrained
node is the corresponding entry of ``f_int`` (equilibrium gives zero at free nodes; the
residual at constrained nodes is the reaction force). Summing over the primary drive
nodeset per component reproduces the ABAQUS ``readODB`` summation.

No MPC or center-pin -- standard cells are clamped BVPs, not periodic fluctuation problems. The
standard path always solves with LU; the ``solver`` and ``petsc_options`` of ``run`` do not apply to
it.
"""
from __future__ import annotations

import numpy as np

from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc as fempetsc

from microstructure_ve.core import _node_array
from microstructure_ve.backends.dolfinx._loading import (
    model_holds,
    prescribed_bcs,
    unconstrained_components,
)

from ._solver import lu_ksp


def drive_dofs(space, nodes):
    """Per displacement component, the flat V dofs of the 1-indexed grid ``nodes``."""
    bs = space.V.dofmap.index_map_bs
    blocks = space.block_of_node[nodes]
    return [blocks * bs + c for c in range(space.dim)]


def _build_dof_maps(space):
    """Return per-component (Vc_space, flat_to_vc_inv) for Dirichlet BC construction.

    ``flat_to_vc_inv[c][flat_dof]`` maps a flat V-dof index to the corresponding dof
    index in the collapsed sub-space ``Vc`` for component ``c``.  Used to translate
    block_of_node-derived flat dofs into the (flat_parent, vc) pairs that ``dirichletbc``
    with ``V.sub(c)`` expects.
    """
    V = space.V
    bs = V.dofmap.index_map_bs
    n_flat = V.dofmap.index_map.size_local * bs
    Vc_spaces = []
    inv_maps = []
    for c in range(space.dim):
        Vc_c, dof_map_c = V.sub(c).collapse()
        inv = np.full(n_flat, -1, dtype=np.int32)
        for k, flat in enumerate(dof_map_c):
            inv[flat] = k
        Vc_spaces.append(Vc_c)
        inv_maps.append(inv)
    return Vc_spaces, inv_maps


def _make_dirichlet_bc(space, Vc_spaces, inv_maps, nodes_1indexed, comp, value):
    """Build one ``fem.dirichletbc`` for ``comp`` (0-indexed) on the given 1-indexed nodes.

    Returns ``(bc, flat_dofs, vc_dofs)``: the condition and the dofs it constrains, in the flat V
    array and in the collapsed sub-space ``Vc``.

    ``nodes_1indexed`` are 1-indexed grid node numbers.  ``comp`` is the 0-indexed
    displacement component (0=x, 1=y, 2=z).  ``value`` is the prescribed displacement
    (real; complex support not needed -- Dirichlet values are always real amplitudes).

    The dofs pair format ``[flat_parent_dofs, vc_dofs]`` matches what
    ``fem.locate_dofs_geometrical((V.sub(c), Vc), marker)`` returns: ``[flat_dofs,
    block_dofs]`` where flat_dofs are indices in the full V flat array and block_dofs
    are indices in the collapsed sub-space ``Vc``.
    """
    V = space.V
    bs = V.dofmap.index_map_bs
    block_of_node = space.block_of_node
    blocks = block_of_node[np.asarray(nodes_1indexed)].astype(np.int32)
    flat_dofs = (blocks * bs + comp).astype(np.int32)
    vc_dofs = inv_maps[comp][flat_dofs].astype(np.int32)
    fbc = fem.Function(Vc_spaces[comp])
    fbc.x.array[:] = 0.0
    if value != 0.0:
        fbc.x.array[vc_dofs] = value
    return fem.dirichletbc(fbc, [flat_dofs, vc_dofs], V.sub(comp)), flat_dofs, vc_dofs


def _parse_bcs_ex(sim, space, Vc_spaces, inv_maps, step=None):
    """Parse the BCs of ``step`` (default ``sim.steps[0]``) into detailed entries.

    Returns ``(entries, drive_nodes)`` where each entry is ``(bc, flat_dofs, vc_dofs)`` --
    the ``fem.DirichletBC`` plus the flat-V and collapsed-subspace dof indices it
    constrains (what a caller needs to read the *current* displacement at those dofs,
    e.g. to warm-start a multistep ramp from the committed state).

    ``drive_nodes``: 1-indexed node array of the *primary* drive nodeset (the first
    ``Prescribed`` drive in the step's subsections), used for RF and U summation.

    Model-level conditions that ``model_holds`` prescribe zero on their dofs, except on a (node,
    dof) that a step drive prescribes, where the step's value wins. Step-level ``Prescribed`` drives
    prescribe the actual drive displacement.
    """
    model = sim.model
    step = sim.steps[0] if step is None else step
    entries = []

    def add(nodes, comp, value):
        entries.append(_make_dirichlet_bc(space, Vc_spaces, inv_maps, nodes, comp, value))

    drives = prescribed_bcs(step.subsections)
    driven = {}
    for s in drives:
        for dof in s.constraint.dofs:
            driven.setdefault(int(dof), set()).update(
                int(n) for n in np.ravel(_node_array(s.target)))

    for bc in model.bcs:
        if model_holds(bc):
            nodes = np.ravel(_node_array(bc.target))
            for dof in bc.constraint.dofs:
                held = np.array([n for n in nodes if int(n) not in driven.get(int(dof), ())],
                                dtype=nodes.dtype)
                if held.size:
                    add(held, dof - 1, 0.0)

    drive_nodes = None
    for s in drives:
        nodes = np.ravel(_node_array(s.target))
        if drive_nodes is None:
            drive_nodes = nodes  # first drive = primary (for RF/U reporting)
        disp_value = float(np.real(s.constraint.value))
        for dof in s.constraint.dofs:
            add(nodes, dof - 1, disp_value)

    return entries, drive_nodes


def _parse_bcs(sim, space, Vc_spaces, inv_maps, step=None):
    """Parse the step's BCs into (dolfinx_bcs, drive_nodes); see ``_parse_bcs_ex``."""
    entries, drive_nodes = _parse_bcs_ex(sim, space, Vc_spaces, inv_maps, step)
    return [bc for bc, _, _ in entries], drive_nodes


def build_solver(sim, prob):
    """Build the direct-Dirichlet FE solver for a standard (non-periodic) ``sim``.

    Returns ``(solve_one, dim)`` where ``solve_one(f)`` returns the readODB-style row for
    one frequency: ``[f, RF_Real..., RF_Imag..., U_Real...]``.

    ``prob`` is the cached mesh-only FE problem (``_run._FEProblem``): its ``space``,
    material fields, stiffness ``forms`` and per-component dof maps are reused across cells
    of the same mesh shape. Only this cell's Dirichlet BCs and per-frequency matrix values
    are rebuilt here.

    RF is the nodal reaction on the primary drive nodeset: after solving for ``u``, compute
    ``f_int = K_full * u`` (K assembled without BC modification) and sum it at the
    drive-nodeset dofs per component, mirroring ABAQUS's ``RF`` summed over the drive
    nodeset. ``U`` is the solved displacement summed over the drive nodeset per component.

    Raises
    ------
    NotImplementedError
        If any spatial component is unconstrained (free-lateral cell): the stiffness matrix would be
        singular (rigid-body-translation null space) and the solved displacement non-unique, so the
        oracle U values cannot be reproduced.
    """
    model = sim.model
    dim = model.nodes.dim

    # Guard: reject free-lateral cells (RBM null space → singular K, non-unique U).
    free = unconstrained_components(model, sim.steps[0])
    if free:
        raise NotImplementedError(
            f"free-lateral standard cell: component(s) {free} have no Dirichlet constraint; "
            "the stiffness matrix is singular (rigid-body-translation null space) and the "
            "solved displacement is non-unique -- oracle U values cannot be reproduced"
        )

    space, matfields, forms = prob.space, prob.matfields, prob.forms
    if prob.vc_spaces is None:  # per-component collapsed spaces + dof maps (shape-only)
        prob.vc_spaces, prob.inv_maps = _build_dof_maps(space)
    Vc_spaces, inv_maps = prob.vc_spaces, prob.inv_maps

    # Parse BCs -- values are real and don't change with frequency
    dirichlet_bcs, drive_nodes = _parse_bcs(sim, space, Vc_spaces, inv_maps)

    drive_flat = drive_dofs(space, drive_nodes)

    # The matrices/vectors/KSP have mesh-only sparsity (Dirichlet BCs only zero rows and set
    # the diagonal, which is always allocated), so they are cached on ``prob`` and reused
    # across cells of this shape; only the values (this cell's BCs + frequency moduli) are
    # reassembled in solve_one.
    if prob.std_mats is None:
        import ufl

        matfields.set_moduli(1.0)  # placeholder to establish sparsity
        A_bc = fempetsc.assemble_matrix(forms.a_form, bcs=dirichlet_bcs)
        A_bc.assemble()
        # reaction = action of the unreduced form on the solution: f_int = K_full * u,
        # as one vector assembly per frequency instead of a second matrix assembly
        u_sol = fem.Function(space.V)
        react_form = fem.form(ufl.action(forms.a_ufl, u_sol))
        ksp = lu_ksp(A_bc, space.mesh.comm)
        prob.std_mats = (A_bc, A_bc.createVecRight(), A_bc.createVecRight(),
                         u_sol, react_form, A_bc.createVecRight(), ksp)
    A_bc, b, x, u_sol, react_form, f_int, ksp = prob.std_mats

    def solve_one(f, elastic=False):
        # elastic=True: a Static step -> real *Elastic moduli (zero loss); else the
        # frequency-domain complex moduli at f. Reassemble in place.
        if elastic:
            matfields.set_moduli_elastic()
        else:
            matfields.set_moduli(f)
        A_bc.zeroEntries()
        fempetsc.assemble_matrix(A_bc, forms.a_form, bcs=dirichlet_bcs)
        A_bc.assemble()

        # Build RHS: zero body forces + lifting for non-homogeneous Dirichlet
        b.set(0.0)
        fempetsc.apply_lifting(b, [forms.a_form], [dirichlet_bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fempetsc.set_bc(b, dirichlet_bcs)
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Solve
        ksp.setOperators(A_bc)
        ksp.solve(b, x)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        u_arr = x.getArray()

        # Reaction forces: K_full * u at drive nodeset dofs, via the action form
        u_sol.x.array[:] = u_arr
        with f_int.localForm() as fl:
            fl.set(0.0)
        fempetsc.assemble_vector(f_int, react_form)
        f_int.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        f_arr = f_int.getArray()

        RF = np.zeros(dim, dtype=complex)
        U = np.zeros(dim, dtype=complex)
        for c, flat_dofs in enumerate(drive_flat):
            RF[c] = f_arr[flat_dofs].sum()
            U[c] = u_arr[flat_dofs].sum()

        return [float(f)] + list(RF.real) + list(RF.imag) + list(U.real)

    return solve_one, dim
