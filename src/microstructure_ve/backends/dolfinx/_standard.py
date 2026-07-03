"""Direct-Dirichlet solve path for standard (non-periodic) boundary conditions.

Assembles the B-bar stiffness with complex moduli, imposes Dirichlet BCs from the
simulation's ``Fixed`` boundary conditions (zero) and step ``Prescribed`` drives
(prescribed displacement), solves with PETSc serial LU, then reports the readODB-style
row: ``[f, RF_Real..., RF_Imag..., U_Real...]`` where RF and U are summed over the
primary drive nodeset nodes for each component.

Reaction-force computation: after solving for ``u``, assemble the unreduced stiffness
``K_full`` (no Dirichlet modification) and compute ``f_int = K_full * u``.  The reaction
at each constrained node is the corresponding entry of ``f_int`` (equilibrium gives zero
at free nodes; the residual at constrained nodes is the reaction force). Summing over the
primary drive nodeset per component reproduces the ABAQUS ``readODB`` summation.

No MPC or center-pin -- standard cells are clamped BVPs, not periodic fluctuation
problems.  The solver kind (LU vs iterative crossover) follows the same prediction as
the periodic path; only the LU path is implemented, so large meshes fall back to LU
with a warning (same as the periodic ``build_solver`` does).
"""
from __future__ import annotations

import numpy as np

from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc as fempetsc

from microstructure_ve.boundary import (
    BoundaryCondition,
    Fixed,
    Prescribed,
)
from microstructure_ve.core import _node_array


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
    return fem.dirichletbc(fbc, [flat_dofs, vc_dofs], V.sub(comp))


def _parse_bcs(sim, space, Vc_spaces, inv_maps):
    """Parse the simulation's BCs into (dolfinx_bcs, drive_nodes).

    ``dolfinx_bcs``: list of ``fem.DirichletBC`` for the FE solve.
    ``drive_nodes``: 1-indexed node array of the *primary* drive nodeset (the first
    ``Prescribed`` drive in the step's subsections), used for RF and U summation.

    Model-level ``Fixed`` boundary conditions prescribe zero on their dofs.
    Baseline ``Prescribed`` conditions in ``model.bcs`` (value=0) are skipped -- they
    carry the ABAQUS initial-state convention, and the step drives override them.
    Step-level ``Prescribed`` drives prescribe the actual drive displacement.
    """
    model = sim.model
    bcs = []

    for bc in model.bcs:
        if isinstance(bc, BoundaryCondition) and isinstance(bc.constraint, Fixed):
            nodes = np.ravel(_node_array(bc.target))
            for dof in bc.constraint.dofs:
                bcs.append(_make_dirichlet_bc(
                    space, Vc_spaces, inv_maps, nodes, dof - 1, 0.0
                ))
        # Prescribed conditions in model.bcs are baselines (value=0); skip.

    drive_nodes = None
    for s in sim.steps[0].subsections:
        if isinstance(s, BoundaryCondition) and isinstance(s.constraint, Prescribed):
            nodes = np.ravel(_node_array(s.target))
            if drive_nodes is None:
                drive_nodes = nodes  # first drive = primary (for RF/U reporting)
            disp_value = float(np.real(s.constraint.value))
            for dof in s.constraint.dofs:
                bcs.append(_make_dirichlet_bc(
                    space, Vc_spaces, inv_maps, nodes, dof - 1, disp_value
                ))

    return bcs, drive_nodes


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

    Raises ``NotImplementedError`` if any spatial component is unconstrained (free-lateral
    cell): the stiffness matrix would be singular (rigid-body-translation null space) and
    the solved displacement is non-unique, so the oracle U values cannot be reproduced.
    """
    model = sim.model

    # Guard: reject free-lateral cells (RBM null space → singular K, non-unique U).
    dim = model.nodes.dim
    constrained = set()
    for bc in model.bcs:
        if isinstance(bc, BoundaryCondition) and isinstance(bc.constraint, Fixed):
            for dof in bc.constraint.dofs:
                constrained.add(int(dof) - 1)
    for s in sim.steps[0].subsections:
        if isinstance(s, BoundaryCondition) and isinstance(s.constraint, Prescribed):
            for dof in s.constraint.dofs:
                constrained.add(int(dof) - 1)
    if constrained != set(range(dim)):
        free = [i for i in range(dim) if i not in constrained]
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

    V = space.V
    bs = V.dofmap.index_map_bs
    block_of_node = space.block_of_node
    blocks_drive = block_of_node[drive_nodes]

    # The matrices/vectors/KSP have mesh-only sparsity (Dirichlet BCs only zero rows and set
    # the diagonal, which is always allocated), so they are cached on ``prob`` and reused
    # across cells of this shape; only the values (this cell's BCs + frequency moduli) are
    # reassembled in solve_one.
    if prob.std_mats is None:
        matfields.set_moduli(1.0)  # placeholder to establish sparsity
        A_bc = fempetsc.assemble_matrix(forms.a_form, bcs=dirichlet_bcs)
        A_bc.assemble()
        A_full = fempetsc.assemble_matrix(forms.a_form)  # no BC modification
        A_full.assemble()
        ksp = PETSc.KSP().create(space.mesh.comm)
        ksp.setOperators(A_bc)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        prob.std_mats = (A_bc, A_full, A_bc.createVecRight(),
                         A_bc.createVecRight(), A_full.createVecRight(), ksp)
    A_bc, A_full, b, x, f_int, ksp = prob.std_mats

    def solve_one(f):
        # Update complex moduli and reassemble (in-place: pass A as first arg to dispatch)
        matfields.set_moduli(f)
        A_bc.zeroEntries()
        fempetsc.assemble_matrix(A_bc, forms.a_form, bcs=dirichlet_bcs)
        A_bc.assemble()
        A_full.zeroEntries()
        fempetsc.assemble_matrix(A_full, forms.a_form)
        A_full.assemble()

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

        # Reaction forces: K_full * u at drive nodeset dofs
        A_full.mult(x, f_int)
        f_arr = f_int.getArray()

        RF = np.zeros(dim, dtype=complex)
        U = np.zeros(dim, dtype=complex)
        for c in range(dim):
            flat_dofs = blocks_drive * bs + c
            RF[c] = f_arr[flat_dofs].sum()
            U[c] = u_arr[flat_dofs].sum()

        return [float(f)] + list(RF.real) + list(RF.imag) + list(U.real)

    return solve_one, dim
