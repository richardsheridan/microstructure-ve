"""Parse the solver-neutral spec dataclasses into plain numpy arrays for the FE solve.

This module is **pure numpy** (no dolfinx), so it is importable and unit-testable under
the numpy-only msve env. It is the seam between ``microstructure_ve``'s dataclasses and
the dolfinx boilerplate in ``_assembly``/``_constraints``/``_solver``/``_homogenize``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microstructure_ve.boundary import (
    DisplacementBoundaryCondition,
    FixedBoundaryCondition,
    PeriodicBoundaryCondition,
)
from microstructure_ve.core import _node_array
from microstructure_ve.steps import Dynamic


def find(items, cls):
    """First element of ``items`` that is an instance of ``cls`` (or None)."""
    for it in items:
        if isinstance(it, cls):
            return it
    return None


def default_frequencies(sim):
    """Log-spaced excitation frequencies from the Dynamic subsection of the first step."""
    dyn = find(sim.steps[0].subsections, Dynamic)
    if dyn is None:
        raise ValueError("no Dynamic subsection found in the first step")
    return np.logspace(np.log10(dyn.f_initial), np.log10(dyn.f_final), dyn.f_count)


def frequencies(sim):
    """The points the sweep is evaluated at -- the single entry point used by ``run`` and the
    ``can_run`` capability check. Currently the ``Dynamic`` sweep of the first step; a
    ``Static``-only (elastic) sim has no Dynamic and is not yet supported (raises)."""
    return default_frequencies(sim)


def drive_displacement(sim):
    """The applied drive displacement amplitude (real part) from any step."""
    for step in sim.steps:
        sub = find(step.subsections, DisplacementBoundaryCondition)
        if sub is not None:
            return float(np.real(sub.displacement))
    raise ValueError(
        "no drive found: put a DisplacementBoundaryCondition giving the applied x "
        "displacement in a step's `subsections` (model-level bcs are not read for the "
        "drive amplitude)"
    )


def material_cell_maps(model):
    """Per-cell material index, per-material Poisson ratios, and complex_modulus fns.

    ``mat_of_cell[c]`` is the material index of cell ``c`` (raveled pixel order, the FE
    cell order before dolfinx's reordering). ``modulus_fns[m]`` is material ``m``'s
    ``complex_modulus`` callable.
    """
    ncells = int(np.prod([d - 1 for d in model.nodes.shape]))
    mat_of_cell = np.empty(ncells, dtype=int)
    poissons, modulus_fns = [], []
    for mi, mat in enumerate(model.materials):
        mat_of_cell[np.asarray(mat.elset.elements) - 1] = mi
        poissons.append(mat.poisson)
        modulus_fns.append(mat.complex_modulus)
    return mat_of_cell, np.array(poissons), modulus_fns


def periodic_pairs(shape):
    """(slave_node, master_node) 1-indexed pairs for fluctuation periodicity.

    Slave = any node with at least one grid index at the max face; master = the same
    node with every max index mapped to 0. Slaves are disjoint (each max-face node
    once); masters never sit on a max face, so no master is also a slave. Reduces to the
    2D edge+corner pairing.
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
    return list(zip(all_nodes[is_slave].tolist(), master_nodes[is_slave].tolist()))


@dataclass
class Geometry:
    """Macro geometry for x-uniaxial loading: axis lengths, cross-section, drive strain.

    ``shape`` is the node grid ``(.., ny+1, nx+1)`` so coordinate axis ``i`` maps to grid
    axis ``dim - 1 - i``. ``L = [Lx, Ly, (Lz)]``; ``cross_area`` is perpendicular to the
    x drive; ``exx = drive_displacement / Lx`` (delta == exx * Lx).
    """

    dim: int
    scale: float
    shape: tuple
    L: list
    Lx: float
    cross_area: float
    exx: float

    @classmethod
    def from_model(cls, model, sim):
        nodes = model.nodes
        dim = nodes.dim
        scale = nodes.scale
        shape = nodes.shape
        L = [(shape[dim - 1 - i] - 1) * scale for i in range(dim)]
        cross_area = float(np.prod(L[1:]))
        return cls(dim, scale, shape, L, L[0], cross_area, drive_displacement(sim) / L[0])


def require_periodic(model):
    """Raise unless the model carries a PeriodicBoundaryCondition (the FE backend needs it)."""
    if find(model.bcs, PeriodicBoundaryCondition) is None:
        raise ValueError("the dolfinx backend requires a PeriodicBoundaryCondition")


def require_x_uniaxial(sim):
    """Raise unless the loading is a single x-normal drive (the only mode implemented).

    The homogenization hard-codes the driven axis as x (dof 1) and reports the x-face
    reaction. A model whose step prescribes anything else -- a y/z normal drive, a shear
    (off-axis dof), or compression's several simultaneous normal drives -- is not yet
    supported; reject it here rather than silently returning the x-uniaxial answer. This
    is the red/green tripwire: when a richer drive lands, relax this guard.
    """
    drives = [
        s
        for step in sim.steps
        for s in step.subsections
        if isinstance(s, DisplacementBoundaryCondition)
    ]
    if len(drives) != 1 or drives[0].first_dof != 1 or drives[0].last_dof != 1:
        raise NotImplementedError(
            "the dolfinx backend currently supports only a single x-uniaxial drive "
            "(one DisplacementBoundaryCondition on dof 1); got "
            f"{len(drives)} step drive(s)"
        )


def infer_lateral_bc(model):
    """Read the lateral traction condition off the model's corner BCs.

    The macro loading is x-uniaxial (driven via the X1 reference corner). The lateral
    condition is encoded in whether the lateral reference corner's *normal* DOF is pinned
    by a ``FixedBoundaryCondition``: pinned -> the lateral macro normal strain is held at
    0 (``"confined"``); floating -> the lateral macro normal stress vanishes (``"free"``).
    This keeps the loading a property of the ``Simulation`` rather than a backend kwarg.

    The lateral reference corners (and their normal DOFs) are ``X0Y1`` dof 2 in 2D and
    ``X0Y1Z0`` dof 2 + ``X0Y0Z1`` dof 3 in 3D. Raises if the lateral DOFs are constrained
    inconsistently (some pinned, some free), which is neither pure confined nor pure free.
    """
    nodes = model.nodes
    laterals = [("X0Y1", 2)] if nodes.dim == 2 else [("X0Y1Z0", 2), ("X0Y0Z1", 3)]

    pinned = set()
    for bc in model.bcs:
        if isinstance(bc, FixedBoundaryCondition):
            for ind in np.ravel(_node_array(bc.node)).tolist():
                for dof in bc.dofs:
                    pinned.add((int(ind), int(dof)))

    flags = []
    for key, normal_dof in laterals:
        corner = np.ravel(_node_array(nodes.nsets[key])).tolist()
        flags.append(any((int(ind), normal_dof) in pinned for ind in corner))

    if all(flags):
        return "confined"
    if not any(flags):
        return "free"
    raise ValueError(
        "cannot infer lateral_bc: the lateral reference-corner normal DOFs are "
        "constrained inconsistently (neither all pinned nor all free)"
    )
