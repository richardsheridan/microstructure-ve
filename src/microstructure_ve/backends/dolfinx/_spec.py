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
    PeriodicBoundaryCondition,
)
from microstructure_ve.steps import Dynamic, Static


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
    ``can_run`` capability check. Returns the ``Dynamic`` sweep of the first step if present;
    returns a single nominal frequency ``[1.0]`` for a ``Static``-only (elastic) sim (whose
    ``complex_modulus`` is frequency-flat so any ``f`` yields the elastic modulus); raises
    ``ValueError`` if neither subsection is present."""
    if find(sim.steps[0].subsections, Dynamic) is not None:
        return default_frequencies(sim)
    if find(sim.steps[0].subsections, Static) is not None:
        return np.array([1.0])
    raise ValueError(
        "no Dynamic or Static subsection found in the first step; "
        "cannot determine frequency sweep"
    )


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
    poissons, youngs, modulus_fns = [], [], []
    for mi, mat in enumerate(model.materials):
        mat_of_cell[np.asarray(mat.elset.elements) - 1] = mi
        poissons.append(mat.poisson)
        youngs.append(mat.youngs)  # *Elastic real modulus (used by Static steps)
        modulus_fns.append(mat.complex_modulus)
    return mat_of_cell, np.array(poissons), np.array(youngs), modulus_fns


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
