"""Parse the solver-neutral spec dataclasses into plain numpy arrays for the FE solve.

This module is pure numpy (no dolfinx), so it is importable and unit-testable under the numpy-only
msve env. It is the seam between ``microstructure_ve``'s dataclasses and the dolfinx modules of this
backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microstructure_ve.boundary import BoundaryCondition, Prescribed
from microstructure_ve.steps import Dynamic, Static


def find(items, cls):
    """First element of ``items`` that is an instance of ``cls`` (or None)."""
    for it in items:
        if isinstance(it, cls):
            return it
    return None


def find_drive(items):
    """First ``BoundaryCondition`` in ``items`` carrying a ``Prescribed`` drive (or None)."""
    for it in items:
        if isinstance(it, BoundaryCondition) and isinstance(it.constraint, Prescribed):
            return it
    return None


def sweep_frequencies(dyn):
    """The log-spaced excitation frequencies of the ``Dynamic`` subsection ``dyn``."""
    return np.logspace(np.log10(dyn.f_initial), np.log10(dyn.f_final), dyn.f_count)


def frequencies(sim):
    """The frame values of a single-step sim's rows.

    Parameters
    ----------
    sim : Simulation
        Only its first step is read.

    Returns
    -------
    numpy.ndarray
        The log-spaced ``Dynamic`` sweep of the first step if it has one, else ``[1.0]`` for a
        ``Static`` first step: the frame value of its one row, which ``run`` solves with the real
        elastic moduli rather than ``complex_modulus`` at 1.0.

    Raises
    ------
    ValueError
        If the first step has neither a ``Dynamic`` nor a ``Static`` subsection.
    """
    dyn = find(sim.steps[0].subsections, Dynamic)
    if dyn is not None:
        return sweep_frequencies(dyn)
    if find(sim.steps[0].subsections, Static) is not None:
        return np.array([1.0])
    raise ValueError(
        "no Dynamic or Static subsection found in the first step; "
        "cannot determine frequency sweep"
    )


def drive_displacement(sim):
    """The real part of the first drive's ``Prescribed`` value, scanning steps in order.

    Raises
    ------
    ValueError
        If no step carries a drive; model-level conditions are not read.
    """
    for step in sim.steps:
        sub = find_drive(step.subsections)
        if sub is not None:
            return float(np.real(sub.constraint.value))
    raise ValueError(
        "no drive found: put a BoundaryCondition with a Prescribed constraint giving the "
        "applied x displacement in a step's `subsections` (model-level bcs are not read "
        "for the drive amplitude)"
    )


def material_cell_maps(model):
    """Per-cell material index and the per-material elastic constants of ``model``.

    Returns
    -------
    mat_of_cell : numpy.ndarray of int
        ``mat_of_cell[c]`` is the material index of cell ``c`` in raveled pixel order, the FE cell
        order before dolfinx's reordering.
    poissons : numpy.ndarray
        Poisson's ratio of each material.
    youngs : numpy.ndarray
        Real ``*Elastic`` Young's modulus of each material, used by ``Static`` steps.
    modulus_fns : list of callable
        ``modulus_fns[m]`` is material ``m``'s ``complex_modulus``.
    """
    ncells = int(np.prod([d - 1 for d in model.nodes.shape]))
    mat_of_cell = np.empty(ncells, dtype=int)
    poissons, youngs, modulus_fns = [], [], []
    for mi, mat in enumerate(model.materials):
        mat_of_cell[np.asarray(mat.elset.elements) - 1] = mi
        poissons.append(mat.response.poisson)
        youngs.append(mat.response.youngs)
        modulus_fns.append(mat.response.complex_modulus)
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
    """Macro geometry of the node grid: dimension, pixel scale, grid shape and axis lengths.

    ``shape`` is the node grid ``(.., ny+1, nx+1)`` so coordinate axis ``i`` maps to grid axis
    ``dim - 1 - i``. ``L = [Lx, Ly, (Lz)]``.
    """

    dim: int
    scale: float
    shape: tuple
    L: list

    @classmethod
    def from_model(cls, model, sim=None):
        """The geometry of ``model``'s node grid. ``sim`` is unused."""
        nodes = model.nodes
        dim = nodes.dim
        scale = nodes.scale
        shape = nodes.shape
        L = [(shape[dim - 1 - i] - 1) * scale for i in range(dim)]
        return cls(dim, scale, shape, L)
