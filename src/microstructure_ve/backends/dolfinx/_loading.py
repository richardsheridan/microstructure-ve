"""Parse a Simulation into the macro loading the homogenizer imposes (pure numpy).

This is the single seam between the solver-neutral spec and the FE homogenization: it reads
the step drive(s) and corner BCs of a *periodic* model into a ``MacroLoading`` -- which
macro-strain components are driven (and to what value), which float so their conjugate macro
stress vanishes (the generalized lateral condition), and which axis/dof to report so the
emitted row matches the ABAQUS reaction at the drive corner.

Extending the backend to a new loading mode means extending *this* parse; ``can_run`` (and
hence the test harness's ``is_fe_green``) then widens automatically. No dolfinx import here,
so it is usable under the numpy-only env.

Supported now: a single ``Static``/``Dynamic`` step driving one or more *normal* macro
strains via the periodic reference corners (uniaxial along any axis; compression = several
normal drives), and off-diagonal (shear) drives via simple shear. Multi-step sims raise
``NotImplementedError`` until those features land.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from microstructure_ve.boundary import (
    DisplacementBoundaryCondition,
    FixedBoundaryCondition,
)
from microstructure_ve.core import _node_array

from . import _spec as spec


@dataclass
class MacroLoading:
    """The macro strain the FE solve imposes and how to report it.

    ``imposed``/``free`` are keyed by a symmetric component ``(i, j)`` with ``i <= j``
    (tensor strain). ``imposed`` gives driven components and their values; ``free`` lists
    components solved so the conjugate volume-averaged stress is zero. The reported readODB
    row is ``RF = sigma-bar[:, primary_axis] * cross_area`` with ``U[primary_dof-1] =
    drive_value`` (the corner displacement), matching the ABAQUS reaction at the drive corner.
    """

    dim: int
    imposed: Dict[Tuple[int, int], float]
    free: List[Tuple[int, int]]
    primary_axis: int
    primary_dof: int          # 1-indexed dof carrying the reported displacement
    drive_value: float        # corner displacement reported as U[primary_dof-1]
    cross_area: float         # area perpendicular to primary_axis
    active: List[Tuple[int, int]] = field(default_factory=list)  # imposed + free components


def _axis_lengths(nodes):
    dim = nodes.dim
    return [(nodes.shape[dim - 1 - i] - 1) * nodes.scale for i in range(dim)]


def _ref_corner_key(dim, axis):
    """Name of the periodic reference corner R_axis (max on ``axis``, min elsewhere)."""
    letters = "XYZ"[:dim]
    return "".join(f"{letters[i]}{1 if i == axis else 0}" for i in range(dim))


def _corner_axis(nset, dim):
    """The axis whose reference corner ``nset`` is (raises if it is not one)."""
    name = getattr(nset, "name", None)
    for a in range(dim):
        if name == _ref_corner_key(dim, a):
            return a
    raise NotImplementedError(
        f"drive nset {name!r} is not a periodic reference corner; the dolfinx backend only "
        "supports corner-driven periodic loading so far"
    )


def macro_loading(sim):
    """Parse ``sim`` into a ``MacroLoading`` (raises NotImplementedError if unsupported)."""
    model = sim.model
    nodes = model.nodes
    dim = nodes.dim

    L = _axis_lengths(nodes)

    # The macro loading is parsed from the first step carrying a drive. Multi-step cells
    # drive the same macro loading every step, so one MacroLoading describes them all (the
    # per-step analysis type -- Static vs Dynamic -- is handled by the run loop).
    drives = []
    for step in sim.steps:
        drives = [s for s in step.subsections if isinstance(s, DisplacementBoundaryCondition)]
        if drives:
            break
    if len(drives) == 0:
        raise ValueError("no DisplacementBoundaryCondition drive in any step")

    # fixed (node, dof) for the free-lateral inference
    pinned = set()
    for bc in model.bcs:
        if isinstance(bc, FixedBoundaryCondition):
            for ind in np.ravel(_node_array(bc.node)):
                for dof in bc.dofs:
                    pinned.add((int(ind), int(dof)))

    imposed = {}
    driven_axes = set()
    primary = None
    for d in drives:
        if d.first_dof != d.last_dof:
            raise NotImplementedError("only single-dof drives are supported")
        a = _corner_axis(d.nset, dim)
        i = d.first_dof - 1  # 0-indexed component
        if i != a:
            # Off-diagonal (shear) drive: R_a displaced in dof i (= b), imposing H[b,a] = γ.
            # Simple shear: conjugate H[a,b] = 0, so symmetric tensor strain E_{ab} = γ/2.
            b = i
            gamma = float(np.real(d.displacement)) / L[a]
            imposed[(min(a, b), max(a, b))] = gamma / 2.0
            driven_axes.add(a)
            if primary is None:
                primary = (a, d.first_dof, float(np.real(d.displacement)))
            continue
        imposed[(a, a)] = float(np.real(d.displacement)) / L[a]
        driven_axes.add(a)
        if primary is None:
            primary = (a, d.first_dof, float(np.real(d.displacement)))

    a0, primary_dof, drive_value = primary

    # For shear modes: all macro-strain components are fixed via no-slip BCs; free is empty.
    # For normal modes: lateral normals not driven are free (solved to zero conjugate stress)
    # iff their reference corner's normal dof is unconstrained; otherwise held at 0 (confined).
    shear_imposed = any(i != j for i, j in imposed)
    free = []
    if not shear_imposed:
        for b in range(dim):
            if b in driven_axes:
                continue
            node = int(np.ravel(_node_array(nodes.nsets[_ref_corner_key(dim, b)]))[0])
            if (node, b + 1) not in pinned:
                free.append((b, b))

    cross_area = float(np.prod([L[k] for k in range(dim) if k != a0]))
    return MacroLoading(
        dim=dim, imposed=imposed, free=free, primary_axis=a0, primary_dof=primary_dof,
        drive_value=drive_value, cross_area=cross_area,
        active=list(imposed.keys()) + free,
    )


def _standard_can_run(sim):
    """True iff ``sim`` is a well-posed standard (non-periodic, direct-Dirichlet) problem.

    Requirements:
    - Exactly one step with at least one ``DisplacementBoundaryCondition`` drive.
    - Frequencies resolve (Static or Dynamic subsection present).
    - Every spatial component (x, y[, z]) has at least one Dirichlet constraint (either
      a ``FixedBoundaryCondition`` or a step drive) so the stiffness matrix is
      non-singular.  Modes with a free lateral direction (no constraint on a transverse
      component) have a rigid-body-translation null space; those are excluded because
      the quasi-static FE solver (no mass term) does not uniquely determine the
      transverse displacement, so the readODB U columns cannot match the ABAQUS oracle.

    Pure numpy; no dolfinx import.
    """
    model = sim.model
    if len(list(sim.steps)) != 1:
        return False
    step = sim.steps[0]

    # Check frequencies resolve
    try:
        spec.frequencies(sim)
    except (NotImplementedError, ValueError):
        return False

    # Need at least one step drive
    has_drive = any(
        isinstance(s, DisplacementBoundaryCondition)
        for s in step.subsections
    )
    if not has_drive:
        return False

    # Collect constrained components (0-indexed) from model BCs and step drives
    dim = model.nodes.dim
    constrained = set()
    for bc in model.bcs:
        if isinstance(bc, FixedBoundaryCondition):
            for dof in bc.dofs:
                constrained.add(int(dof) - 1)
    for s in step.subsections:
        if isinstance(s, DisplacementBoundaryCondition):
            for dof in range(s.first_dof, s.last_dof + 1):
                constrained.add(int(dof) - 1)

    # Reject if any component is unconstrained (rigid-body translation mode)
    if constrained != set(range(dim)):
        return False

    return True


def can_run(sim):
    """True iff the dolfinx backend can solve ``sim`` (drives parse + frequencies resolve).

    Pure numpy; the test harness's ``is_fe_green`` calls this, so support widens
    automatically as ``macro_loading`` / the frequency resolution grow.

    Handles two cases:
    - *Periodic* sims: validated by ``macro_loading`` (corner-driven PBC path).
    - *Standard* sims (no ``PeriodicBoundaryCondition``): validated by
      ``_standard_can_run`` (direct-Dirichlet path, well-posed cells only).
    """
    from microstructure_ve.boundary import PeriodicBoundaryCondition

    from microstructure_ve.steps import Dynamic, Static

    has_pbc = any(isinstance(bc, PeriodicBoundaryCondition) for bc in sim.model.bcs)
    if has_pbc:
        try:
            macro_loading(sim)
        except (NotImplementedError, ValueError):
            return False
        # every step must be a resolvable analysis (Static or Dynamic); the run loop
        # sweeps each step in turn (multi-step = several such steps)
        return all(
            spec.find(step.subsections, Dynamic) is not None
            or spec.find(step.subsections, Static) is not None
            for step in sim.steps
        )
    else:
        return _standard_can_run(sim)
