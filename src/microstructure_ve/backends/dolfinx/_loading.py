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
normal drives). Off-diagonal (shear) drives and multi-step sims raise ``NotImplementedError``
until those features land.
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

    if len(list(sim.steps)) != 1:
        raise NotImplementedError("the dolfinx backend supports a single step only")
    L = _axis_lengths(nodes)

    drives = [s for s in sim.steps[0].subsections
              if isinstance(s, DisplacementBoundaryCondition)]
    if len(drives) == 0:
        raise ValueError("no DisplacementBoundaryCondition drive in the step")

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
            raise NotImplementedError(
                "off-diagonal (shear) drive is not supported by the dolfinx backend yet"
            )
        imposed[(a, a)] = float(np.real(d.displacement)) / L[a]
        driven_axes.add(a)
        if primary is None:
            primary = (a, d.first_dof, float(np.real(d.displacement)))

    a0, primary_dof, drive_value = primary

    # lateral normals not driven: free (solved to zero conjugate stress) iff their reference
    # corner's normal dof is unconstrained; otherwise held at 0 (confined)
    free = []
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


def can_run(sim):
    """True iff the dolfinx backend can solve ``sim`` (drives parse + frequencies resolve).

    Pure numpy; the test harness's ``is_fe_green`` calls this, so support widens
    automatically as ``macro_loading`` / the frequency resolution grow.
    """
    try:
        macro_loading(sim)
        spec.frequencies(sim)
    except (NotImplementedError, ValueError):
        return False
    return True
