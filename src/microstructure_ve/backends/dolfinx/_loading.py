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
    BoundaryCondition,
    Fixed,
    Prescribed,
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
    # The NON-symmetric macro displacement gradient the corner drives impose: the corner-
    # driven PBC constrains u(X + L_a e_a) - u(X) = column a of H (scaled by L_a), so a
    # corner R_a displaced by ``value`` in dof i gives H[i, a] = value / L_a. All other
    # H entries backed by a pinned corner dof are 0 (the zeros are implicit -- only driven
    # entries appear here). Small-strain consumers keep using the symmetrized ``imposed``
    # (shear E_ab = gamma/2); the finite-strain solver needs the raw H because simple shear
    # (H[b,a] = gamma, H[a,b] = 0) and pure shear differ at finite strain. ``free`` doubles
    # as the free DIAGONAL slots of H (a floating lateral corner normal).
    H_imposed: Dict[Tuple[int, int], float] = field(default_factory=dict)


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


def macro_loading(sim, step=None):
    """Parse ``sim`` into a ``MacroLoading`` (raises NotImplementedError if unsupported).

    By default the drive is read from the first step carrying one; most multi-step cells drive
    the same macro loading every step, so one ``MacroLoading`` describes them all. Pass ``step``
    to parse a *specific* step's drive instead -- needed when steps drive different magnitudes
    (e.g. a harmonic Dynamic step at one amplitude followed by a larger Static plastic load).
    """
    model = sim.model
    nodes = model.nodes
    dim = nodes.dim

    L = _axis_lengths(nodes)

    def _drives(items):
        return [s for s in items
                if isinstance(s, BoundaryCondition) and isinstance(s.constraint, Prescribed)]

    if step is not None:
        drives = _drives(step.subsections)
    else:
        drives = []
        for st in sim.steps:
            drives = _drives(st.subsections)
            if drives:
                break
    if len(drives) == 0:
        raise ValueError("no Prescribed drive in any step")

    # fixed (node, dof) for the free-lateral inference
    pinned = set()
    for bc in model.bcs:
        if isinstance(bc, BoundaryCondition) and isinstance(bc.constraint, Fixed):
            for ind in np.ravel(_node_array(bc.target)):
                for dof in bc.constraint.dofs:
                    pinned.add((int(ind), int(dof)))

    imposed = {}
    H_imposed = {}
    driven_axes = set()
    primary = None
    for d in drives:
        dofs = list(d.constraint.dofs)
        if len(dofs) != 1:
            raise NotImplementedError("only single-dof drives are supported")
        drive_dof = int(dofs[0])
        value = float(np.real(d.constraint.value))
        a = _corner_axis(d.target, dim)
        i = drive_dof - 1  # 0-indexed component
        H_imposed[(i, a)] = value / L[a]  # raw macro gradient: corner R_a moved in dof i
        if i != a:
            # Off-diagonal (shear) drive: R_a displaced in dof i (= b), imposing H[b,a] = γ.
            # Simple shear: conjugate H[a,b] = 0, so symmetric tensor strain E_{ab} = γ/2.
            b = i
            gamma = value / L[a]
            imposed[(min(a, b), max(a, b))] = gamma / 2.0
            driven_axes.add(a)
            if primary is None:
                primary = (a, drive_dof, value)
            continue
        imposed[(a, a)] = value / L[a]
        driven_axes.add(a)
        if primary is None:
            primary = (a, drive_dof, value)

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
        H_imposed=H_imposed,
    )


def _standard_can_run(sim):
    """True iff ``sim`` is a well-posed standard (non-periodic, direct-Dirichlet) problem.

    Requirements:
    - Exactly one step with at least one ``Prescribed`` drive -- OR several such steps
      when every material is hyperelastic (the persistent finite-strain solver re-parses
      each Static step's drives and warm-starts from the committed state); the other
      standard paths are single-step.
    - Frequencies resolve (Static or Dynamic subsection present); for a hyperelastic
      multistep every step must carry a ``Static``.
    - Every spatial component (x, y[, z]) has at least one Dirichlet constraint (either
      a ``Fixed`` boundary condition or a step drive) so the stiffness matrix is
      non-singular.  Modes with a free lateral direction (no constraint on a transverse
      component) have a rigid-body-translation null space; those are excluded because
      the quasi-static FE solver (no mass term) does not uniquely determine the
      transverse displacement, so the readODB U columns cannot match the ABAQUS oracle.

    Pure numpy; no dolfinx import.
    """
    from microstructure_ve.constitutive import (
        ArrudaBoyce, NeoHookean, Polynomial, ReducedPolynomial,
    )
    from microstructure_ve.steps import Static

    model = sim.model
    steps = list(sim.steps)
    if len(steps) != 1:
        all_hyper = all(
            isinstance(m.response, (ArrudaBoyce, ReducedPolynomial, Polynomial,
                                    NeoHookean))
            for m in model.materials
        ) and len(list(model.materials)) > 0
        if not all_hyper:
            return False
        if any(spec.find(step.subsections, Static) is None for step in steps):
            return False
    else:
        # Check frequencies resolve (single-step: Static or Dynamic)
        try:
            spec.frequencies(sim)
        except (NotImplementedError, ValueError):
            return False

    dim = model.nodes.dim
    model_fixed = set()
    for bc in model.bcs:
        if isinstance(bc, BoundaryCondition) and isinstance(bc.constraint, Fixed):
            for dof in bc.constraint.dofs:
                model_fixed.add(int(dof) - 1)

    for step in steps:
        # Every step needs at least one drive, and (with the model Fixed BCs) must leave
        # no component unconstrained (rigid-body translation mode)
        constrained = set(model_fixed)
        has_drive = False
        for s in step.subsections:
            if isinstance(s, BoundaryCondition) and isinstance(s.constraint, Prescribed):
                has_drive = True
                for dof in s.constraint.dofs:
                    constrained.add(int(dof) - 1)
        if not has_drive or constrained != set(range(dim)):
            return False

    return True


def can_run(sim):
    """True iff the dolfinx backend can solve ``sim`` (drives parse + frequencies resolve).

    Pure numpy; the test harness's ``is_fe_green`` calls this, so support widens
    automatically as ``macro_loading`` / the frequency resolution grow.

    Handles two cases:
    - *Periodic* sims: validated by ``macro_loading`` (corner-driven PBC path).
    - *Standard* sims (no ``PeriodicBoundaryConstraint``): validated by
      ``_standard_can_run`` (direct-Dirichlet path, well-posed cells only).
    """
    from microstructure_ve.boundary import PeriodicBoundaryConstraint
    from microstructure_ve.constitutive import (
        ArrudaBoyce, NeoHookean, Polynomial, ReducedPolynomial,
    )
    from microstructure_ve.steps import Dynamic, Static

    # Hyperelastic (finite-strain) sims: the total-Lagrangian solver handles Static steps
    # only -- steady-state dynamics about a nonlinear preload is unsupported -- and every
    # phase must be hyperelastic of the SAME model class (an *Elastic phase under NLGEOM
    # is hypoelastic in ABAQUS, which the total-Lagrangian energy formulation cannot
    # reproduce; different classes would need distinct coefficient layouts per cell).
    hyper = [m.response for m in sim.model.materials
             if isinstance(m.response, (ArrudaBoyce, ReducedPolynomial, Polynomial,
                                        NeoHookean))]
    if hyper:
        if len(hyper) != len(list(sim.model.materials)):
            return False
        if len({type(r) for r in hyper}) != 1:
            return False
        if any(spec.find(step.subsections, Dynamic) is not None for step in sim.steps):
            return False

    has_pbc = any(isinstance(bc, PeriodicBoundaryConstraint) for bc in sim.model.bcs)
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
