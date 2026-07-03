"""Parametrized feature-matrix builder + structural ``.inp`` parser for the test suite.

This is the single source of truth for the perturbation-mode x traction x boundary-
condition x test-type matrix exercised by ``test_matrix_abaqus`` (structural assertions
on the emitted ``.inp``) and ``test_matrix_dolfinx`` (the red/green FE harness).

The macro loading is imposed exactly as the rest of the package does it: for periodic
models by driving the PBC **reference corners** (``X1Y0``/``X0Y1``/``X0Y0Z1`` ...), for
standard (non-periodic) models by prescribing displacement directly on whole faces. Only
*physically meaningful* (mode, traction) cells are enumerated -- see ``matrix_cells`` and
the pruning rule in the module docs of the plan.

Nothing here imports dolfinx, so it is usable under the numpy-only env.
"""
import io

import numpy as np

from microstructure_ve.backends.abaqus import write_inp
from microstructure_ve.boundary import (
    BoundaryCondition,
    Fixed,
    PeriodicBoundaryConstraint,
    Prescribed,
)
from microstructure_ve.constitutive import Elastic, Plastic, TabularViscoelastic
from microstructure_ve.core import ElementSet, GridElements, GridNodes, NodeSet
from microstructure_ve.materials import Material
from microstructure_ve.steps import Dynamic, Heading, Model, Simulation, Static, Step
from microstructure_ve.utils import load_viscoelasticity

from tests._helpers import DISPLACEMENT, PMMA_DATA, SCALE

# Tiny realistic density (kg/micron^3): keeps ABAQUS's steady-state inertia term (-w^2 M)
# negligible against the quasi-static FE backend across the whole sweep, so the oracle
# comparison isn't polluted by an inertia offset at the top frequency.
DENSITY = 2.65e-15

# ---------------------------------------------------------------------------
# axis literals
# ---------------------------------------------------------------------------
MODES = (
    "uniaxial_x", "uniaxial_y", "uniaxial_z",
    "shear_xy", "shear_yz", "shear_xz", "compression",
)
TRACTIONS = ("free", "no_slip", "confined_slip")
BCS = ("periodic", "standard")
TEST_TYPES = ("elastic", "viscoelastic", "hyperelastic_plastic", "viscoelastic_transient")
STUB_TEST_TYPES = ("viscoelastic_transient",)
DIMS = (2, 3)

# Multi-step viscoelastic test types: each maps to an ordered step pattern ("S" = a *STATIC
# step, "D" = a *STEADY STATE DYNAMICS, PERTURBATION step). They exercise multi-step .inp
# emission and the ODB reader's multi-step iteration. Appended (via matrix_cases) only to
# the single cell below, not crossed over the whole matrix.
MULTISTEP_PATTERNS = {
    "viscoelastic_sd": ("S", "D"),
    "viscoelastic_ds": ("D", "S"),
    "viscoelastic_sdsd": ("S", "D", "S", "D"),
}
MULTISTEP_CELL = {"mode": "uniaxial_x", "traction": "free", "bc": "periodic", "dim": 2}

# J2-plasticity cells: ``hyperelastic_plastic`` is crossed over the whole matrix (every cell),
# like elastic/viscoelastic. A *monotonic-hardening* table (yield rises 40->70 MPa) is path-
# independent, so the FE return map matches ABAQUS for the single-step cells. The plastic drive
# is much smaller than the elastic cells' DISPLACEMENT (~33% strain) so the equivalent plastic
# strain lands *inside* the table (not on the perfectly-plastic plateau).
PLASTIC_YIELD_STRESS = [40.0, 70.0]      # MPa, monotonic hardening
PLASTIC_PLASTIC_STRAIN = [0.0, 0.05]
PLASTIC_DISPLACEMENT = 5.0e-4

# Multi-step *plastic* test types on one robust periodic cell. Unlike the viscoelastic patterns
# these carry a per-step drive *scale*: a plastic Static step (``"S"``) drives ``value*scale``
# (so a reversal/cyclic pattern exercises cross-step plastic hysteresis), while Dynamic steps
# (``"D"``) and the first step stay at the reference (+1) drive. ``_sd``/``_ds`` exercise step
# ordering + dispatch; ``_reversal``/``_cyclic`` exercise persistent plastic state across steps.
PLASTIC_MULTISTEP_CELL = {"mode": "uniaxial_x", "traction": "confined_slip",
                          "bc": "periodic", "dim": 2}
PLASTIC_MULTISTEP_PATTERNS = {
    "hyperelastic_plastic_sd": (("S", 1.0), ("D", 1.0)),
    "hyperelastic_plastic_ds": (("D", 1.0), ("S", 1.0)),
    "hyperelastic_plastic_reversal": (("S", 1.0), ("S", -1.0)),
    "hyperelastic_plastic_cyclic": (("S", 1.0), ("S", -1.0), ("S", 1.0)),
}

_AXES = ("x", "y", "z")


def _is_viscoelastic(test_type):
    """True for the frequency-domain (tabular material + Dynamic) test types."""
    return test_type == "viscoelastic" or test_type in MULTISTEP_PATTERNS


def _is_plastic(test_type):
    """True for any J2-plasticity test type (single-step or a plastic multistep ordering)."""
    return test_type == "hyperelastic_plastic" or test_type in PLASTIC_MULTISTEP_PATTERNS


def _resolve_displacement(test_type, displacement):
    """Plastic cells use their own (smaller) drive; everything else keeps ``displacement``."""
    if _is_plastic(test_type) and displacement == DISPLACEMENT:
        return PLASTIC_DISPLACEMENT
    return displacement


def _step_plan(test_type):
    """Ordered ``[(step_kind, drive_scale)]`` for ``test_type`` ('S'=Static, 'D'=Dynamic)."""
    if test_type in ("elastic", "hyperelastic_plastic"):
        return [("S", 1.0)]
    if test_type == "viscoelastic":
        return [("D", 1.0)]
    if test_type in MULTISTEP_PATTERNS:
        return [(kind, 1.0) for kind in MULTISTEP_PATTERNS[test_type]]
    if test_type in PLASTIC_MULTISTEP_PATTERNS:
        return list(PLASTIC_MULTISTEP_PATTERNS[test_type])
    raise ValueError(f"unknown test_type {test_type!r}")


# ---------------------------------------------------------------------------
# the pruned matrix
# ---------------------------------------------------------------------------
def _kept_mode_tractions(dim):
    """Physically meaningful ``(mode, traction)`` pairs for ``dim`` (the pruning rule).

    ``free``/``confined_slip`` describe the lateral normal faces -> normal modes only;
    ``no_slip`` constrains the displacement faces' tangential slip -> shear modes only;
    ``compression`` drives every face (no free lateral) -> ``confined_slip`` only.
    """
    normals = ["uniaxial_x", "uniaxial_y"] + (["uniaxial_z"] if dim == 3 else [])
    shears = ["shear_xy"] + (["shear_yz", "shear_xz"] if dim == 3 else [])
    pairs = []
    for m in normals:
        pairs += [(m, "free"), (m, "confined_slip")]
    pairs += [("compression", "confined_slip")]
    for m in shears:
        pairs += [(m, "no_slip")]
    return pairs


def matrix_cells():
    """Yield ``{mode, traction, bc, dim}`` for every kept physical cell (both bc, both dim)."""
    for dim in DIMS:
        for mode, traction in _kept_mode_tractions(dim):
            for bc in BCS:
                yield {"mode": mode, "traction": traction, "bc": bc, "dim": dim}


def matrix_cases():
    """Yield ``(cell, test_type)`` for the whole suite: every cell x {elastic, viscoelastic,
    hyperelastic_plastic}, plus the appended multi-step viscoelastic cases on ``MULTISTEP_CELL``
    and the multi-step plastic orderings on ``PLASTIC_MULTISTEP_CELL``.

    The single enumerator every consumer (abaqus structural / dolfinx red / parity / oracle
    generation) iterates, so the multi-step cases are added in exactly one place.
    """
    for cell in matrix_cells():
        for test_type in ("elastic", "viscoelastic", "hyperelastic_plastic"):
            yield cell, test_type
    for test_type in MULTISTEP_PATTERNS:
        yield dict(MULTISTEP_CELL), test_type
    for test_type in PLASTIC_MULTISTEP_PATTERNS:
        yield dict(PLASTIC_MULTISTEP_CELL), test_type


def _validate_cell(mode, traction, bc, dim):
    if dim not in DIMS:
        raise ValueError(f"unsupported dim {dim!r}")
    if bc not in BCS:
        raise ValueError(f"unknown bc {bc!r}")
    if (mode, traction) not in _kept_mode_tractions(dim):
        raise ValueError(
            f"pruned/invalid cell: ({mode!r}, {traction!r}) is not a meaningful "
            f"combination in {dim}D"
        )


def is_fe_green(cell, test_type):
    """True iff the DOLFINx backend can solve this cell today.

    Derived from the backend's own (pure-numpy) capability check, so support widens
    automatically as the backend's loading parse / frequency resolution grow -- no per-
    feature edits here. Everything else stays red (the expected-failure side of the harness).
    """
    from microstructure_ve.backends.dolfinx import _loading

    return _loading.can_run(matrix_simulation(test_type=test_type, **cell))


# ---------------------------------------------------------------------------
# mode -> drive specification
# ---------------------------------------------------------------------------
def _drive_specs(mode, dim):
    """Ordered ``[(axis_letter, drive_dof)]``; the first entry is the primary drive.

    Normal mode along axis ``a`` -> drive ``R_a`` in dof ``a``. Shear in plane ``(a, b)``
    -> drive ``R_a`` in dof ``b`` (engineering simple shear). Compression -> drive every
    ``R_a`` in dof ``a``.
    """
    table = {
        "uniaxial_x": [("x", 1)],
        "uniaxial_y": [("y", 2)],
        "uniaxial_z": [("z", 3)],
        "shear_xy": [("x", 2)],
        "shear_yz": [("y", 3)],
        "shear_xz": [("x", 3)],
        "compression": [("x", 1), ("y", 2)] + ([("z", 3)] if dim == 3 else []),
    }
    spec = table[mode]
    needed = max(dof for _, dof in spec)
    if needed > dim:
        raise ValueError(f"mode {mode!r} needs {needed}D, got {dim}D")
    return spec


def _origin_key(dim):
    return "X0Y0" if dim == 2 else "X0Y0Z0"


def _ref_corner_key(dim, axis_letter):
    if dim == 2:
        return {"x": "X1Y0", "y": "X0Y1"}[axis_letter]
    return {"x": "X1Y0Z0", "y": "X0Y1Z0", "z": "X0Y0Z1"}[axis_letter]


def _normal_dof(axis_letter):
    return _AXES.index(axis_letter) + 1


# ---------------------------------------------------------------------------
# corner-driven (periodic) macro BCs -- the general engine
# ---------------------------------------------------------------------------
def _macro_corner_bcs_general(nodes, mode, traction):
    """Corner BCs imposing ``mode`` under ``traction`` for a periodic model.

    Origin pinned. A normal mode drives ``R_a`` in dof ``a`` with its transverse dofs
    pinned; lateral reference corners are fully pinned (``confined_slip``) or have their
    normal dof floating (``free``). A shear mode drives ``R_a`` in the shear dof and pins
    everything else (no-slip simple shear). Returns ``(bcs, [(drive_nset, drive_dof)])``.
    """
    dim = nodes.dim
    axes = _AXES[:dim]
    all_dofs = list(range(1, dim + 1))
    drives = _drive_specs(mode, dim)
    driven_axes = {a for a, _ in drives}

    fixed = {}

    def addfix(key, dofs):
        fixed.setdefault(key, set()).update(dofs)

    addfix(_origin_key(dim), all_dofs)
    if mode.startswith("shear"):
        a, ddof = drives[0]
        addfix(_ref_corner_key(dim, a), set(all_dofs) - {ddof})
        for b in axes:
            if b != a:
                addfix(_ref_corner_key(dim, b), all_dofs)
    else:
        for b in axes:
            cb = _ref_corner_key(dim, b)
            ndof = _normal_dof(b)
            if b in driven_axes:
                addfix(cb, set(all_dofs) - {ndof})  # drive the normal, pin its transverse
            elif traction == "confined_slip":
                addfix(cb, all_dofs)                # hold the lateral normal at 0
            elif traction == "free":
                addfix(cb, set(all_dofs) - {ndof})  # let the lateral normal float
            else:
                raise ValueError(f"bad traction {traction!r} for normal mode {mode!r}")

    bcs = [BoundaryCondition(nodes.nsets[k], Fixed(sorted(d))) for k, d in fixed.items() if d]
    drive_specs = [(nodes.nsets[_ref_corner_key(dim, a)], dof) for a, dof in drives]
    return bcs, drive_specs


# ---------------------------------------------------------------------------
# standard (non-periodic) face BCs
# ---------------------------------------------------------------------------
def _face_closure(nodes, face):
    """Whole-face NodeSet (interior + bounding edges/vertices) for ``face`` like ``"X1"``."""
    axis_letter, side = face[0], face[1]
    coord_axis = {"X": 0, "Y": 1, "Z": 2}[axis_letter]
    grid_axis = nodes.dim - 1 - coord_axis
    idx = [slice(None)] * nodes.dim
    idx[grid_axis] = -1 if side == "1" else 0
    return NodeSet.from_slice(face + "ALL", tuple(idx), nodes)


def _rbm_pins(nodes, a):
    """Minimal statically-determinate pins removing the rigid-body modes a free-traction
    standard cell leaves once the loaded min-face normal is fixed.

    Fixing the whole loaded min-face in its normal dof already kills translation along ``a``
    and the rotations that move that face along ``a``. What remains is the transverse
    translation(s) and (3D) the rotation about ``a``. Pin the transverse translations at the
    origin corner, and (3D) the residual rotation at a second corner offset along the first
    transverse axis. These pins are statically determinate (they carry ~zero reaction), so
    they fix the RBM offset without perturbing the stress field -- only making ``U`` unique.
    """
    transverse = [b for b in _AXES[:nodes.dim] if b != a]
    pins = [BoundaryCondition(nodes.nsets[_origin_key(nodes.dim)],
                              Fixed([_normal_dof(b) for b in transverse]))]
    if nodes.dim == 3:
        b, c = transverse
        pins.append(BoundaryCondition(nodes.nsets[_ref_corner_key(nodes.dim, b)],
                                      Fixed([_normal_dof(c)])))
    return pins


def _standard_face_bcs(nodes, mode, traction):
    """Direct face-Dirichlet BCs (no PBC). Returns ``(bcs, drive_specs, face_nsets)``."""
    dim = nodes.dim
    axes = _AXES[:dim]
    all_dofs = list(range(1, dim + 1))
    drives = _drive_specs(mode, dim)

    face_nsets = {}

    def face(axis_letter, side):
        name = axis_letter.upper() + side
        if name not in face_nsets:
            face_nsets[name] = _face_closure(nodes, name)
        return face_nsets[name]

    bcs, drive_specs = [], []
    if mode.startswith("shear"):
        a, ddof = drives[0]
        bcs.append(BoundaryCondition(face(a, "0"), Fixed(all_dofs)))  # clamped face
        other = [d for d in all_dofs if d != ddof]
        if other:
            bcs.append(BoundaryCondition(face(a, "1"), Fixed(other)))  # no-slip on driven face
        drive_specs.append((face(a, "1"), ddof))
    elif mode == "compression":
        for b in axes:
            ndof = _normal_dof(b)
            bcs.append(BoundaryCondition(face(b, "0"), Fixed([ndof])))
            drive_specs.append((face(b, "1"), ndof))
    else:  # single-axis normal mode
        a, ddof = drives[0]
        for b in axes:
            if b == a or traction == "confined_slip":
                bcs.append(BoundaryCondition(face(b, "0"), Fixed([_normal_dof(b)])))
        if traction == "free":
            # the lateral faces are traction-free, leaving rigid-body modes; pin them
            # minimally so the BVP is well-posed (see _rbm_pins)
            bcs.extend(_rbm_pins(nodes, a))
        drive_specs.append((face(a, "1"), ddof))
    return bcs, drive_specs, list(face_nsets.values())


# ---------------------------------------------------------------------------
# materials / geometry
# ---------------------------------------------------------------------------
def _checkerboard(n, dim):
    return np.indices((n,) * dim).sum(0) % 2


# The tabular table is defined on this log-uniform grid, and the viscoelastic sweep
# (MATRIX_FREQS) is an exact subset of its nodes. Both ABAQUS (linear-in-f) and the FE
# backend (linear-in-log-f) interpolate the *Viscoelastic table; interpolating AT a node
# returns that node's value under either scheme, so sweeping only on nodes makes the
# ABAQUS-vs-FE comparison interpolation-free (no lin-f/log-f offset). With shift=0 and no
# broadening, the material's table frequencies are exactly ``freq`` (= apply_shift()), so
# the nodes the solvers see are exactly _MATRIX_TABLE_FREQS.
_MATRIX_TABLE_FREQS = np.logspace(-2.0, 2.0, 9)   # 2 nodes/decade over 1e-2..1e2
MATRIX_FREQS = _MATRIX_TABLE_FREQS[::4]           # nodes 0,4,8 -> [1e-2, 1e0, 1e2]
_TABLE_YOUNGS = None


def _matrix_table_youngs():
    """PMMA complex modulus sampled onto _MATRIX_TABLE_FREQS (interpolated once, cached)."""
    global _TABLE_YOUNGS
    if _TABLE_YOUNGS is None:
        freq, youngs_cplx = load_viscoelasticity(PMMA_DATA)
        order = np.argsort(freq)
        lt, lf = np.log10(freq[order]), np.log10(_MATRIX_TABLE_FREQS)
        re = np.interp(lf, lt, youngs_cplx.real[order])
        im = np.interp(lf, lt, youngs_cplx.imag[order])
        _TABLE_YOUNGS = re + 1j * im
    return _TABLE_YOUNGS


def tabular_material(elset, nu):
    """A ``TabularViscoelastic`` response (on ``elset``) off the PMMA master curve.

    Used for ``test_type="viscoelastic"`` cells so the harness exercises the genuine
    frequency-domain (``*Viscoelastic, frequency=TABULAR``) path rather than a flat
    modulus. Its table lives on ``_MATRIX_TABLE_FREQS`` and the sweep ``MATRIX_FREQS`` is a
    subset of those nodes, so ABAQUS and the FE backend evaluate it without interpolating
    (see the note above). ``youngs`` == the table's reference modulus (index 0) so
    ``complex_modulus`` reproduces ``youngs_cplx`` exactly at every node.
    """
    youngs_cplx = _matrix_table_youngs()
    return Material(
        elset, density=1.18e-15,
        response=TabularViscoelastic(
            poisson=nu, youngs=float(youngs_cplx[0].real),
            freq=_MATRIX_TABLE_FREQS, youngs_cplx=youngs_cplx,
        ),
    )


def _build_geometry(dim, n, scale, homogeneous, E, nu, test_type):
    """Mesh + materials. ``viscoelastic`` cells use a tabular phase; ``elastic`` use flat
    elastic phases. Heterogeneous cells mix an elastic filler with the second phase."""
    img = np.zeros((n,) * dim, dtype=int) if homogeneous else _checkerboard(n, dim)
    nodes = GridNodes.from_matl_img(img, scale)
    etype = "CPE4" if dim == 2 else "C3D8"
    elements = GridElements(nodes, type=etype)
    sets = ElementSet.from_matl_img(img)

    def plastic_material(elset, youngs):
        return Material(elset, density=DENSITY,
                        response=Plastic(poisson=nu, youngs=youngs,
                                         yield_stress=PLASTIC_YIELD_STRESS,
                                         plastic_strain=PLASTIC_PLASTIC_STRAIN))

    def second_phase(elset):
        if _is_viscoelastic(test_type):
            return tabular_material(elset, nu)
        if _is_plastic(test_type):
            return plastic_material(elset, 5.0 * E)
        return Material(elset, density=DENSITY, response=Elastic(poisson=nu, youngs=5.0 * E))

    if homogeneous:
        if _is_viscoelastic(test_type):
            materials = [tabular_material(sets[0], nu)]
        elif _is_plastic(test_type):
            materials = [plastic_material(sets[0], E)]
        else:
            materials = [Material(sets[0], density=DENSITY, response=Elastic(poisson=nu, youngs=E))]
    else:
        # an elastic filler + a second (viscoelastic / plastic / stiffer-elastic) phase
        materials = [Material(sets[0], density=DENSITY, response=Elastic(poisson=nu, youngs=E)),
                     second_phase(sets[1])]
    return nodes, elements, materials


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------
def matrix_simulation(mode, traction, bc, dim, test_type="elastic", n=None, scale=SCALE,
                      displacement=DISPLACEMENT, homogeneous=False, E=3000.0, nu=0.3):
    """Build a ``Simulation`` for one matrix cell.

    Raises ``ValueError`` for a pruned/invalid cell and ``NotImplementedError`` for the
    stub test types (so callers can assert the guard). ``homogeneous=True`` builds a
    single-material RVE (for the FE analytic invariants); otherwise a two-material
    checkerboard.
    """
    _validate_cell(mode, traction, bc, dim)
    if test_type in STUB_TEST_TYPES:
        raise NotImplementedError(
            f"stub: {test_type} is not implemented yet. Intended: a time-domain "
            "(transient) viscoelastic step"
        )
    plan = _step_plan(test_type)  # raises ValueError for an unknown test_type
    displacement = _resolve_displacement(test_type, displacement)
    if n is None:
        n = 6 if dim == 2 else 4

    nodes, elements, materials = _build_geometry(dim, n, scale, homogeneous, E, nu, test_type)
    value = -displacement if mode == "compression" else displacement

    if bc == "periodic":
        corner_bcs, drive_specs = _macro_corner_bcs_general(nodes, mode, traction)
        base_bcs = [PeriodicBoundaryConstraint(nodes=nodes)] + corner_bcs
        extra_nsets = ()
    else:
        face_bcs, drive_specs, face_nsets = _standard_face_bcs(nodes, mode, traction)
        base_bcs = list(face_bcs)
        extra_nsets = face_nsets

    baselines = [BoundaryCondition(ns, Prescribed([d], 0.0)) for ns, d in drive_specs]
    model = Model(nodes=nodes, elements=elements, materials=materials,
                  bcs=base_bcs + baselines, nsets=extra_nsets)

    def static_step(dr):
        return Step(subsections=[Static()] + dr, perturbation=False)

    def dynamic_step(dr):
        # sweep exactly on the tabular table nodes (see MATRIX_FREQS) -> interpolation-free
        dyn = Dynamic(f_initial=float(MATRIX_FREQS[0]), f_final=float(MATRIX_FREQS[-1]),
                      f_count=len(MATRIX_FREQS), bias=1)
        return Step(subsections=[dyn] + dr, perturbation=True)

    # one step per (kind, drive_scale): a plastic reversal/cyclic pattern drives each Static step
    # at value*scale (so plastic state accumulates/reverses across steps).
    make_step = {"S": static_step, "D": dynamic_step}
    steps = []
    for kind, scale in plan:
        dr = [BoundaryCondition(ns, Prescribed([d], value * scale)) for ns, d in drive_specs]
        steps.append(make_step[kind](dr))

    heading = Heading(f"matrix {dim}d {mode} {traction} {bc} {test_type}")
    return Simulation(heading=heading, model=model, steps=steps)


# ---------------------------------------------------------------------------
# expected emitted structure (kept beside the builder so they can't drift apart)
# ---------------------------------------------------------------------------
def cell_expectations(mode, traction, bc, dim, test_type, displacement=DISPLACEMENT):
    """The structural facts ``test_matrix_abaqus`` asserts on the emitted ``.inp``.

    Returns ``{steps, has_periodic, drive_name, drive_dof, drive_value, fixed_checks}`` where
    ``steps`` is the ordered list of ``(step_type, perturbation)`` the ``.inp`` should emit
    and ``fixed_checks`` is a list of ``(nset_name, dof, should_be_fixed)``.
    """
    drives = _drive_specs(mode, dim)
    a0, dof0 = drives[0]
    displacement = _resolve_displacement(test_type, displacement)
    value = -displacement if mode == "compression" else displacement

    if bc == "periodic":
        drive_name = _ref_corner_key(dim, a0)
        has_periodic = True
        checks = []
        if mode.startswith("shear"):
            # the loaded corner's normal dof is clamped (no-slip simple shear)
            checks.append((_ref_corner_key(dim, a0), _normal_dof(a0), True))
        else:
            driven = {a for a, _ in drives}
            for b in _AXES[:dim]:
                if b in driven:
                    continue
                checks.append((_ref_corner_key(dim, b), _normal_dof(b),
                               traction == "confined_slip"))
    else:
        drive_name = a0.upper() + "1ALL"
        has_periodic = False
        checks = []
        if mode.startswith("shear"):
            checks.append((a0.upper() + "0ALL", _normal_dof(a0), True))
        elif mode == "compression":
            for b in _AXES[:dim]:
                checks.append((b.upper() + "0ALL", _normal_dof(b), True))
        else:
            for b in _AXES[:dim]:
                # min face along a driven/confined axis is a roller; lateral faces under
                # 'free' get no fixed BC at all
                fixed = (b == a0) or (traction == "confined_slip")
                checks.append((b.upper() + "0ALL", _normal_dof(b), fixed))

    step_of = {"S": ("STATIC", False), "D": ("STEADY STATE DYNAMICS", True)}
    pattern = [kind for kind, _scale in _step_plan(test_type)]

    return {
        "steps": [step_of[kind] for kind in pattern],
        "has_periodic": has_periodic,
        "drive_name": drive_name,
        "drive_dof": dof0,
        "drive_value": value,
        "fixed_checks": checks,
    }


# ---------------------------------------------------------------------------
# .inp parser
# ---------------------------------------------------------------------------
def parse_inp(text):
    """Split emitted ``.inp`` text into the keyword blocks the structural tests check."""
    boundary, displacement_boundary = [], []
    n_equation = 0
    steps = []  # ordered [(step_type, perturbation)] per *STEP block
    section = None  # "fixed" | "disp" | None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("*Equation"):
            n_equation += 1
            section = None
        elif line.lower().startswith("*boundary, type=displacement"):
            section = "disp"
        elif line.lower().startswith("*boundary"):
            section = "fixed"
        elif line.upper().startswith("*STEP"):
            steps.append([None, "PERTURBATION" in line.upper()])
            section = None
        elif line.upper().startswith("*STATIC"):
            steps[-1][0] = "STATIC"
            section = None
        elif line.upper().startswith("*STEADY STATE DYNAMICS"):
            steps[-1][0] = "STEADY STATE DYNAMICS"
            section = None
        elif line.startswith("*"):
            section = None
        elif section == "fixed":
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                boundary.append((parts[0], int(parts[1]), int(parts[2])))
        elif section == "disp":
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                displacement_boundary.append(
                    (parts[0], int(parts[1]), int(parts[2]), float(parts[3]))
                )

    return {
        "boundary": boundary,
        "displacement_boundary": displacement_boundary,
        "n_equation": n_equation,
        "has_periodic": n_equation > 0,
        "steps": [tuple(s) for s in steps],
    }


def emit_inp_text(sim):
    """Emit ``sim`` to an in-memory ``.inp`` string."""
    buf = io.StringIO()
    write_inp(sim, buf)
    return buf.getvalue()
