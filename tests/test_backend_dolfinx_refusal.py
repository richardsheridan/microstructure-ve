"""DOLFINx ``run`` refuses the sims its own capability check rejects.

``_loading.can_run`` is the backend's statement of what it supports. ``run`` on a sim that
check rejects must raise a refusal naming the reason instead of solving the wrong problem
or failing deep inside a solver: ``NotImplementedError`` for a well-formed sim the backend
does not support, ``ValueError`` for a malformed one. The catalogue below covers each kind
of rejected sim, so a rule the check holds and ``run`` ignores (or the reverse) shows up
as a failing sub-test.
"""
import copy
import unittest

from microstructure_ve.boundary import BoundaryCondition, Fixed, Prescribed
from microstructure_ve.constitutive import ArrudaBoyce, Elastic
from microstructure_ve.steps import Dynamic, Static, Step

from tests._helpers import homogeneous_simulation, needs_dolfinx
from tests._matrix import hold_by_prescribed, matrix_simulation


_N = 4
_STANDARD = {"mode": "uniaxial_x", "traction": "confined_slip", "bc": "standard", "dim": 2}
_PERIODIC = {"mode": "uniaxial_x", "traction": "free", "bc": "periodic", "dim": 2}
_PERIODIC_CONFINED = {"mode": "uniaxial_x", "traction": "confined_slip", "bc": "periodic",
                      "dim": 2}


def _drives(step):
    return [s for s in step.subsections if isinstance(s, BoundaryCondition)]


def _without_drives(sim):
    for step in sim.steps:
        step.subsections = [s for s in step.subsections if not isinstance(s, BoundaryCondition)]
    return sim


def _without_y_constraint(sim):
    """Drop every model-level bc that fixes only the y component, leaving it unconstrained."""
    sim.model.bcs = [
        b for b in sim.model.bcs
        if not (isinstance(b, BoundaryCondition) and isinstance(b.constraint, Fixed)
                and list(b.constraint.dofs) == [2])
    ]
    return sim


def _with_nonzero_baselines(sim, value):
    """Give every model-level ``Prescribed`` (the zero baselines on the driven dofs) ``value``."""
    sim.model.bcs = [
        BoundaryCondition(b.target, Prescribed(list(b.constraint.dofs), value))
        if isinstance(b, BoundaryCondition) and isinstance(b.constraint, Prescribed) else b
        for b in sim.model.bcs
    ]
    return sim


def _without_drives_in_step(sim, index):
    """Remove step ``index``'s own drives, leaving the other steps as built."""
    step = sim.steps[index]
    step.subsections = [s for s in step.subsections if not isinstance(s, BoundaryCondition)]
    return sim


def _driveless_first_step(sim):
    """Put a copy of the first step with no drive of its own ahead of the built steps."""
    sim.steps = [copy.deepcopy(sim.steps[0])] + list(sim.steps)
    return _without_drives_in_step(sim, 0)


def _with_repeated_step(sim):
    """Follow the sim's only step with a copy of it."""
    sim.steps = [sim.steps[0], copy.deepcopy(sim.steps[0])]
    return sim


def _step_driving(sim, index, nset_name, dof):
    """Move step ``index``'s only drive, at its amplitude, to ``dof`` of ``nset_name``."""
    step = sim.steps[index]
    (drive,) = _drives(step)
    moved = BoundaryCondition(sim.model.nodes.nsets[nset_name],
                              Prescribed([dof], drive.constraint.value))
    step.subsections = [moved if s is drive else s for s in step.subsections]
    return sim


def _step_drives_reversed(sim, index):
    """List step ``index``'s drives in reverse order, leaving the rest of the step as built."""
    step = sim.steps[index]
    rest = [s for s in step.subsections if not isinstance(s, BoundaryCondition)]
    step.subsections = rest + _drives(step)[::-1]
    return sim


def _standard_two_static_elastic():
    return _with_repeated_step(matrix_simulation(test_type="elastic", n=_N, **_STANDARD))


def _periodic_hyperelastic_with_elastic_phase():
    sim = matrix_simulation(test_type="reduced_polynomial", n=_N, **_PERIODIC)
    sim.model.materials[1].response = Elastic(poisson=0.3, youngs=1000.0)
    return sim


def _periodic_two_hyperelastic_classes():
    sim = matrix_simulation(test_type="reduced_polynomial", n=_N, **_PERIODIC)
    sim.model.materials[1].response = ArrudaBoyce(mu=1000.0, lm=2.0, poisson=0.45)
    return sim


def _periodic_hyperelastic_then_dynamic():
    sim = matrix_simulation(test_type="reduced_polynomial", n=_N, **_PERIODIC)
    dyn = Dynamic(f_initial=1.0, f_final=1.0, f_count=1, bias=1)
    sim.steps = list(sim.steps) + [
        Step(subsections=[dyn] + _drives(sim.steps[0]), perturbation=True)]
    return sim


def _periodic_drive_off_reference_corner():
    sim = matrix_simulation(test_type="elastic", n=_N, **_PERIODIC)
    _drives(sim.steps[0])[0].target = sim.model.nodes.nsets["X0Y0"]
    return sim


def _periodic_step_without_procedure():
    sim = matrix_simulation(test_type="elastic", n=_N, **_PERIODIC)
    step = sim.steps[0]
    step.subsections = [s for s in step.subsections if not isinstance(s, Static)]
    return sim


def _with_element_type(etype, dim):
    """A homogeneous periodic sim, otherwise supported, whose elements are of type ``etype``.

    The type is set after construction, so a string the spec does not accept still reaches
    the backend.
    """
    sim = homogeneous_simulation(n=3, dim=dim)
    sim.model.elements.type = etype
    return sim


def refused_element_types():
    """``(name, element type, grid dimension)`` for each element type ``run`` refuses."""
    return [
        ("plane stress", "CPS4R", 2),
        ("2D reduced integration", "CPE4R", 2),
        ("3D reduced integration", "C3D8R", 3),
        ("unknown to the spec", "CPE8", 2),
    ]


def rejected_sims():
    """``(name, build, exception type)`` for each kind of sim ``can_run`` rejects."""
    return [
        ("standard two-step plastic",
         lambda: matrix_simulation(test_type="hyperelastic_plastic_reversal", n=_N,
                                   **_STANDARD),
         NotImplementedError),
        ("standard static then dynamic",
         lambda: matrix_simulation(test_type="viscoelastic_sd", n=_N, **_STANDARD),
         NotImplementedError),
        ("standard dynamic then static",
         lambda: matrix_simulation(test_type="viscoelastic_ds", n=_N, **_STANDARD),
         NotImplementedError),
        ("standard two static elastic steps", _standard_two_static_elastic,
         NotImplementedError),
        ("standard component left unconstrained",
         lambda: _without_y_constraint(matrix_simulation(test_type="elastic", n=_N,
                                                         **_STANDARD)),
         NotImplementedError),
        ("periodic hyperelastic with an elastic phase",
         _periodic_hyperelastic_with_elastic_phase, NotImplementedError),
        ("periodic two hyperelastic classes", _periodic_two_hyperelastic_classes,
         NotImplementedError),
        ("periodic hyperelastic with a dynamic step", _periodic_hyperelastic_then_dynamic,
         NotImplementedError),
        ("periodic drive off a reference corner", _periodic_drive_off_reference_corner,
         NotImplementedError),
        ("standard no drive",
         lambda: _without_drives(matrix_simulation(test_type="elastic", n=_N, **_STANDARD)),
         ValueError),
        ("periodic no drive",
         lambda: _without_drives(matrix_simulation(test_type="elastic", n=_N, **_PERIODIC)),
         ValueError),
        ("periodic step with neither static nor dynamic", _periodic_step_without_procedure,
         ValueError),
        ("periodic dynamic step without a drive of its own",
         lambda: _without_drives_in_step(
             matrix_simulation(test_type="viscoelastic_sd", n=_N, **_PERIODIC), 1),
         NotImplementedError),
        ("periodic plastic dynamic step without a drive of its own",
         lambda: _without_drives_in_step(
             matrix_simulation(test_type="hyperelastic_plastic_sd", n=_N,
                               **_PERIODIC_CONFINED), 1),
         NotImplementedError),
        ("periodic later static step drives another axis",
         lambda: _step_driving(
             matrix_simulation(test_type="viscoelastic_halved_ss", n=_N, **_PERIODIC), 1,
             "X0Y1", 2),
         NotImplementedError),
        ("periodic dynamic step drives another axis than the static step",
         lambda: _step_driving(
             matrix_simulation(test_type="viscoelastic_halved_sd", n=_N, **_PERIODIC), 1,
             "X0Y1", 2),
         NotImplementedError),
        ("periodic plastic later static step drives another axis",
         lambda: _step_driving(
             matrix_simulation(test_type="hyperelastic_plastic_reversal", n=_N, **_PERIODIC),
             1, "X0Y1", 2),
         NotImplementedError),
        ("periodic hyperelastic later static step drives another axis",
         lambda: _step_driving(_with_repeated_step(
             matrix_simulation(test_type="reduced_polynomial", n=_N, **_PERIODIC)), 1,
             "X0Y1", 2),
         NotImplementedError),
        ("periodic later step drives the same components in another order",
         lambda: _step_drives_reversed(
             matrix_simulation(test_type="viscoelastic_halved_ss", n=_N, mode="compression",
                               traction="confined_slip", bc="periodic", dim=2), 1),
         NotImplementedError),
        ("periodic later step drives off a reference corner",
         lambda: _step_driving(
             matrix_simulation(test_type="viscoelastic_halved_ss", n=_N, **_PERIODIC), 1,
             "X0Y0", 1),
         NotImplementedError),
        ("periodic driveless first static step",
         lambda: _driveless_first_step(
             matrix_simulation(test_type="elastic", n=_N, **_PERIODIC)),
         ValueError),
        ("periodic plastic driveless first step",
         lambda: _driveless_first_step(
             matrix_simulation(test_type="hyperelastic_plastic", n=_N, **_PERIODIC_CONFINED)),
         ValueError),
        ("periodic hyperelastic driveless first step",
         lambda: _driveless_first_step(
             matrix_simulation(test_type="reduced_polynomial", n=_N, **_PERIODIC)),
         ValueError),
        ("periodic nonzero model-level Prescribed on an undriven dof",
         lambda: hold_by_prescribed(matrix_simulation(test_type="elastic", n=_N,
                                                      **_PERIODIC_CONFINED), "X0Y1", 2, 1e-3),
         ValueError),
        ("periodic imaginary model-level Prescribed on an undriven dof",
         lambda: hold_by_prescribed(matrix_simulation(test_type="elastic", n=_N,
                                                      **_PERIODIC_CONFINED), "X0Y1", 2, 1e-3j),
         ValueError),
        ("standard nonzero model-level Prescribed on an undriven dof",
         lambda: hold_by_prescribed(matrix_simulation(test_type="elastic", n=_N, **_STANDARD),
                                    "Y0ALL", 2, 1e-3),
         ValueError),
        ("standard nonzero model-level Prescribed on a driven dof",
         lambda: _with_nonzero_baselines(
             matrix_simulation(test_type="elastic", n=_N, **_STANDARD), 1e-3),
         ValueError),
    ] + [
        (f"element type {etype} ({name})",
         lambda etype=etype, dim=dim: _with_element_type(etype, dim), NotImplementedError)
        for name, etype, dim in refused_element_types()
    ]


@needs_dolfinx
class RunRefusesUnsupportedTests(unittest.TestCase):
    def assertRefuses(self, sim, exc_type, **kwargs):
        from microstructure_ve.backends.dolfinx import run

        with self.assertRaises(exc_type) as cm:
            run(sim, **kwargs)
        self.assertTrue(str(cm.exception).strip(), "refusal does not name its reason")

    def test_run_refuses_every_sim_can_run_rejects(self):
        """For each kind of sim ``can_run`` rejects, ``run`` raises the refusal for that kind
        and does not solve."""
        from microstructure_ve.backends.dolfinx import _loading

        for name, build, exc_type in rejected_sims():
            with self.subTest(name):
                sim = build()
                self.assertFalse(_loading.can_run(sim), "catalogue sim is accepted")
                self.assertRefuses(sim, exc_type)

    def test_element_type_refusal_names_the_type(self):
        """A sim refused for its element type says which type, so the caller knows what to
        change."""
        from microstructure_ve.backends.dolfinx import run

        for name, etype, dim in refused_element_types():
            with self.subTest(name):
                with self.assertRaises(NotImplementedError) as cm:
                    run(_with_element_type(etype, dim))
                self.assertIn(etype, str(cm.exception))

    def test_parallel_sweep_refuses_in_the_calling_process(self):
        """With ``workers>1`` a rejected multi-frequency sim is refused in the caller with
        the reason, not reported as a broken worker pool."""
        sim = _without_y_constraint(
            matrix_simulation(test_type="viscoelastic", n=_N, **_STANDARD))
        self.assertRefuses(sim, NotImplementedError, workers=2)


if __name__ == "__main__":
    unittest.main()
