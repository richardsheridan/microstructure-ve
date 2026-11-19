"""DOLFINx backend in a real-scalar PETSc build.

A frequency-domain solve needs complex arithmetic. A real build with no complex interpreter
configured refuses it with ``ScalarBuildError`` instead of dropping the loss modulus, and keeps
solving Static steps in real arithmetic. These tests run only in a real build.
"""
import os
import pathlib
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np

from tests._helpers import homogeneous_simulation, needs_real
from tests._matrix import MULTISTEP_CELL, PLASTIC_MULTISTEP_CELL, matrix_simulation

_COMPLEX_VAR = "MSVE_DOLFINX_COMPLEX_PYTHON"
_REAL_VAR = "MSVE_DOLFINX_REAL_PYTHON"
_N = 4
_PERIODIC = MULTISTEP_CELL
_STANDARD = {"mode": "uniaxial_x", "traction": "free", "bc": "standard", "dim": 2}


class _WorkerStarted(Exception):
    """A process was started where none may be."""


@needs_real
class RealBuildTests(unittest.TestCase):
    def setUp(self):
        from microstructure_ve.backends.dolfinx import clear_cache

        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(_COMPLEX_VAR, None)
        os.environ.pop(_REAL_VAR, None)
        clear_cache()
        self.addCleanup(clear_cache)

    def _complex_var_settings(self):
        """Unset, then empty: both mean no complex interpreter is configured."""
        yield "unset"
        os.environ[_COMPLEX_VAR] = ""
        yield "empty"
        os.environ.pop(_COMPLEX_VAR)

    def test_scalar_build_error_is_a_runtime_error_exported_by_the_package(self):
        """``ScalarBuildError`` is importable from the backend package and is caught by an
        ``except RuntimeError``."""
        from microstructure_ve.backends.dolfinx import ScalarBuildError

        self.assertTrue(issubclass(ScalarBuildError, RuntimeError))

    def test_run_refuses_a_sim_with_a_dynamic_step_before_solving_or_writing(self):
        """``run`` on a sim with any Dynamic step raises ``ScalarBuildError`` naming the
        complex-interpreter variable, before any step solves (``cancel`` is never polled) and
        without creating or touching ``output_path``."""
        sims = {
            "periodic dynamic": homogeneous_simulation(n=3, dim=2),
            "standard dynamic": matrix_simulation(test_type="viscoelastic", n=_N, **_STANDARD),
            "periodic static then dynamic": matrix_simulation(
                test_type="viscoelastic_sd", n=_N, **_PERIODIC),
            "periodic dynamic then static": matrix_simulation(
                test_type="viscoelastic_ds", n=_N, **_PERIODIC),
            "periodic plastic static then dynamic": matrix_simulation(
                test_type="hyperelastic_plastic_sd", n=_N, **PLASTIC_MULTISTEP_CELL),
        }
        from microstructure_ve.backends.dolfinx import ScalarBuildError, run

        sentinel = b"written before the call\n"
        for setting in self._complex_var_settings():
            for name, sim in sims.items():
                for existing in (False, True):
                    with self.subTest(name, variable=setting, existing_output=existing), \
                            tempfile.TemporaryDirectory() as tmp:
                        out = pathlib.Path(tmp) / "rows.tsv"
                        if existing:
                            out.write_bytes(sentinel)
                        polls = []
                        with self.assertRaises(ScalarBuildError) as cm:
                            run(sim, output_path=out, cancel=lambda: polls.append(1) or False)
                        self.assertIn(_COMPLEX_VAR, str(cm.exception))
                        self.assertEqual(polls, [], "cancel polled: a step began")
                        if existing:
                            self.assertEqual(out.read_bytes(), sentinel)
                        else:
                            self.assertFalse(out.exists(), "output_path written")

    def test_parallel_sweep_refuses_before_any_worker_process_starts(self):
        """With ``workers>1`` a multi-frequency Dynamic sim is refused in the caller with
        ``ScalarBuildError``, and no worker process is started."""
        from multiprocessing.process import BaseProcess

        sim = homogeneous_simulation(n=3, dim=2, f_count=2)
        from microstructure_ve.backends.dolfinx import ScalarBuildError, run

        starts = []

        def start(process):
            starts.append(process)
            raise _WorkerStarted("a worker process was started")

        with mock.patch.object(BaseProcess, "start", autospec=True, side_effect=start):
            with self.assertRaises(ScalarBuildError) as cm:
                run(sim, workers=2)
        self.assertIn(_COMPLEX_VAR, str(cm.exception))
        self.assertEqual(starts, [])

    def test_solve_one_refuses_frequency_domain_and_solves_elastic(self):
        """``build_solver`` returns for any sim; its ``solve_one(f)`` raises
        ``ScalarBuildError`` naming the complex-interpreter variable, whatever the sim's steps,
        while ``solve_one(f, elastic=True)`` returns a row."""
        sims = {
            "periodic dynamic": homogeneous_simulation(n=3, dim=2),
            "periodic static": matrix_simulation(test_type="elastic", n=_N, **_PERIODIC),
            "standard static": matrix_simulation(test_type="elastic", n=_N, **_STANDARD),
        }
        from microstructure_ve.backends.dolfinx import ScalarBuildError, build_solver

        for setting in self._complex_var_settings():
            for name, sim in sims.items():
                with self.subTest(name, variable=setting):
                    solve_one, dim = build_solver(sim)
                    with self.assertRaises(ScalarBuildError) as cm:
                        solve_one(1.0)
                    self.assertIn(_COMPLEX_VAR, str(cm.exception))
                    row = np.asarray(solve_one(1.0, elastic=True))
                    self.assertEqual(row.shape, (1 + 3 * dim,))
                    self.assertTrue(np.all(np.isfinite(row)))

    def test_static_runs_solve_without_complex_warning(self):
        """A Static run (elastic, plastic or hyperelastic; periodic or standard) returns its
        rows and never casts a complex value into the real build's fields."""
        from numpy.exceptions import ComplexWarning

        from microstructure_ve.backends.dolfinx import clear_cache, run

        for test_type in ("elastic", "hyperelastic_plastic", "reduced_polynomial"):
            for cell in (_PERIODIC, _STANDARD):
                with self.subTest(test_type, bc=cell["bc"]):
                    sim = matrix_simulation(test_type=test_type, n=_N, **cell)
                    clear_cache()  # a cold cache, so every material fill runs
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        rows = run(sim)
                    self.assertEqual(rows.shape, (len(list(sim.steps)), 1 + 3 * cell["dim"]))
                    self.assertTrue(np.all(np.isfinite(rows)))
                    complex_casts = [str(w.message) for w in caught
                                     if issubclass(w.category, ComplexWarning)]
                    self.assertEqual(complex_casts, [], "a complex value was cast to real")


if __name__ == "__main__":
    unittest.main()
