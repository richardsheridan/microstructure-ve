"""DOLFINx backend arithmetic: which PETSc scalar build solves each sim.

A frequency-domain solve needs complex arithmetic. A real build with no complex interpreter
configured refuses it with ``ScalarBuildError`` instead of dropping the loss modulus, and keeps
solving Static steps in real arithmetic. When the other build's interpreter is named by its
variable, ``run`` hands a sim this build should not solve to a peer process running that
interpreter, and returns the peer's rows as its own. A complex build with no real interpreter
configured solves Static steps itself and warns.
"""
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from unittest import mock

import numpy as np

from tests._helpers import (
    HAS_COMPLEX,
    homogeneous_simulation,
    needs_complex,
    needs_dolfinx,
    needs_real,
)
from tests._matrix import MULTISTEP_CELL, PLASTIC_MULTISTEP_CELL, matrix_simulation

_COMPLEX_VAR = "MSVE_DOLFINX_COMPLEX_PYTHON"
_REAL_VAR = "MSVE_DOLFINX_REAL_PYTHON"
_N = 4
_PERIODIC = MULTISTEP_CELL
_STANDARD = {"mode": "uniaxial_x", "traction": "free", "bc": "standard", "dim": 2}
_STATIC_TYPES = ("elastic", "hyperelastic_plastic", "reduced_polynomial")

needs_real_peer = unittest.skipUnless(
    HAS_COMPLEX and os.environ.get(_REAL_VAR),
    f"needs a complex-scalar PETSc build with {_REAL_VAR} naming a real-build Python")


class _WorkerStarted(Exception):
    """A process was started where none may be."""


class _StopRequested(Exception):
    """Raised by a ``cancel`` callback to stop a run with its own exception."""


def _solve_events():
    """The LU factorizations and back-substitutions PETSc has logged in this process."""
    from petsc4py import PETSc

    return sum(PETSc.Log.Event(name).getPerfInfo()["count"]
               for name in ("MatLUFactorNum", "MatSolve"))


def _assert_round_off(actual, expected):
    """``actual`` equals ``expected`` to round-off, relative to the largest entry."""
    expected = np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-12,
                               atol=1e-12 * np.nanmax(np.abs(expected)))


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
        """With no complex interpreter configured, ``run`` on a sim with any Dynamic step raises ``ScalarBuildError`` naming the
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


def _routed_case():
    """The variable naming the other build's interpreter, and a sim this build hands to it."""
    if HAS_COMPLEX:
        return _REAL_VAR, matrix_simulation(test_type="elastic", n=_N, **_STANDARD)
    return _COMPLEX_VAR, homogeneous_simulation(n=3, dim=2)


@needs_dolfinx
class PeerConfigurationTests(unittest.TestCase):
    """A variable that names no usable interpreter of the other build is reported with
    ``ScalarBuildError`` naming the variable and its value, after the call's own arguments are
    checked."""

    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(_COMPLEX_VAR, None)
        os.environ.pop(_REAL_VAR, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)

    def _assert_refused(self, var, path, sim):
        from microstructure_ve.backends.dolfinx import ScalarBuildError, run

        with self.assertRaises(ScalarBuildError) as cm:
            run(sim)
        self.assertIn(var, str(cm.exception))
        self.assertIn(str(path), str(cm.exception))

    def test_missing_interpreter_is_reported_after_argument_checks(self):
        """A path that does not exist or cannot be executed raises ``ScalarBuildError``; a call
        whose own arguments are bad raises their ``ValueError`` instead, before any interpreter
        is tried."""
        from microstructure_ve.backends.dolfinx import run

        var, sim = _routed_case()
        not_executable = self.tmp / "python"
        not_executable.write_text("")
        not_executable.chmod(0o644)
        bad_arguments = {"threads": {"threads": 0}, "solver": {"solver": "bogus"},
                         "return_energy": {"return_energy": True}}
        for path in (self.tmp / "no-such-python", not_executable):
            os.environ[var] = str(path)
            for name, kwargs in bad_arguments.items():
                with self.subTest(path=path.name, bad_argument=name):
                    with self.assertRaises(ValueError):
                        run(sim, **kwargs)
            with self.subTest(path=path.name):
                self._assert_refused(var, path, sim)

    def test_interpreter_that_is_not_the_other_build_is_refused(self):
        """An interpreter of this process's own scalar build, or a program that exits without
        answering, raises ``ScalarBuildError``."""
        var, sim = _routed_case()
        paths = {"same build": sys.executable}
        if shutil.which("false"):
            paths["exits at once"] = shutil.which("false")
        for name, path in paths.items():
            with self.subTest(name):
                os.environ[var] = path
                self._assert_refused(var, path, sim)


@needs_complex
class ComplexBuildWithoutRealPeerTests(unittest.TestCase):
    """With no real-build interpreter configured, the complex build solves Static steps in
    complex arithmetic and says so with a ``ScalarBuildWarning`` naming the variable."""

    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(_COMPLEX_VAR, None)
        os.environ.pop(_REAL_VAR, None)

    def _real_var_settings(self):
        """Unset, then empty: both mean no real interpreter is configured."""
        yield "unset"
        os.environ[_REAL_VAR] = ""
        yield "empty"
        os.environ.pop(_REAL_VAR)

    def test_static_solve_warns_once_naming_the_real_interpreter_variable(self):
        """Static runs from different call sites, under Python's default warning filter, emit
        one ``ScalarBuildWarning`` (a ``UserWarning``) naming the variable; a fresh warning
        context, and a mixed Static and Dynamic sim, see it again."""
        from microstructure_ve.backends.dolfinx import ScalarBuildWarning, run

        self.assertTrue(issubclass(ScalarBuildWarning, UserWarning))
        elastic = matrix_simulation(test_type="elastic", n=_N, **_STANDARD)
        hyperelastic = matrix_simulation(test_type="reduced_polynomial", n=_N, **_PERIODIC)
        mixed = matrix_simulation(test_type="viscoelastic_sd", n=_N, **_PERIODIC)
        for setting in self._real_var_settings():
            with self.subTest(variable=setting):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("default")
                    first = run(elastic)
                    second = run(hyperelastic)
                found = [str(w.message) for w in caught
                         if issubclass(w.category, ScalarBuildWarning)]
                self.assertEqual(len(found), 1, found)
                self.assertIn(_REAL_VAR, found[0])
                self.assertTrue(np.all(np.isfinite(first)) and np.all(np.isfinite(second)))
                with self.assertWarns(ScalarBuildWarning) as cm:
                    run(mixed)
                self.assertIn(_REAL_VAR, str(cm.warning))

    def test_solves_without_a_static_step_in_run_do_not_warn(self):
        """A Dynamic-only ``run``, and ``build_solver`` with ``solve_one(f, elastic=True)``,
        emit no ``ScalarBuildWarning``."""
        from microstructure_ve.backends.dolfinx import ScalarBuildWarning, build_solver, run

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run(homogeneous_simulation(n=3, dim=2))
            solve_one, _ = build_solver(
                matrix_simulation(test_type="elastic", n=_N, **_STANDARD))
            solve_one(1.0, elastic=True)
        found = [str(w.message) for w in caught if issubclass(w.category, ScalarBuildWarning)]
        self.assertEqual(found, [])


_CHILD = """
import pickle, sys
from petsc4py import PETSc
from microstructure_ve.backends.dolfinx import run
from microstructure_ve.backends.dolfinx._arithmetic import is_complex_scalar
if is_complex_scalar(PETSc.ScalarType):
    sys.exit("the child interpreter is not the real-scalar build")
with open(sys.argv[1], "rb") as f:
    sims = pickle.load(f)
rows = [run(sim) for sim in sims]
with open(sys.argv[2], "wb") as f:
    pickle.dump(rows, f)
"""

# A routed scenario run in a complex-build child, so a deadlock ends with the child, not the suite.
# Arguments: the pickled sims ``(long_sim, small_sim)``, the output pickle, the scenario name.
_CONCURRENT_CHILD = """
import os, pickle, sys, threading, time
from petsc4py import PETSc
from microstructure_ve.backends.dolfinx import clear_cache, run
PETSc.Log.begin()
EVENTS = [PETSc.Log.Event(name) for name in ("MatLUFactorNum", "MatSolve")]
JOIN_S = 120
with open(sys.argv[1], "rb") as f:
    long_sim, small_sim = pickle.load(f)
out = {}
before = sum(e.getPerfInfo()["count"] for e in EVENTS)

def call(key, sim, **kwargs):
    try:
        out[key] = run(sim, **kwargs)
    except BaseException as e:
        out[key + "_error"] = type(e).__name__ + ": " + str(e)

def two_threads_one_peer():
    a_polled, b_started = threading.Event(), threading.Event()
    b_start = []

    def cancel_a():
        a_polled.set()
        return b_started.is_set() and time.monotonic() - b_start[0] > 1.0

    def b():
        b_start.append(time.monotonic())
        b_started.set()
        call("b", small_sim)

    a = threading.Thread(target=call, args=("a", long_sim),
                         kwargs=dict(n_incr=30, cancel=cancel_a), daemon=True)
    a.start()
    a_polled.wait(JOIN_S)
    time.sleep(1.0)
    b = threading.Thread(target=b, daemon=True)
    b.start()
    return [a, b]

def clear_cache_in_cancel():
    def cancel():
        clear_cache()
        return False

    t = threading.Thread(target=call, args=("small", small_sim), kwargs=dict(cancel=cancel),
                         daemon=True)
    t.start()
    return [t]

threads = globals()[sys.argv[3]]()
deadline = time.monotonic() + JOIN_S
for t in threads:
    t.join(max(0.0, deadline - time.monotonic()))
out["hung"] = any(t.is_alive() for t in threads)
out["events"] = sum(e.getPerfInfo()["count"] for e in EVENTS) - before
with open(sys.argv[2], "wb") as f:
    pickle.dump(out, f)
os._exit(0)
"""

# A real-build child that cancels a routed workers>1 sweep once the complex peer's pool has
# started, then waits for every process it had below it to go. Arguments: the pickled sim, the
# output pickle, and a file that receives the recorded pids at the moment of cancelling.
_POOL_CHILD = """
import os, pickle, sys, time
from petsc4py import PETSc
from microstructure_ve.backends.dolfinx import run
from microstructure_ve.backends.dolfinx._arithmetic import is_complex_scalar
if is_complex_scalar(PETSc.ScalarType):
    sys.exit("the child interpreter is not the real-scalar build")
ME = os.getpid()
GONE_S = 30

def stat(pid):
    # (parent pid, state), or None once /proc/<pid> is gone.
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
    except OSError:
        return None
    return int(fields[1]), fields[0]

def process_table():
    table = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            entry = stat(name)
            if entry is not None:
                table[int(name)] = entry
    return table

def descendants(table):
    found, frontier = set(), {ME}
    while frontier:
        frontier = {pid for pid, (ppid, _) in table.items() if ppid in frontier} - found
        found |= frontier
    return found

recorded = []

def cancel():
    table = process_table()
    below = descendants(table)
    if not any(table[pid][0] != ME for pid in below):
        return False
    recorded.extend(sorted(below))
    with open(sys.argv[3], "w") as f:
        f.write(" ".join(map(str, recorded)))
    return True

def leaked(pid):
    # A zombie of this process is gone for good once this process reaps it; any other zombie,
    # like any live process, has been left behind.
    entry = stat(pid)
    return entry is not None and not (entry[1] == "Z" and entry[0] == ME)

with open(sys.argv[1], "rb") as f:
    sim = pickle.load(f)
error = None
try:
    run(sim, workers=2, cancel=cancel)
except BaseException as e:
    error = type(e).__name__ + ": " + str(e)
deadline = time.monotonic() + GONE_S
while [pid for pid in recorded if leaked(pid)] and time.monotonic() < deadline:
    time.sleep(0.2)
survivors = {pid: stat(pid) for pid in recorded if leaked(pid)}
with open(sys.argv[2], "wb") as f:
    pickle.dump({"error": error, "recorded": recorded, "survivors": survivors}, f)
"""


@needs_real_peer
class RealPeerTests(unittest.TestCase):
    """In the complex build, a sim with no Dynamic step is solved by the real-build peer named
    by the variable, and ``run`` answers as a native run would."""

    @classmethod
    def setUpClass(cls):
        from petsc4py import PETSc

        PETSc.Log.begin()

    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(_COMPLEX_VAR, None)
        self.real_python = os.environ[_REAL_VAR]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)

    def _native(self, sim, **kwargs):
        """``run`` in this process with no real interpreter configured."""
        from microstructure_ve.backends.dolfinx import run

        with mock.patch.dict(os.environ), warnings.catch_warnings():
            os.environ.pop(_REAL_VAR)
            warnings.simplefilter("ignore")
            before = _solve_events()
            out = run(sim, **kwargs)
        self.assertGreater(_solve_events(), before, "a native solve logged no PETSc events")
        return out

    def _routed(self, sim, **kwargs):
        """``run`` with the real interpreter configured: no solve in this process, no warning
        about the real interpreter."""
        from microstructure_ve.backends.dolfinx import run

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            before = _solve_events()
            out = run(sim, **kwargs)
        self.assertEqual(_solve_events() - before, 0, "the routed sim solved in this process")
        self.assertEqual([str(w.message) for w in caught if _REAL_VAR in str(w.message)], [])
        return out

    def test_static_runs_through_the_real_peer_match_native_runs(self):
        """Static sims (elastic, plastic, hyperelastic; periodic and standard) return the native
        rows to round-off, and ``output_path`` holds them in the native file format."""
        for test_type in _STATIC_TYPES:
            for cell in (_PERIODIC, _STANDARD):
                with self.subTest(test_type, bc=cell["bc"]):
                    sim = matrix_simulation(test_type=test_type, n=_N, **cell)
                    native_file = self.tmp / f"{test_type}-{cell['bc']}-native.tsv"
                    routed_file = self.tmp / f"{test_type}-{cell['bc']}-routed.tsv"
                    native = self._native(sim, output_path=native_file)
                    routed = self._routed(sim, output_path=routed_file)
                    self.assertEqual(routed.dtype, np.float64)
                    _assert_round_off(routed, native)
                    self.assertEqual(routed_file.read_text().splitlines()[0],
                                     native_file.read_text().splitlines()[0])
                    np.testing.assert_allclose(np.loadtxt(routed_file, skiprows=1, ndmin=2),
                                               routed, rtol=1e-7)

    def test_return_energy_crosses_the_real_peer(self):
        """A hyperelastic Static sim with ``return_energy=True`` returns the native rows and
        energies to round-off."""
        for cell in (_PERIODIC, _STANDARD):
            with self.subTest(bc=cell["bc"]):
                sim = matrix_simulation(test_type="reduced_polynomial", n=_N, **cell)
                native_rows, native_energies = self._native(sim, return_energy=True)
                rows, energies = self._routed(sim, return_energy=True)
                _assert_round_off(rows, native_rows)
                _assert_round_off(energies, native_energies)

    def test_convergence_error_from_the_real_peer_keeps_its_type(self):
        """A Static sim whose Newton solve fails in the peer raises ``ConvergenceError`` in the
        caller with the peer's traceback attached as a note, and the next routed run solves."""
        from microstructure_ve.backends.dolfinx import ConvergenceError, run

        sim = matrix_simulation(mode="compression", traction="confined_slip", bc="periodic",
                                dim=2, test_type="reduced_polynomial", displacement=0.0145)
        before = _solve_events()
        with self.assertRaises(ConvergenceError) as cm:
            run(sim)
        self.assertEqual(_solve_events() - before, 0, "the routed sim solved in this process")
        self.assertTrue(getattr(cm.exception, "__notes__", None), "no peer traceback attached")
        elastic = matrix_simulation(test_type="elastic", n=_N, **_STANDARD)
        _assert_round_off(self._routed(elastic), self._native(elastic))

    def test_cancel_stops_a_routed_solve_and_the_next_routed_run_solves(self):
        """A ``cancel`` that turns truthy, or raises, while the peer solves a long Static step
        stops the run with ``Cancelled`` or the callback's own exception, and the next routed
        run returns the native rows."""
        from microstructure_ve.backends.dolfinx import Cancelled, run

        long_sim = matrix_simulation(test_type="reduced_polynomial", mode="uniaxial_x",
                                     traction="free", bc="periodic", dim=3, n=10)
        elastic = matrix_simulation(test_type="elastic", n=_N, **_STANDARD)
        native = self._native(elastic)

        def stop_after(delay, stop):
            start = []

            def cancel():
                start.append(time.monotonic())
                return time.monotonic() - start[0] > delay and stop()
            return cancel

        def raise_own():
            raise _StopRequested("stopped by the caller")

        for expected, stop in ((Cancelled, lambda: True), (_StopRequested, raise_own)):
            with self.subTest(expected.__name__):
                before = _solve_events()
                with self.assertRaises(expected):
                    run(long_sim, n_incr=30, cancel=stop_after(2.0, stop))
                self.assertEqual(_solve_events() - before, 0,
                                 "the routed sim solved in this process")
                _assert_round_off(self._routed(elastic), native)

    def _concurrent_child(self, scenario):
        """Run ``scenario`` of ``_CONCURRENT_CHILD`` in a complex-build child configured as this
        process is; return its outcomes, and the native rows of its small sim."""
        long_sim = matrix_simulation(test_type="reduced_polynomial", mode="uniaxial_x",
                                     traction="free", bc="periodic", dim=3, n=10)
        small_sim = matrix_simulation(test_type="elastic", n=_N, **_STANDARD)
        native = self._native(small_sim)
        sims_file, out_file = self.tmp / "sims.pkl", self.tmp / "out.pkl"
        sims_file.write_bytes(pickle.dumps((long_sim, small_sim)))
        log_file = self.tmp / "child.txt"
        with open(log_file, "wb") as log:
            proc = subprocess.run(
                [sys.executable, "-c", _CONCURRENT_CHILD, str(sims_file), str(out_file),
                 scenario],
                env=dict(os.environ), cwd=self.tmp, stdin=subprocess.DEVNULL, stdout=log,
                stderr=log, timeout=900)
        self.assertEqual(proc.returncode, 0, log_file.read_text())
        out = pickle.loads(out_file.read_bytes())
        self.assertFalse(out["hung"], f"a routed run did not return: {out}")
        self.assertEqual(out["events"], 0, "a routed sim solved in the child")
        return out, native

    def test_cancel_of_one_thread_leaves_a_run_queued_on_the_same_peer_solving(self):
        """While one thread's routed run of a long sim is in flight, a second thread's routed
        run queues on the same peer; cancelling the first raises ``Cancelled`` there, and the
        second still returns the native rows."""
        out, native = self._concurrent_child("two_threads_one_peer")
        self.assertIn("a_error", out, "the long run was not stopped")
        self.assertTrue(out["a_error"].startswith("Cancelled"), out["a_error"])
        self.assertNotIn("b_error", out)
        _assert_round_off(out["b"], native)

    def test_clear_cache_from_a_routed_runs_own_cancel_returns(self):
        """A routed run whose ``cancel`` calls ``clear_cache`` and returns False completes with
        the native rows instead of waiting on its own peer."""
        out, native = self._concurrent_child("clear_cache_in_cancel")
        self.assertNotIn("small_error", out)
        _assert_round_off(out["small"], native)

    def test_peer_from_another_source_tree_is_refused(self):
        """A real interpreter that imports ``microstructure_ve`` from another source tree
        raises ``ScalarBuildError`` naming the variable and the path."""
        import microstructure_ve

        from microstructure_ve.backends.dolfinx import ScalarBuildError, run

        package = pathlib.Path(microstructure_ve.__file__).resolve().parent
        shutil.copytree(package, self.tmp / "src" / package.name,
                        ignore=shutil.ignore_patterns("__pycache__"))
        wrapper = self.tmp / "python"
        wrapper.write_text(
            "#!/bin/sh\n"
            f'PYTHONPATH="{self.tmp / "src"}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
            "export PYTHONPATH\n"
            f'exec "{self.real_python}" "$@"\n')
        wrapper.chmod(0o755)
        os.environ[_REAL_VAR] = str(wrapper)
        with self.assertRaises(ScalarBuildError) as cm:
            run(matrix_simulation(test_type="elastic", n=_N, **_STANDARD))
        self.assertIn(_REAL_VAR, str(cm.exception))
        self.assertIn(str(wrapper), str(cm.exception))

    def test_real_build_caller_solves_dynamic_sims_through_the_complex_peer(self):
        """A real-build process with the complex interpreter configured returns, for a
        Dynamic-only and a Static-then-Dynamic sim, the rows this complex process computes."""
        sims = [homogeneous_simulation(n=3, dim=2, f_count=2),
                matrix_simulation(test_type="viscoelastic_sd", n=_N, **_PERIODIC)]
        native = [self._native(sim) for sim in sims]
        sims_file, rows_file = self.tmp / "sims.pkl", self.tmp / "rows.pkl"
        sims_file.write_bytes(pickle.dumps(sims))
        env = dict(os.environ)
        env.pop(_REAL_VAR)
        env[_COMPLEX_VAR] = sys.executable
        stderr_file = self.tmp / "stderr.txt"
        with open(stderr_file, "wb") as stderr:
            proc = subprocess.run(
                [self.real_python, "-c", _CHILD, str(sims_file), str(rows_file)],
                env=env, cwd=self.tmp, stdin=subprocess.DEVNULL, stdout=stderr, stderr=stderr,
                timeout=600)
        self.assertEqual(proc.returncode, 0, stderr_file.read_text())
        routed = pickle.loads(rows_file.read_bytes())
        self.assertEqual(len(routed), len(sims))
        for got, want in zip(routed, native):
            _assert_round_off(got, want)

    def test_cancelled_parallel_sweep_through_the_complex_peer_leaves_no_process(self):
        """A real-build process cancels a routed ``workers=2`` sweep once the complex peer's pool
        has started: ``run`` raises ``Cancelled`` and every process below the caller at that
        moment is gone within a bounded wait."""
        import signal

        sim = homogeneous_simulation(n=8, dim=2, f_count=40)
        sims_file, out_file, pids_file = (self.tmp / name
                                          for name in ("sim.pkl", "out.pkl", "pids.txt"))
        sims_file.write_bytes(pickle.dumps(sim))
        complex_exe = os.path.realpath(sys.executable)

        def kill_leftovers():
            if not pids_file.exists():
                return
            for pid in map(int, pids_file.read_text().split()):
                try:
                    if os.path.realpath(f"/proc/{pid}/exe") == complex_exe:
                        os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        self.addCleanup(kill_leftovers)

        env = dict(os.environ)
        env.pop(_REAL_VAR)
        env[_COMPLEX_VAR] = sys.executable
        log_file = self.tmp / "child.txt"
        with open(log_file, "wb") as log:
            proc = subprocess.run(
                [self.real_python, "-c", _POOL_CHILD, str(sims_file), str(out_file),
                 str(pids_file)],
                env=env, cwd=self.tmp, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                timeout=600)
        self.assertEqual(proc.returncode, 0, log_file.read_text())
        out = pickle.loads(out_file.read_bytes())
        self.assertIsNotNone(out["error"], "the sweep was not stopped")
        self.assertTrue(out["error"].startswith("Cancelled"), out["error"])
        self.assertTrue(out["recorded"], "cancel never saw the peer's pool")
        self.assertEqual(out["survivors"], {}, "processes outlived the cancelled sweep")


if __name__ == "__main__":
    unittest.main()
