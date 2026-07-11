"""DOLFINx FE backend tests (require the fenicsx env; skip cleanly otherwise).

Tests 1-4 are self-contained analytic checks needing no ABAQUS: mesh/dof construction,
the periodic MPC, and homogeneous closed-form uniaxial stresses. ABAQUS-parity tests
live in test_backend_dolfinx_parity once the oracles are committed.
"""
import unittest

import numpy as np

from tests._helpers import (
    SCALE,
    homogeneous_simulation,
    oracle_simulation_prony,
    synthetic_simulation,
)

try:
    import dolfinx  # noqa: F401
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

needs_dolfinx = unittest.skipUnless(HAS_DOLFINX, "needs the fenicsx env (dolfinx)")


def _lame(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def _worker_threadpool_counts():
    """Run inside a spawned worker (top-level so it pickles): the per-pool thread counts
    threadpoolctl reports after ``_init_worker`` has applied its cap."""
    import threadpoolctl

    return [pool["num_threads"] for pool in threadpoolctl.threadpool_info()]


@needs_dolfinx
class AssemblyTests(unittest.TestCase):
    def test_mesh_total_measure_exact(self):
        from microstructure_ve.backends.dolfinx import _assembly as assembly, _spec as spec

        for dim, n in [(2, 4), (3, 3)]:
            sim = homogeneous_simulation(n=n, dim=dim)
            geom = spec.Geometry.from_model(sim.model, sim)
            space = assembly.Space.build(geom)
            self.assertAlmostEqual(space.area, (n * SCALE) ** dim, places=12)

    def test_dg0_assignment_round_trips_microstructure(self):
        # a 2-material checkerboard: mu field must reflect each cell's modulus after the
        # original_cell_index remap (naive order would scramble the microstructure)
        from microstructure_ve.backends.dolfinx import _assembly as assembly, _spec as spec
        from microstructure_ve.core import ElementSet, GridElements, GridNodes
        from microstructure_ve.constitutive import Elastic
        from microstructure_ve.materials import Material
        from microstructure_ve.steps import Model
        import numpy as np

        img = np.array([[0, 1], [1, 0]])
        nodes = GridNodes.from_matl_img(img, SCALE)
        elements = GridElements(nodes, type="CPE4")
        e0, e1 = ElementSet.from_matl_img(img)
        E0, E1, nu = 1000.0, 5000.0, 0.3
        model = Model(nodes=nodes, elements=elements,
                      materials=[Material(e0, 1.0, Elastic(nu, E0)),
                                 Material(e1, 1.0, Elastic(nu, E1))])
        # build a minimal space + fields (no full solver needed)
        from microstructure_ve.backends.dolfinx._spec import Geometry
        geom = Geometry(2, SCALE, nodes.shape, [2 * SCALE, 2 * SCALE], 2 * SCALE, 2 * SCALE, 1.0)
        space = assembly.Space.build(geom)
        mats = assembly.MaterialFields.from_model(space, model)
        mats.set_moduli(1.0)
        mu = mats.mu_fn.x.array.real
        expected = {E0 / (2 * (1 + nu)), E1 / (2 * (1 + nu))}
        self.assertEqual(set(np.round(np.unique(mu), 6)), {round(x, 6) for x in expected})
        # the two materials each own two cells
        counts = sorted(np.bincount(np.searchsorted(sorted(expected), mu)).tolist())
        self.assertEqual(counts, [2, 2])


@needs_dolfinx
class AssemblyBundlesAreDataclassesTests(unittest.TestCase):
    def test_space_materialfields_forms_are_dataclasses(self):
        import dataclasses

        from microstructure_ve.backends.dolfinx import _assembly as assembly, _spec as spec

        sim = homogeneous_simulation(n=3, dim=2)
        geom = spec.Geometry.from_model(sim.model, sim)
        space = assembly.Space.build(geom)
        mats = assembly.MaterialFields.from_model(space, sim.model)
        forms = assembly.Forms.build(space, mats, bbar=True)
        for obj in (space, mats, forms):
            self.assertTrue(dataclasses.is_dataclass(obj))
        # closures became methods
        self.assertTrue(callable(space.coord))
        self.assertTrue(callable(mats.set_moduli))


@needs_dolfinx
class NodeDofMapTests(unittest.TestCase):
    def test_node_dof_round_trip(self):
        from microstructure_ve.backends.dolfinx import _assembly as assembly, _spec as spec

        sim = homogeneous_simulation(n=3, dim=2)
        nodes = sim.model.nodes
        geom = spec.Geometry.from_model(sim.model, sim)
        space = assembly.Space.build(geom)
        # every boundary node set's coords recover its grid position
        for name, nset in nodes.nsets.items():
            for node in nset.node_inds:
                xy = space.coord(node)
                grid = np.rint(xy / SCALE).astype(int)
                self.assertTrue(np.all(grid >= 0) and np.all(grid < nodes.shape[::-1]))


@needs_dolfinx
class PeriodicMpcTests(unittest.TestCase):
    def test_mpc_builds_and_enforces_periodicity(self):
        # a homogeneous confined solve has ~zero fluctuation; a correct MPC keeps the
        # solve consistent and yields the exact analytic stress (checked below). Here we
        # just assert the MPC finalizes with the expected number of slaves.
        from microstructure_ve.backends.dolfinx import (
            _assembly as assembly, _constraints as constraints, _spec as spec)

        sim = homogeneous_simulation(n=3, dim=2)
        geom = spec.Geometry.from_model(sim.model, sim)
        space = assembly.Space.build(geom)
        mpc = constraints.periodic_mpc(space)
        n_pairs = len(spec.periodic_pairs(space.shape))
        # one scalar constraint per component per slave node
        self.assertEqual(mpc.num_local_slaves, n_pairs * space.dim)


@needs_dolfinx
class AnalyticHomogeneousTests(unittest.TestCase):
    def _sigma(self, sim):
        # lateral_bc is now read off the Simulation's corner BCs, not passed in
        from microstructure_ve.backends.dolfinx import _run as run, _spec as spec

        geom = spec.Geometry.from_model(sim.model, sim)
        row = run.run(sim)[0]
        dim = sim.model.nodes.dim
        rf_real = np.array(row[1:1 + dim])
        return rf_real / geom.cross_area, geom.exx  # sigma-bar normal row 0..dim-1

    def test_confined_plane_strain(self):
        E, nu = 3000.0, 0.3
        lam, mu = _lame(E, nu)
        sim = homogeneous_simulation(n=4, dim=2, E=E, nu=nu, lateral_bc="confined")
        sigma, exx = self._sigma(sim)
        self.assertAlmostEqual(sigma[0], (lam + 2 * mu) * exx, delta=abs(sigma[0]) * 1e-6)

    def test_free_lateral_plane_strain(self):
        E, nu = 3000.0, 0.3
        lam, mu = _lame(E, nu)
        sim = homogeneous_simulation(n=4, dim=2, E=E, nu=nu, lateral_bc="free")
        sigma, exx = self._sigma(sim)
        expected = 4 * mu * (lam + mu) / (lam + 2 * mu) * exx
        self.assertAlmostEqual(sigma[0], expected, delta=abs(expected) * 1e-6)
        self.assertAlmostEqual(sigma[1], 0.0, delta=abs(expected) * 1e-6)  # lateral free

    def test_free_lateral_3d_is_uniaxial_stress(self):
        E, nu = 3000.0, 0.3
        sim = homogeneous_simulation(n=3, dim=3, E=E, nu=nu, lateral_bc="free")
        sigma, exx = self._sigma(sim)
        self.assertAlmostEqual(sigma[0], E * exx, delta=abs(E * exx) * 1e-6)
        self.assertAlmostEqual(sigma[1], 0.0, delta=abs(E * exx) * 1e-6)
        self.assertAlmostEqual(sigma[2], 0.0, delta=abs(E * exx) * 1e-6)


@needs_dolfinx
class PronyViscoelasticBehaviorTests(unittest.TestCase):
    """The silent-correctness fix: a Prony material must produce a frequency-varying,
    lossy homogenized modulus -- not the flat elastic the base ``complex_modulus`` gave.
    """

    def _homogenized_Ex(self, sim):
        from microstructure_ve.backends.dolfinx import _run as run, _spec as spec

        geom = spec.Geometry.from_model(sim.model, sim)
        out = run.run(sim)
        dim = sim.model.nodes.dim
        rf = out[:, 1:1 + dim] + 1j * out[:, 1 + dim:1 + 2 * dim]
        return rf[:, 0] / (geom.cross_area * geom.exx)  # confined M ~ (lam+2mu) ∝ E*(f)

    def test_prony_modulus_is_frequency_varying_and_lossy(self):
        # wide sweep so both the relaxed and glassy plateaus are reached (tau=1s)
        sim = oracle_simulation_prony(f_initial=1e-6, f_final=1e6, f_count=25)
        Ex = self._homogenized_Ex(sim)
        storage, loss = Ex.real, Ex.imag
        # storage stiffens from relaxed toward glassy across the sweep (not flat-elastic)
        self.assertGreater(storage[-1], 1.5 * storage[0])
        # a genuine loss peak in the transition, vanishing at both plateaus
        self.assertGreater(loss.max(), 0.05 * storage.mean())
        self.assertLess(loss[0], 1e-3 * storage[0])
        self.assertLess(loss[-1], 1e-3 * storage[-1])
        self.assertTrue(np.all(loss >= -1e-6 * storage.mean()))


@needs_dolfinx
class FrequencyParallelTests(unittest.TestCase):
    def test_workers_match_serial(self):
        # frequency-varying (viscoelastic) sim so the parallel split is non-trivial
        from microstructure_ve.backends.dolfinx import _run as run

        # frequency sweep + traction now come from the Simulation itself
        sim = synthetic_simulation()
        serial = run.run(sim, workers=1)
        parallel = run.run(sim, workers=2)
        np.testing.assert_allclose(parallel, serial, rtol=1e-9, atol=0)

    def test_concurrent_runs_from_threads_match_serial(self):
        # The runner-level thread-safety guarantee: run() may be called from any number
        # of threads; calls serialize on the backend lock and each returns exactly what a
        # serial call would. Same mesh shape on purpose -- every thread contends for the
        # one cached _FEProblem, whose material refill + reassembly is the shared state
        # an unlocked race would corrupt (different E per sim makes corruption visible).
        from concurrent.futures import ThreadPoolExecutor

        from microstructure_ve.backends.dolfinx import _run as run

        sims = [homogeneous_simulation(n=3, dim=2, E=1000.0 * (i + 1), f_count=3)
                for i in range(4)]
        run.clear_cache()
        serial = [run.run(s) for s in sims]
        run.clear_cache()
        try:
            with ThreadPoolExecutor(max_workers=4) as ex:
                threaded = list(ex.map(run.run, sims))
        finally:
            run.clear_cache()  # don't leave racily-built state for later tests
        for got, want in zip(threaded, serial):
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=0)

    def test_parallel_sweep_does_not_hold_backend_lock(self):
        # A workers>1 sweep touches no parent-process FE state (workers rebuild the
        # solver from the pickled sim), so its fan-out must run OUTSIDE _BACKEND_LOCK:
        # another thread can take the lock (e.g. for its own serial run) while the
        # sweep is in flight. The cancel callback -- polled in the sweeping thread as
        # futures complete -- doubles as the "mid-sweep" synchronization point.
        import threading

        from microstructure_ve.backends.dolfinx import _run as run

        sim = homogeneous_simulation(n=3, dim=2, f_count=3)
        inside_sweep = threading.Event()
        probed = threading.Event()

        def cancel():
            inside_sweep.set()
            probed.wait(timeout=60)  # hold the sweep open while the main thread probes
            return False

        result = {}

        def sweep():
            result["out"] = run.run(sim, workers=2, cancel=cancel)

        t = threading.Thread(target=sweep)
        t.start()
        try:
            self.assertTrue(inside_sweep.wait(timeout=300), "sweep never reached cancel poll")
            got_lock = run._BACKEND_LOCK.acquire(timeout=15)
            if got_lock:
                run._BACKEND_LOCK.release()
        finally:
            probed.set()
            t.join(timeout=300)
        self.assertFalse(t.is_alive(), "parallel sweep did not finish")
        self.assertTrue(got_lock, "workers>1 sweep held _BACKEND_LOCK during the fan-out")
        serial = run.run(sim, workers=1)
        np.testing.assert_allclose(result["out"], serial, rtol=1e-9, atol=0)

    def test_parent_env_untouched_during_parallel_sweep(self):
        # Workers cap their OWN BLAS/OpenMP threads in-process (threadpoolctl); the parent's
        # os.environ must never be mutated. A watcher thread samples the BLAS env vars while
        # the sweep runs -- every sample must equal the pre-sweep snapshot. This is red for
        # any parent-side env juggling: the old whole-sweep cap held "1" for the entire
        # sweep, and the warmup variant set "1" during spawn -- both are caught here. When
        # the env is never touched (the guarantee) green is exact: no sample can differ.
        import os
        import threading
        import time

        from microstructure_ve.backends.dolfinx import _run as run

        blas_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        before = {v: os.environ.get(v) for v in blas_vars}
        samples = []
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                samples.append({v: os.environ.get(v) for v in blas_vars})
                time.sleep(0.001)

        # enough workers + solves that spawn and the solve phase span many 1 ms samples
        sim = homogeneous_simulation(n=4, dim=2, f_count=8)
        w = threading.Thread(target=watch)
        w.start()
        try:
            run.run(sim, workers=2)
        finally:
            stop.set()
            w.join()
        mutated = [s for s in samples if s != before]
        self.assertEqual(samples and mutated, [],
                         f"parent BLAS env was mutated during the sweep: {mutated[:3]}")
        self.assertEqual({v: os.environ.get(v) for v in blas_vars}, before)

    def test_workers_run_single_threaded(self):
        # The cap must actually take effect in the worker: a task reporting the worker's
        # own threadpool_info() shows every native pool pinned to one thread.
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        from microstructure_ve.backends.dolfinx import _run as run

        sim = homogeneous_simulation(n=3, dim=2, f_count=2)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=1, mp_context=ctx, initializer=run._init_worker,
            initargs=(sim, True, "auto", None),
        ) as ex:
            counts = ex.submit(_worker_threadpool_counts).result(timeout=120)
        self.assertTrue(counts, "no native thread pools reported")
        self.assertTrue(all(n == 1 for n in counts), f"workers not single-threaded: {counts}")

    def test_workers_inside_worker_process_raises(self):
        # simulate being a spawned worker that re-ran the driver: parent_process() != None
        from unittest import mock

        from microstructure_ve.backends.dolfinx import _run as run

        sim = homogeneous_simulation(n=3, dim=2, f_count=2)  # >1 freq to enter the parallel path
        with mock.patch("multiprocessing.parent_process", return_value=object()):
            with self.assertRaises(RuntimeError) as cm:
                run.run(sim, workers=2)
        self.assertIn("worker process", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
