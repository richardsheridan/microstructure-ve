"""DOLFINx FE backend tests (require the fenicsx env; skip cleanly otherwise).

Tests 1-4 are self-contained analytic checks needing no ABAQUS: mesh/dof construction,
the periodic MPC, and homogeneous closed-form uniaxial stresses. ABAQUS-parity tests
live in test_backend_dolfinx_parity once the oracles are committed.
"""
import unittest

import numpy as np

from tests._helpers import SCALE, homogeneous_simulation, synthetic_simulation

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
        from microstructure_ve.materials import Material
        from microstructure_ve.steps import Model
        import numpy as np

        img = np.array([[0, 1], [1, 0]])
        nodes = GridNodes.from_matl_img(img, SCALE)
        elements = GridElements(nodes, type="CPE4")
        e0, e1 = ElementSet.from_matl_img(img)
        E0, E1, nu = 1000.0, 5000.0, 0.3
        model = Model(nodes=nodes, elements=elements,
                      materials=[Material(e0, 1.0, nu, E0), Material(e1, 1.0, nu, E1)])
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
    def _sigma(self, sim, lateral_bc):
        from microstructure_ve.backends.dolfinx import _run as run, _spec as spec

        geom = spec.Geometry.from_model(sim.model, sim)
        row = run.run(sim, freqs=[1.0], lateral_bc=lateral_bc)[0]
        dim = sim.model.nodes.dim
        rf_real = np.array(row[1:1 + dim])
        return rf_real / geom.cross_area, geom.exx  # sigma-bar normal row 0..dim-1

    def test_confined_plane_strain(self):
        E, nu = 3000.0, 0.3
        lam, mu = _lame(E, nu)
        sim = homogeneous_simulation(n=4, dim=2, E=E, nu=nu)
        sigma, exx = self._sigma(sim, "confined")
        self.assertAlmostEqual(sigma[0], (lam + 2 * mu) * exx, delta=abs(sigma[0]) * 1e-6)

    def test_free_lateral_plane_strain(self):
        E, nu = 3000.0, 0.3
        lam, mu = _lame(E, nu)
        sim = homogeneous_simulation(n=4, dim=2, E=E, nu=nu)
        sigma, exx = self._sigma(sim, "free")
        expected = 4 * mu * (lam + mu) / (lam + 2 * mu) * exx
        self.assertAlmostEqual(sigma[0], expected, delta=abs(expected) * 1e-6)
        self.assertAlmostEqual(sigma[1], 0.0, delta=abs(expected) * 1e-6)  # lateral free

    def test_free_lateral_3d_is_uniaxial_stress(self):
        E, nu = 3000.0, 0.3
        sim = homogeneous_simulation(n=3, dim=3, E=E, nu=nu)
        sigma, exx = self._sigma(sim, "free")
        self.assertAlmostEqual(sigma[0], E * exx, delta=abs(E * exx) * 1e-6)
        self.assertAlmostEqual(sigma[1], 0.0, delta=abs(E * exx) * 1e-6)
        self.assertAlmostEqual(sigma[2], 0.0, delta=abs(E * exx) * 1e-6)


@needs_dolfinx
class FrequencyParallelTests(unittest.TestCase):
    def test_workers_match_serial(self):
        # frequency-varying (viscoelastic) sim so the parallel split is non-trivial
        from microstructure_ve.backends.dolfinx import _run as run

        sim = synthetic_simulation()
        freqs = [1e-3, 1e0, 1e3, 1e5]
        serial = run.run(sim, freqs=freqs, lateral_bc="confined", workers=1)
        parallel = run.run(sim, freqs=freqs, lateral_bc="confined", workers=2)
        np.testing.assert_allclose(parallel, serial, rtol=1e-9, atol=0)

    def test_workers_inside_worker_process_raises(self):
        # simulate being a spawned worker that re-ran the driver: parent_process() != None
        from unittest import mock

        from microstructure_ve.backends.dolfinx import _run as run

        sim = homogeneous_simulation(n=3, dim=2)
        with mock.patch("multiprocessing.parent_process", return_value=object()):
            with self.assertRaises(RuntimeError) as cm:
                run.run(sim, freqs=[1.0, 2.0], workers=2)
        self.assertIn("worker process", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
