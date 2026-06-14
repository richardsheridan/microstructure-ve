"""Spec parsing for the DOLFINx backend (pure numpy -> runs in the msve tier).

These exercise ``microstructure_ve.backends._dolfinx.spec``, which has no dolfinx
dependency, so they run everywhere the numpy-only suite runs.
"""
import unittest

import numpy as np

from microstructure_ve.backends.dolfinx import _spec as spec

from tests._helpers import synthetic_simulation


class PeriodicPairsTests(unittest.TestCase):
    def test_2d_edge_and_corner_pairing(self):
        # 3x3 node grid: right edge + top edge + far corner map to their min images
        pairs = set(spec.periodic_pairs((3, 3)))
        self.assertEqual(pairs, {(3, 1), (6, 4), (7, 1), (8, 2), (9, 1)})

    def test_slaves_disjoint_and_masters_off_max_face(self):
        shape = (5, 4)
        pairs = spec.periodic_pairs(shape)
        slaves = [s for s, _ in pairs]
        masters = {m for _, m in pairs}
        self.assertEqual(len(slaves), len(set(slaves)), "slaves not disjoint")
        self.assertTrue(set(slaves).isdisjoint(masters), "a master is also a slave")
        # number of slaves = total nodes minus the interior-min block (rows<max & cols<max)
        ny, nx = shape
        self.assertEqual(len(slaves), ny * nx - (ny - 1) * (nx - 1))

    def test_3d_smoke(self):
        pairs = spec.periodic_pairs((3, 3, 3))
        slaves = [s for s, _ in pairs]
        self.assertEqual(len(slaves), len(set(slaves)))
        # 27 nodes - 2*2*2 interior-min block = 19 max-face nodes
        self.assertEqual(len(slaves), 27 - 8)


class SpecParsingTests(unittest.TestCase):
    def setUp(self):
        self.sim = synthetic_simulation()
        self.model = self.sim.model

    def test_default_frequencies(self):
        freqs = spec.default_frequencies(self.sim)
        self.assertEqual(len(freqs), 30)
        np.testing.assert_allclose([freqs[0], freqs[-1]], [1e-7, 1e5], rtol=1e-12)

    def test_drive_displacement(self):
        self.assertAlmostEqual(spec.drive_displacement(self.sim), 0.005)

    def test_geometry(self):
        g = spec.geometry(self.model, self.sim)
        self.assertEqual(g.dim, 2)
        np.testing.assert_array_equal(g.shape, [7, 7])
        self.assertAlmostEqual(g.Lx, 6 * 0.0025)
        self.assertAlmostEqual(g.cross_area, 6 * 0.0025)
        self.assertAlmostEqual(g.exx, 0.005 / (6 * 0.0025))

    def test_material_cell_maps_partition(self):
        mat_of_cell, poissons, modulus_fns = spec.material_cell_maps(self.model)
        self.assertEqual(len(mat_of_cell), 36)  # 6x6 pixels
        self.assertEqual(len(modulus_fns), len(self.model.materials))
        self.assertEqual(set(np.unique(mat_of_cell)), set(range(len(modulus_fns))))
        # each cell's assigned material owns that 1-indexed element
        for mi, mat in enumerate(self.model.materials):
            for e in mat.elset.elements:
                self.assertEqual(mat_of_cell[e - 1], mi)

    def test_require_periodic(self):
        spec.require_periodic(self.model)  # synthetic sim has a PBC -> no raise


if __name__ == "__main__":
    unittest.main()
