"""Image-processing and IO helper functions."""
import pathlib
import tempfile
import unittest

import numpy as np

from microstructure_ve import (
    assign_intph,
    in_sorted,
    load_matlab_microstructure,
    load_viscoelasticity,
    periodic_assign_intph,
)


class InSortedTests(unittest.TestCase):
    def test_membership(self):
        arr = np.array([1, 3, 5, 7])
        self.assertTrue(in_sorted(arr, 1))
        self.assertTrue(in_sorted(arr, 7))
        self.assertFalse(in_sorted(arr, 6))
        self.assertFalse(in_sorted(arr, 8))  # past the end


class AssignIntphTests(unittest.TestCase):
    def test_layers_by_distance(self):
        img = np.ones((5, 5), dtype=int)
        img[2, 2] = 0  # single particle pixel
        intph = assign_intph(img, [1])
        self.assertEqual(intph[2, 2], 0)            # particle
        self.assertEqual(intph[2, 1], 1)            # orthogonal neighbor, dist 1
        self.assertEqual(intph[0, 0], 2)            # far matrix
        self.assertEqual(set(np.unique(intph)), {0, 1, 2})


class PeriodicAssignIntphTests(unittest.TestCase):
    def test_wraparound_creates_interphase(self):
        img = np.ones((6, 6), dtype=int)
        img[0, 3] = 0  # particle on the top edge
        nonper = assign_intph(img, [1])
        per = periodic_assign_intph(img, [1])
        # the bottom-edge cell is far from the particle without periodicity (matrix),
        # but one step from its wrapped image with periodicity (interphase)
        self.assertEqual(nonper[5, 3], 2)
        self.assertEqual(per[5, 3], 1)
        self.assertEqual(per.shape, img.shape)


class LoadViscoelasticityTests(unittest.TestCase):
    def test_sorts_and_builds_complex(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "ve.txt"
            path.write_text("10 2 3\n1 4 5\n100 6 7\n")
            freq, youngs = load_viscoelasticity(path)
        np.testing.assert_array_equal(freq, [1, 10, 100])
        self.assertTrue(np.all(np.diff(freq) > 0))
        np.testing.assert_array_equal(youngs, [4 + 5j, 2 + 3j, 6 + 7j])


class LoadMatlabMicrostructureTests(unittest.TestCase):
    def test_roundtrip(self):
        from scipy.io import savemat

        arr = np.array([[0, 1], [1, 0]])
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "ms.mat"
            savemat(str(path), {"ms": arr})
            out = load_matlab_microstructure(str(path), "ms")
        np.testing.assert_array_equal(out, arr)


if __name__ == "__main__":
    unittest.main()
