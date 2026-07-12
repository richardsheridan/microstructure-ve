"""The Mechanical-MNIST benchmark: NeoHookean solve sanity + parity with committed data.

``tests/_mechanical_mnist.py`` builds the benchmark's uniaxial-extension Simulation; here
we check (fenicsx env only, self-skipping elsewhere) that the coupled-Neo-Hookean DOLFINx
solve linearizes to the correct small-strain elasticity, and (once the data excerpts and
energy output land) that the solved strain energies match the benchmark's committed rows.
"""
import copy
import unittest

import numpy as np

from microstructure_ve.constitutive import Elastic

from tests._mechanical_mnist import (
    DATA_DIR, DISP_VALS, E_HIGH, POISSON, load_bitmap, load_psi, mm_simulation,
)

try:
    import dolfinx  # noqa: F401
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

needs_dolfinx = unittest.skipUnless(HAS_DOLFINX, "needs the fenicsx env (dolfinx)")


def _homogeneous_bitmap():
    """All pixels 255 -> a single material with E = E_HIGH."""
    return np.full((28, 28), 255)


class DataFilesTests(unittest.TestCase):
    """The committed Mechanical-MNIST excerpts (numpy-only; runs in both envs)."""

    def test_bitmaps_shape_and_range(self):
        for which in ("train", "test"):
            for i in range(10):
                img = load_bitmap(f"{which}_data_{i}")
                self.assertEqual(img.shape, (28, 28))
                self.assertTrue(np.all(img == np.rint(img)))
                self.assertGreaterEqual(img.min(), 0)
                self.assertLessEqual(img.max(), 255)

    def test_psi_excerpts_shape_and_baseline(self):
        for which in ("train", "test"):
            psi = load_psi(which)
            self.assertEqual(psi.shape, (10, len(DISP_VALS)))
            # column 0 is the subtracted d=0 baseline; energies grow monotonically
            # (the d=0.001 column can round to 0 at the file's output precision, so
            # strict growth is only asserted from d=0.01 on)
            np.testing.assert_array_equal(psi[:, 0], 0.0)
            self.assertTrue(np.all(np.diff(psi, axis=1) >= 0))
            self.assertTrue(np.all(np.diff(psi[:, 2:], axis=1) > 0))

    def test_readme_carries_attribution(self):
        text = (DATA_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("MIT license", text)
        self.assertIn("elejeune11/Mechanical-MNIST", text)


@needs_dolfinx
class NeoHookeanSolverTests(unittest.TestCase):
    def test_homogeneous_small_strain_matches_linear_elastic(self):
        # At 1e-4 nominal strain the coupled Neo-Hookean must reproduce the linear
        # plane-strain solve on the same mesh (it linearizes to Lame (lam, mu) exactly).
        from microstructure_ve.backends.dolfinx import _run as run

        d = 28.0 * 1e-4
        sim = mm_simulation(_homogeneous_bitmap(), disp_vals=[d])
        lin = copy.deepcopy(sim)
        for m in lin.model.materials:
            m.response = Elastic(poisson=POISSON, youngs=E_HIGH)
        for s in lin.steps:
            s.nlgeom = False

        fe = run.run(sim)
        ref = run.run(lin)
        self.assertEqual(fe.shape, ref.shape)
        # y-reaction on the driven top edge; x-reaction is 0 by symmetry on both
        np.testing.assert_allclose(fe[0, 2], ref[0, 2], rtol=1e-3)
        # drive displacement column must be exact
        np.testing.assert_allclose(fe[0, -1], ref[0, -1], rtol=1e-12)

    def test_homogeneous_large_strain_is_geometrically_nonlinear(self):
        # At 50% nominal strain the finite-strain solve must depart from the linear one
        # by a clearly finite margin. This guards the dispatch: if NeoHookean were not
        # recognized as hyperelastic, the sim would silently fall back to the linear path
        # and the two reactions would agree exactly.
        from microstructure_ve.backends.dolfinx import _run as run

        d = 14.0
        sim = mm_simulation(_homogeneous_bitmap(), disp_vals=[d])
        lin = copy.deepcopy(sim)
        for m in lin.model.materials:
            m.response = Elastic(poisson=POISSON, youngs=E_HIGH)
        for s in lin.steps:
            s.nlgeom = False

        fe = run.run(sim, n_incr=5)
        ref = run.run(lin)
        rel = abs(fe[0, 2] - ref[0, 2]) / abs(ref[0, 2])
        self.assertGreater(rel, 1e-2,
                           msg=f"NH reaction {fe[0, 2]:.6e} vs linear {ref[0, 2]:.6e} -- "
                               "finite-strain solver not engaged?")

    def test_return_energy_clapeyron(self):
        # In the linear regime the stored energy of the converged state is half the work
        # of the (linearly grown) top-edge reaction: psi = RF_y * d / 2 (Clapeyron).
        from microstructure_ve.backends.dolfinx import _run as run

        d = 28.0 * 1e-4
        sim = mm_simulation(_homogeneous_bitmap(), disp_vals=[d])
        rows, psi = run.run(sim, return_energy=True)
        self.assertEqual(psi.shape, (1,))
        np.testing.assert_allclose(psi[0], 0.5 * rows[0, 2] * d, rtol=1e-3)

    def test_multistep_warm_started_equals_single_step(self):
        # Hyperelasticity is path-independent: a 3-step warm-started ramp must land on
        # the same converged state (row and energy) as one step straight to the final d.
        from microstructure_ve.backends.dolfinx import _run as run

        bitmap = np.zeros((28, 28))
        bitmap[10:18, 10:18] = 255  # a stiff block so the fields are heterogeneous
        multi = mm_simulation(bitmap, disp_vals=[2.0, 5.0, 8.0])
        single = mm_simulation(bitmap, disp_vals=[8.0])
        rows_m, psi_m = run.run(multi, return_energy=True, n_incr=2)
        rows_s, psi_s = run.run(single, return_energy=True)
        self.assertEqual(rows_m.shape[0], 3)
        np.testing.assert_allclose(rows_m[-1], rows_s[0], rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(psi_m[-1], psi_s[0], rtol=1e-6)
        # and the intermediate steps must actually differ (each drives its own value)
        self.assertGreater(abs(psi_m[1] - psi_m[0]), 1e-6)

    def test_return_energy_rejects_non_hyperelastic(self):
        from microstructure_ve.backends.dolfinx import _run as run

        d = 28.0 * 1e-4
        lin = mm_simulation(_homogeneous_bitmap(), disp_vals=[d])
        for m in lin.model.materials:
            m.response = Elastic(poisson=POISSON, youngs=E_HIGH)
        for s in lin.steps:
            s.nlgeom = False
        with self.assertRaises(ValueError):
            run.run(lin, return_energy=True)


if __name__ == "__main__":
    unittest.main()
