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

from tests._mechanical_mnist import E_HIGH, POISSON, mm_simulation

try:
    import dolfinx  # noqa: F401
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

needs_dolfinx = unittest.skipUnless(HAS_DOLFINX, "needs the fenicsx env (dolfinx)")


def _homogeneous_bitmap():
    """All pixels 255 -> a single material with E = E_HIGH."""
    return np.full((28, 28), 255)


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
