"""Non-converged nonlinear ``Static`` steps raise ``ConvergenceError`` from ``run``.

Each test drives one solver path of the DOLFINx backend to a state that does not meet its
convergence test and checks that ``run`` raises instead of returning a row. The hyperelastic
and standard plastic paths use drives that defeat the solver on their own; the periodic
plastic path, which no public drive defeats, lowers a module-level iteration cap as a seam.
The linear path's equivalent lives in ``tests/test_iterative_solver.py``.
"""
import unittest
from unittest import mock

import numpy as np

from microstructure_ve.constitutive import Plastic

from tests._matrix import matrix_simulation
from tests._mechanical_mnist import mm_simulation

try:
    import dolfinx  # noqa: F401
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

needs_dolfinx = unittest.skipUnless(HAS_DOLFINX, "needs the fenicsx env (dolfinx)")


def _stiff_block_bitmap():
    """A Mechanical-MNIST bitmap with one stiff square inclusion in a soft background."""
    bitmap = np.zeros((28, 28))
    bitmap[10:18, 10:18] = 255
    return bitmap


@needs_dolfinx
class ConvergenceErrorTests(unittest.TestCase):
    def test_convergence_error_is_a_runtime_error_exported_beside_cancelled(self):
        """``ConvergenceError`` is importable from the backend package and is caught by an
        ``except RuntimeError`` written for the earlier plain error."""
        from microstructure_ve.backends.dolfinx import ConvergenceError

        self.assertTrue(issubclass(ConvergenceError, RuntimeError))

    def test_periodic_hyperelastic_newton_failure_raises(self):
        """A periodic hyperelastic cell compressed nearly flat, whose final Newton solve does
        not converge, raises instead of returning a row."""
        from microstructure_ve.backends.dolfinx import ConvergenceError, run

        sim = matrix_simulation(mode="compression", traction="confined_slip", bc="periodic",
                                dim=2, test_type="reduced_polynomial", displacement=0.0145)
        with self.assertRaises(ConvergenceError):
            run(sim)

    def test_standard_hyperelastic_newton_failure_raises(self):
        """A standard-BC Neo-Hookean cell compressed to half its height in one increment, whose
        Newton solve does not converge, raises instead of returning a row."""
        from microstructure_ve.backends.dolfinx import ConvergenceError, run

        sim = mm_simulation(_stiff_block_bitmap(), disp_vals=[-14.0])
        with self.assertRaises(ConvergenceError):
            run(sim, n_incr=1)

    def test_standard_plastic_newton_failure_raises(self):
        """A standard-BC plastic cell with a near-zero yield stress driven far past yield, whose
        Newton solve does not converge, raises instead of returning a row."""
        from microstructure_ve.backends.dolfinx import ConvergenceError, run

        sim = mm_simulation(_stiff_block_bitmap(), disp_vals=[100.0])
        for m in sim.model.materials:
            m.response = Plastic(poisson=0.3, youngs=m.response.youngs,
                                 yield_stress=[0.01, 0.01], plastic_strain=[0.0, 0.05])
        for s in sim.steps:
            s.nlgeom = False
        with self.assertRaises(ConvergenceError):
            run(sim)

    def test_periodic_plastic_newton_failure_raises(self):
        """A fully prescribed periodic plastic cell whose Newton solve is cut short of
        convergence raises instead of returning a row."""
        from microstructure_ve.backends.dolfinx import ConvergenceError, _plastic, run

        sim = matrix_simulation(mode="uniaxial_x", traction="confined_slip", bc="periodic",
                                dim=2, test_type="hyperelastic_plastic")
        with mock.patch.object(_plastic, "_MAX_NEWTON", 1):
            with self.assertRaises(ConvergenceError):
                run(sim)

    def test_free_lateral_root_find_failure_raises(self):
        """A periodic plastic cell with a free lateral face whose root-find for zero lateral
        traction is cut short of convergence raises, although each of its Newton solves
        converges."""
        from microstructure_ve.backends.dolfinx import ConvergenceError, _plastic, run

        sim = matrix_simulation(mode="uniaxial_x", traction="free", bc="periodic", dim=2,
                                test_type="hyperelastic_plastic")
        with mock.patch.object(_plastic, "_MAX_OUTER", 1):
            with self.assertRaises(ConvergenceError):
                run(sim)


if __name__ == "__main__":
    unittest.main()
