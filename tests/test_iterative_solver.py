"""The GMRES+ILU IterativeSolver path (requires the fenicsx env).

The iterative solver solves the *same* linear system as the LU path, so it is validated
against ``LuSolver`` (not ABAQUS): on a small periodic cell the homogenized rows must agree.
The suite's meshes are far below the LU time budget, so ``select_solver_kind`` never picks
iterative on its own -- we force it with ``solver="iterative"``. Also pins the cancellation
contract (the KSP monitor raises ``Cancelled`` mid-solve) and the loud non-convergence guard.
"""
import unittest

import numpy as np

from tests._matrix import matrix_simulation

try:
    import dolfinx  # noqa: F401
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

needs_dolfinx = unittest.skipUnless(HAS_DOLFINX, "needs the fenicsx env (dolfinx)")

_CELL = dict(mode="uniaxial_x", traction="free", bc="periodic", dim=2)


@needs_dolfinx
class IterativeSolverTests(unittest.TestCase):
    def setUp(self):
        from microstructure_ve.backends.dolfinx import clear_cache

        clear_cache()  # isolate: don't reuse a cached solver of the other kind

    def test_iterative_matches_lu(self):
        from microstructure_ve.backends.dolfinx import clear_cache, run

        for tt in ("elastic", "viscoelastic"):
            for homog in (True, False):
                with self.subTest(test_type=tt, homogeneous=homog):
                    sim_kw = dict(test_type=tt, homogeneous=homog, **_CELL)
                    clear_cache()
                    lu = run(matrix_simulation(**sim_kw), solver="lu")
                    clear_cache()
                    it = run(matrix_simulation(**sim_kw), solver="iterative")
                    scale = float(np.max(np.abs(lu[:, 1:3]))) or 1e-30
                    np.testing.assert_allclose(it, lu, rtol=1e-6, atol=scale * 1e-6)

    def test_auto_uses_lu_on_small_mesh(self):
        # below the LU time budget, "auto" must stay on LU -- identical to forcing it
        from microstructure_ve.backends.dolfinx import clear_cache, run

        sim_kw = dict(test_type="viscoelastic", homogeneous=False, **_CELL)
        clear_cache()
        auto = run(matrix_simulation(**sim_kw))            # solver="auto" default
        clear_cache()
        lu = run(matrix_simulation(**sim_kw), solver="lu")
        np.testing.assert_array_equal(auto, lu)

    def test_cancel_raises_during_solve(self):
        from microstructure_ve.backends.dolfinx import run
        from microstructure_ve.backends.dolfinx._solver import Cancelled

        sim = matrix_simulation(test_type="viscoelastic", homogeneous=False, **_CELL)
        with self.assertRaises(Cancelled):
            run(sim, solver="iterative", cancel=lambda: True)

    def test_nonconvergence_raises_loudly(self):
        from microstructure_ve.backends.dolfinx import run

        sim = matrix_simulation(test_type="viscoelastic", homogeneous=False, **_CELL)
        with self.assertRaises(RuntimeError) as cm:
            run(sim, solver="iterative", petsc_options={"ksp_max_it": 1, "ksp_rtol": 1e-14})
        self.assertIn("did not converge", str(cm.exception))

    def test_auto_solver_kind_is_lu_in_2d_iterative_for_big_3d(self):
        # benchmarking: 2D LU completes to >500k ndof while GMRES+ILU diverges there, so auto
        # must stay LU in 2D; 3D LU fill explodes, so auto switches to iterative above budget.
        from microstructure_ve.backends.dolfinx._solver import select_solver_kind

        self.assertEqual(select_solver_kind(1_000_000, 2), "lu")   # huge 2D -> still LU
        self.assertEqual(select_solver_kind(2_000, 2), "lu")
        self.assertEqual(select_solver_kind(2_000, 3), "lu")       # small 3D -> LU
        self.assertEqual(select_solver_kind(50_000, 3), "iterative")  # big 3D -> iterative

    def test_bad_solver_name(self):
        from microstructure_ve.backends.dolfinx import run

        sim = matrix_simulation(test_type="elastic", homogeneous=True, **_CELL)
        with self.assertRaises(ValueError):
            run(sim, solver="bogus")


if __name__ == "__main__":
    unittest.main()
