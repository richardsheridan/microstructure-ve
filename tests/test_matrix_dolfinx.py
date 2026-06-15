"""DOLFINx red/green harness over the feature matrix (requires the fenicsx env).

One test per matrix cell x {elastic, viscoelastic}, generated dynamically. Cells the FE
backend supports today (``is_fe_green``) actually solve and assert closed-form physics on
a homogeneous RVE; every other cell is wrapped in ``unittest.expectedFailure`` -- it is
*expected* to raise/fail now (no Dynamic step, non-periodic, or a non-x drive the backend
rejects). When a feature lands, its cell stops failing and surfaces as an **unexpected
success**, turning the suite red so the decorator can be dropped (and ``is_fe_green``
widened). That flip is the whole point of the red-green setup.
"""
import unittest

import numpy as np

from tests._matrix import is_fe_green, matrix_cases, matrix_simulation

try:
    import dolfinx  # noqa: F401
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

needs_dolfinx = unittest.skipUnless(HAS_DOLFINX, "needs the fenicsx env (dolfinx)")


def _lame(E, nu):
    """Lamé parameters; ``E`` may be complex (a viscoelastic E*(f))."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


@needs_dolfinx
class MatrixDolfinxTests(unittest.TestCase):
    """Methods are attached below; green cells assert, red cells expectedFailure."""

    def _assert_green_invariants(self, cell, sim, rows):
        # Homogeneous RVE: the FE homogenized response must equal the constitutive response
        # for the imposed macro strain (a uniform field on one material). This is mode-
        # agnostic -- it reads the imposed/free strain off the backend's MacroLoading and
        # contracts it with the analytic isotropic stiffness from E*(f) -- so normal, shear,
        # and compression cells are all checked without per-mode code.
        from microstructure_ve.backends.dolfinx import _loading

        dim = cell["dim"]
        arr = np.asarray(rows, dtype=complex)
        self.assertEqual(arr.shape[1], 1 + 3 * dim)
        self.assertTrue(np.all(np.isfinite(arr)))
        if cell["bc"] != "periodic" or len(list(sim.steps)) > 1:
            # Non-periodic (standard) cells are clamped Dirichlet BVPs, not a uniform-field
            # homogenization; multi-step cells mix Static (real *Elastic) and Dynamic rows in
            # one table. Neither fits the single closed-form C:E invariant, so their numerical
            # correctness is pinned by test_matrix_parity against the ABAQUS oracle.
            return

        loading = _loading.macro_loading(sim)
        a0 = loading.primary_axis
        (mat,) = sim.model.materials
        nu = mat.poisson
        for row in rows:
            f = row[0]
            Estar = complex(mat.complex_modulus(np.array([f]))[0])
            lam, mu = _lame(Estar, nu)

            E = np.zeros((dim, dim), dtype=complex)
            for (i, j), v in loading.imposed.items():
                E[i, j] = v
                E[j, i] = v
            free_axes = [b for (b, _) in loading.free]
            if free_axes:  # choose free normals so each conjugate normal stress vanishes
                tr_fixed = sum(E[k, k] for k in range(dim) if k not in free_axes)
                A = np.full((len(free_axes), len(free_axes)), lam, dtype=complex)
                A[np.diag_indices_from(A)] += 2 * mu
                xs = np.linalg.solve(A, np.full(len(free_axes), -lam * tr_fixed, dtype=complex))
                for b, x in zip(free_axes, xs):
                    E[b, b] = x

            sigma = lam * np.trace(E) * np.eye(dim) + 2 * mu * E
            RF = sigma[:, a0] * loading.cross_area
            fe_RF = np.array(row[1:1 + dim]) + 1j * np.array(row[1 + dim:1 + 2 * dim])
            scale = float(np.max(np.abs(RF))) or 1e-30
            np.testing.assert_allclose(fe_RF, RF, rtol=1e-4, atol=scale * 1e-5)

        U = np.array(rows[0][1 + 2 * dim:1 + 3 * dim])
        self.assertAlmostEqual(U[loading.primary_dof - 1], loading.drive_value,
                               delta=abs(loading.drive_value) * 1e-6 + 1e-12)


def _make_test(cell, test_type):
    def test(self):
        from microstructure_ve.backends.dolfinx import _run as run

        sim = matrix_simulation(test_type=test_type, homogeneous=True, **cell)
        rows = run.run(sim)  # raises for every not-yet-supported cell (the red ones)
        self._assert_green_invariants(cell, sim, rows)

    return test


for _cell, _tt in matrix_cases():
    _name = (
        f"test_{_cell['dim']}d_{_cell['mode']}_{_cell['traction']}"
        f"_{_cell['bc']}_{_tt}"
    )
    _method = _make_test(_cell, _tt)
    if not is_fe_green(_cell, _tt):
        _method = unittest.expectedFailure(_method)
    setattr(MatrixDolfinxTests, _name, _method)

del _cell, _tt, _name, _method


if __name__ == "__main__":
    unittest.main()
