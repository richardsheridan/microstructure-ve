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
        # green cells are homogeneous x-uniaxial periodic viscoelastic: the homogenized
        # complex response must equal the material's own E*(f) through the Lamé relation
        # (a uniform field on one material -> homogenized stress == constitutive stress).
        from microstructure_ve.backends.dolfinx import _spec as spec

        dim = cell["dim"]
        geom = spec.Geometry.from_model(sim.model, sim)
        (mat,) = sim.model.materials
        nu = mat.poisson
        for row in rows:
            f = row[0]
            Estar = complex(mat.complex_modulus(np.array([f]))[0])
            lam, mu = _lame(Estar, nu)
            sigma = (np.array(row[1:1 + dim]) + 1j * np.array(row[1 + dim:1 + 2 * dim]))
            sigma = sigma / geom.cross_area  # complex sigma-bar normal row
            if cell["traction"] == "confined_slip":
                target = (lam + 2 * mu) * geom.exx
            else:  # free -> apparent uniaxial; lateral normal stresses vanish
                num = 4 * mu * (lam + mu) / (lam + 2 * mu) if dim == 2 else Estar
                target = num * geom.exx
                for k in range(1, dim):
                    self.assertAlmostEqual(sigma[k].real, 0.0, delta=abs(target) * 1e-5)
                    self.assertAlmostEqual(sigma[k].imag, 0.0, delta=abs(target) * 1e-5)
            self.assertAlmostEqual(sigma[0].real, target.real, delta=abs(target) * 1e-5)
            self.assertAlmostEqual(sigma[0].imag, target.imag, delta=abs(target) * 1e-5)


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
