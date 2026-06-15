"""FE-vs-ABAQUS parity over the feature matrix (requires the fenicsx env).

For every matrix cell the DOLFINx backend currently supports (``is_fe_green``), solve the
*same* Simulation the committed ABAQUS oracle under ``tests/data/`` was generated from and
require the homogenized reaction-force response (storage ``RF_Real``, loss ``RF_Imag``) and
the drive displacement ``U`` to match the oracle.

The matrix oracles are generated so this comparison is both interpolation-free (the sweep
lands exactly on the tabular table nodes) and inertia-free (tiny density), so agreement is
tight -- observed ~1e-7, asserted at rtol 1e-4. As FE features land and ``is_fe_green``
widens, the corresponding cells join this parity check automatically; a green cell that
disagrees with ABAQUS fails loudly here.

This complements ``test_matrix_dolfinx`` (which checks closed-form physics on a homogeneous
RVE and tracks the red cells as expected failures); here the RVE is the heterogeneous
oracle sim and the truth is ABAQUS.
"""
import pathlib
import unittest

import numpy as np

from tests._matrix import is_fe_green, matrix_cells, matrix_simulation

try:
    import dolfinx  # noqa: F401
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

needs_dolfinx = unittest.skipUnless(HAS_DOLFINX, "needs the fenicsx env (dolfinx)")
DATA = pathlib.Path(__file__).resolve().parent / "data"
RTOL = 1e-4  # generous vs the ~1e-7 observed; robust to ABAQUS .dat frequency rounding


def _oracle_name(cell, tt):
    return (f"oracle_{cell['dim']}d_{cell['mode']}_{cell['traction']}"
            f"_{cell['bc']}_{tt}.tsv")


@needs_dolfinx
class MatrixParityTests(unittest.TestCase):
    """One method per FE-supported cell is attached below."""


def _make_test(cell, tt):
    def test(self):
        from microstructure_ve.backends.dolfinx import _run as run

        dim = cell["dim"]
        oracle = np.atleast_2d(np.loadtxt(DATA / _oracle_name(cell, tt), skiprows=1))
        oracle = oracle[np.argsort(oracle[:, 0])]
        fe = run.run(matrix_simulation(test_type=tt, **cell))  # the oracle's own sim
        fe = fe[np.argsort(fe[:, 0])]
        self.assertEqual(fe.shape, oracle.shape)

        scale = float(np.max(np.abs(oracle[:, 1:1 + dim])))  # storage sets the abs floor
        storage = slice(1, 1 + dim)
        loss = slice(1 + dim, 1 + 2 * dim)
        drive = slice(1 + 2 * dim, 1 + 3 * dim)
        np.testing.assert_allclose(fe[:, storage], oracle[:, storage],
                                   rtol=RTOL, atol=scale * RTOL)
        np.testing.assert_allclose(fe[:, loss], oracle[:, loss],
                                   rtol=RTOL, atol=scale * RTOL)
        np.testing.assert_allclose(fe[:, drive], oracle[:, drive], rtol=1e-6, atol=1e-9)

    return test


_attached = 0
for _cell in matrix_cells():
    for _tt in ("elastic", "viscoelastic"):
        if is_fe_green(_cell, _tt):
            _name = (f"test_{_cell['dim']}d_{_cell['mode']}_{_cell['traction']}"
                     f"_{_cell['bc']}_{_tt}")
            setattr(MatrixParityTests, _name, _make_test(_cell, _tt))
            _attached += 1

assert _attached, "no FE-green matrix cells to check parity for"
del _cell, _tt


if __name__ == "__main__":
    unittest.main()
