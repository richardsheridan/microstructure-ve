"""FE-vs-ABAQUS parity over the feature matrix (requires a DOLFINx env).

For every matrix cell the DOLFINx backend currently supports (``is_fe_green``), solve the
*same* Simulation the committed ABAQUS oracle under ``tests/data/`` was generated from and
require the homogenized reaction-force response (storage ``RF_Real``, loss ``RF_Imag``) and
the drive displacement ``U`` to match the oracle.

The matrix oracles are generated so this comparison is both interpolation-free (the sweep lands
exactly on the tabular table nodes) and inertia-free (tiny density), so what remains is the
discretization gap, held to ``RTOL`` or to a relaxation documented below. As FE features land and
``is_fe_green`` widens, the corresponding cells join this parity check automatically; a green cell
that disagrees with ABAQUS fails loudly here.

This complements ``test_matrix_dolfinx`` (which checks closed-form physics on a homogeneous
RVE and tracks the red cells as expected failures); here the RVE is the heterogeneous
oracle sim and the truth is ABAQUS.
"""
import pathlib
import unittest

import numpy as np

from tests._helpers import needs_complex, needs_dolfinx
from tests._matrix import (
    has_dynamic_step,
    hold_by_prescribed,
    is_fe_green,
    matrix_cases,
    matrix_simulation,
)

DATA = pathlib.Path(__file__).resolve().parent / "data"
RTOL = 1e-4  # robust to ABAQUS .dat frequency rounding

# Per-case relaxations, test name -> rtol, each entry commented with its discretization reason.
_RTOL_OVERRIDES = {}

# The finite-strain (hyperelastic) responses, periodic and standard cells alike. ABAQUS
# CPE4/C3D8 use the F-bar average-dilatation modification F_bar = F*(Jbar/J)^(1/n) with n the
# element dimension (in 2D plane strain an in-plane-only scaling, which changes the deviatoric
# invariants), while the backend's selective reduced integration (volumetric energy at the
# centroid) leaves them at the raw F. The two differ wherever J varies within an element and
# converge to each other under mesh refinement (docs/dolfinx_backend_design.md#parity-fidelity).
# The TODO in _FormsMixin._build_forms would close the gap by reproducing the F-bar pairing.
_HYPER_TYPES = ("reduced_polynomial", "polynomial", "arruda_boyce")
_HYPER_RTOL = 3e-2


def _oracle_name(cell, tt):
    return (f"oracle_{cell['dim']}d_{cell['mode']}_{cell['traction']}"
            f"_{cell['bc']}_{tt}.tsv")


class _OracleParity:
    def assertMatchesOracle(self, sim, cell, tt, rtol):
        """Solve ``sim`` and hold its rows to the committed ABAQUS oracle of matrix cell ``cell``
        under test type ``tt``."""
        from microstructure_ve.backends.dolfinx import _run as run

        dim = cell["dim"]
        oracle = np.atleast_2d(np.loadtxt(DATA / _oracle_name(cell, tt), skiprows=1))
        # stable sort so rows that share a frequency (a multi-step Static frame and a
        # Dynamic frame both at 1.0) keep their emission order -- the FE emits rows in the
        # same per-step/per-frame order as the ABAQUS reader, so they stay aligned.
        oracle = oracle[np.argsort(oracle[:, 0], kind="stable")]
        fe = run.run(sim)
        fe = fe[np.argsort(fe[:, 0], kind="stable")]
        self.assertEqual(fe.shape, oracle.shape)

        scale = float(np.max(np.abs(oracle[:, 1:1 + dim])))  # storage sets the abs floor
        storage = slice(1, 1 + dim)
        loss = slice(1 + dim, 1 + 2 * dim)
        drive = slice(1 + 2 * dim, 1 + 3 * dim)
        np.testing.assert_allclose(fe[:, storage], oracle[:, storage],
                                   rtol=rtol, atol=scale * rtol)
        np.testing.assert_allclose(fe[:, loss], oracle[:, loss],
                                   rtol=rtol, atol=scale * rtol)

        # Drive (U) column. Normally U is an exactly-reproduced imposed quantity (the periodic
        # corner displacement, or a prescribed face displacement) -> tight 1e-6. For a *standard*
        # plastic cell the reported U also sums the loaded face's transverse SLIP, which is a
        # solved output: the driven component stays exact but the transverse slip is
        # discretization-sensitive (the mean-dilatation plastic B-bar vs ABAQUS's CPE4 B-bar),
        # agreeing only to the relaxed tolerance below while the homogenized reaction above still
        # matches at the default tolerance. Relax the U column there (the periodic volume-averaged reaction is insensitive).
        if cell["bc"] == "standard" and tt.startswith("hyperelastic_plastic"):
            dscale = float(np.max(np.abs(oracle[:, drive])))
            np.testing.assert_allclose(fe[:, drive], oracle[:, drive],
                                       rtol=5e-3, atol=dscale * 5e-3)
        elif cell["bc"] == "standard" and tt in _HYPER_TYPES:
            # finite-strain transverse slip: same discretization sensitivity, amplified by
            # the finite strain (see _HYPER_RTOL above); the driven component
            # itself still reproduces exactly
            dscale = float(np.max(np.abs(oracle[:, drive])))
            np.testing.assert_allclose(fe[:, drive], oracle[:, drive],
                                       rtol=_HYPER_RTOL,
                                       atol=dscale * _HYPER_RTOL)
        else:
            np.testing.assert_allclose(fe[:, drive], oracle[:, drive], rtol=1e-6, atol=1e-9)


@needs_dolfinx
class MatrixParityTests(_OracleParity, unittest.TestCase):
    """One method per FE-supported cell is attached below."""


def _make_test(cell, tt, rtol):
    def test(self):
        sim = matrix_simulation(test_type=tt, **cell)  # the oracle's own sim
        self.assertMatchesOracle(sim, cell, tt, rtol)

    return test


_attached = 0
for _cell, _tt in matrix_cases():
    if is_fe_green(_cell, _tt):
        _name = (f"test_{_cell['dim']}d_{_cell['mode']}_{_cell['traction']}"
                 f"_{_cell['bc']}_{_tt}")
        if _tt in _HYPER_TYPES:
            # F-bar vs SRI formulation gap (see _HYPER_RTOL); a periodic drive column stays
            # at its exact check
            _rtol = _HYPER_RTOL
        else:
            _rtol = _RTOL_OVERRIDES.get(_name, RTOL)
        _method = _make_test(_cell, _tt, _rtol)
        if has_dynamic_step(_tt):
            _method = needs_complex(_method)
        setattr(MatrixParityTests, _name, _method)
        _attached += 1

assert _attached, "no FE-green matrix cells to check parity for"
del _cell, _tt


@needs_dolfinx
class ModelPrescribedHoldsTests(_OracleParity, unittest.TestCase):
    """A model-level zero ``Prescribed`` on a dof no step drives holds that dof, as a model-level
    ``Fixed`` there does and as ABAQUS does."""

    def test_periodic_lateral_corner_held_by_zero_prescribed(self):
        """The periodic confined cell with its lateral corner's normal held by a model-level zero
        ``Prescribed`` returns the confined oracle's rows, not the free-lateral ones."""
        cell = {"mode": "uniaxial_x", "traction": "confined_slip", "bc": "periodic", "dim": 2}
        sim = hold_by_prescribed(matrix_simulation(test_type="elastic", **cell), "X0Y1", 2, 0.0)
        self.assertMatchesOracle(sim, cell, "elastic", RTOL)

    def test_standard_lateral_face_held_by_zero_prescribed(self):
        """The standard confined cell with its lateral face's normal held by a model-level zero
        ``Prescribed`` solves and returns the confined oracle's rows."""
        cell = {"mode": "uniaxial_x", "traction": "confined_slip", "bc": "standard", "dim": 2}
        sim = hold_by_prescribed(matrix_simulation(test_type="elastic", **cell), "Y0ALL", 2, 0.0)
        self.assertMatchesOracle(sim, cell, "elastic", RTOL)


if __name__ == "__main__":
    unittest.main()
