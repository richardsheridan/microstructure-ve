"""Material physics queries: ``complex_modulus`` and the normalization it inverts.

These are correctness assertions (not characterization): ``complex_modulus`` must be
the exact inverse of ``normalize_constant_nu_modulus`` at the table nodes, and must
interpolate in log10(frequency), matching how ABAQUS rebuilds E*(f) from the emitted
``*VISCOELASTIC, FREQUENCY=TABULAR`` block.
"""
import pathlib
import unittest

import numpy as np

from microstructure_ve import (
    ElementSet,
    TabularViscoelasticMaterial,
    load_viscoelasticity,
)

PMMA_DATA = pathlib.Path(__file__).resolve().parent.parent / "PMMA_shifted_R10_data.txt"


def _elset():
    return ElementSet(1, np.array([1]))


def _pmma_material(shift=0.0, left=1.0, right=1.0, youngs=None):
    freq, youngs_cplx = load_viscoelasticity(PMMA_DATA)
    if youngs is None:
        youngs = youngs_cplx[0].real
    return (
        TabularViscoelasticMaterial(
            _elset(),
            density=1.18e-15,
            poisson=0.35,
            youngs=youngs,
            freq=freq,
            youngs_cplx=youngs_cplx,
            shift=shift,
            left_broadening=left,
            right_broadening=right,
        ),
        freq,
        youngs_cplx,
    )


class ComplexModulusRoundTrip(unittest.TestCase):
    def test_recovers_youngs_cplx_at_table_nodes(self):
        # youngs == youngs_cplx[0].real => complex_modulus at the (shifted) table
        # frequencies must recover the original complex Young's modulus exactly.
        mat, _, youngs_cplx = _pmma_material()
        E = mat.complex_modulus(mat.apply_shift())
        np.testing.assert_allclose(E, youngs_cplx, rtol=1e-10, atol=0)

    def test_roundtrip_survives_shift_and_broadening(self):
        # The reconstruction samples the same nodes regardless of where apply_shift
        # places them, so querying at apply_shift() still recovers youngs_cplx.
        mat, _, youngs_cplx = _pmma_material(shift=-4.0, left=1.8, right=1.5)
        E = mat.complex_modulus(mat.apply_shift())
        np.testing.assert_allclose(E, youngs_cplx, rtol=1e-10, atol=0)

    def test_youngs_scales_the_reconstruction(self):
        # complex_modulus rescales by self.youngs / youngs_cplx[0].real.
        _, _, youngs_cplx = _pmma_material()
        e0 = youngs_cplx[0].real
        mat, _, _ = _pmma_material(youngs=2.0 * e0)
        E = mat.complex_modulus(mat.apply_shift())
        np.testing.assert_allclose(E, 2.0 * youngs_cplx, rtol=1e-10, atol=0)


class ComplexModulusInterpolation(unittest.TestCase):
    def test_interpolates_in_log_frequency(self):
        # Two-point table: storage/loss chosen so wg-star is easy to reason about.
        freq = np.array([1.0e0, 1.0e2])
        youngs_cplx = np.array([10.0 + 1.0j, 6.0 + 3.0j])
        mat = TabularViscoelasticMaterial(
            _elset(), density=1.0, poisson=0.3,
            youngs=youngs_cplx[0].real, freq=freq, youngs_cplx=youngs_cplx,
        )
        wgstar, _ = mat.normalize_constant_nu_modulus()
        # query at the geometric mean (log-midpoint) -> average of the endpoints
        f_mid = np.array([10.0])
        wgi_mid = 0.5 * (wgstar.imag[0] + wgstar.imag[1])
        wgr_mid = 0.5 * (wgstar.real[0] + wgstar.real[1])
        expected = mat.youngs * ((1.0 - wgi_mid) + 1j * wgr_mid)
        got = mat.complex_modulus(f_mid)
        np.testing.assert_allclose(got, [expected], rtol=1e-12, atol=0)

    def test_log_interp_differs_from_linear_f(self):
        # Same table; the log-f interpolant at f=10 must NOT equal the linear-f one,
        # otherwise the "interpolate in log-frequency" contract is silently broken.
        freq = np.array([1.0e0, 1.0e2])
        youngs_cplx = np.array([10.0 + 1.0j, 6.0 + 3.0j])
        mat = TabularViscoelasticMaterial(
            _elset(), density=1.0, poisson=0.3,
            youngs=youngs_cplx[0].real, freq=freq, youngs_cplx=youngs_cplx,
        )
        got = mat.complex_modulus(np.array([10.0]))[0]
        # linear-in-f fraction at f=10 between 1 and 100
        frac = (10.0 - 1.0) / (100.0 - 1.0)
        wgstar, _ = mat.normalize_constant_nu_modulus()
        wgi_lin = wgstar.imag[0] + frac * (wgstar.imag[1] - wgstar.imag[0])
        wgr_lin = wgstar.real[0] + frac * (wgstar.real[1] - wgstar.real[0])
        linear = mat.youngs * ((1.0 - wgi_lin) + 1j * wgr_lin)
        self.assertGreater(abs(got - linear), 1e-3)


if __name__ == "__main__":
    unittest.main()
