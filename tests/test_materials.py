"""Constitutive-response physics queries: ``complex_modulus`` and the normalization it inverts.

These are correctness assertions (not characterization): the ``TabularViscoelastic``
response's ``complex_modulus`` must be the exact inverse of ``normalize_constant_nu_modulus``
at the table nodes, and must interpolate in log10(frequency), matching how ABAQUS rebuilds
E*(f) from the emitted ``*VISCOELASTIC, FREQUENCY=TABULAR`` block. They target the response
object directly -- that is the unit under test (a ``Material`` merely binds it to an elset).
"""
import pathlib
import unittest

import numpy as np

from microstructure_ve.constitutive import (
    Plastic,
    PronyViscoelastic,
    TabularViscoelastic,
)
from microstructure_ve.utils import load_viscoelasticity

PMMA_DATA = pathlib.Path(__file__).resolve().parent.parent / "PMMA_shifted_R10_data.txt"


def _pmma_material(shift=0.0, left=1.0, right=1.0, youngs=None):
    freq, youngs_cplx = load_viscoelasticity(PMMA_DATA)
    if youngs is None:
        youngs = youngs_cplx[0].real
    return (
        TabularViscoelastic(
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
        mat = TabularViscoelastic(
            poisson=0.3,
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
        mat = TabularViscoelastic(
            poisson=0.3,
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


class PlasticResponseFields(unittest.TestCase):
    def _mat(self, yield_stress, plastic_strain):
        return Plastic(
            poisson=0.15, youngs=5.0e5,
            yield_stress=yield_stress, plastic_strain=plastic_strain,
        )

    def test_constructs_and_round_trips_the_hardening_table(self):
        mat = self._mat([250.0, 300.0, 360.0], [0.0, 0.02, 0.05])
        self.assertEqual(mat.yield_stress, [250.0, 300.0, 360.0])
        self.assertEqual(mat.plastic_strain, [0.0, 0.02, 0.05])

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            self._mat([250.0, 300.0], [0.0])

    def test_complex_modulus_is_frequency_flat_elastic(self):
        # plasticity is amplitude/path-dependent, not frequency-dependent: the
        # complex_modulus query is just the elastic youngs at every frequency.
        mat = self._mat([250.0], [0.0])
        np.testing.assert_array_equal(
            mat.complex_modulus(np.array([1e0, 1e3])), [5.0e5 + 0j, 5.0e5 + 0j]
        )


class PronyComplexModulus(unittest.TestCase):
    """``complex_modulus`` for a generalized-Maxwell (Prony) response.

    E*(f) is built from the shear/bulk Prony series at angular frequency w = 2*pi*f and
    recombined as E* = 9 K* G* / (3 K* + G*). These pin the physical limits (relaxed at
    f->0, instantaneous at f->inf), the loss sign/shape, and the exact single-term value.
    """

    YOUNGS = 3000.0
    POISSON = 0.3

    def _mat(self, G=(2000.0,), K=(1000.0,), tau=(1.0,), youngs=None, poisson=None):
        return PronyViscoelastic(
            poisson=self.POISSON if poisson is None else poisson,
            youngs=self.YOUNGS if youngs is None else youngs,
            shear_modulus_coefficients=np.array(G, dtype=float),
            bulk_modulus_coefficients=np.array(K, dtype=float),
            relaxation_times=np.array(tau, dtype=float),
        )

    def test_low_frequency_recovers_youngs(self):
        # w->0: every Maxwell arm relaxes, so E* -> the long-term (elastic) youngs, real.
        E = self._mat().complex_modulus(np.array([1e-12]))
        np.testing.assert_allclose(E.real, [self.YOUNGS], rtol=1e-6)
        np.testing.assert_allclose(E.imag, [0.0], atol=1e-3)

    def test_high_frequency_recovers_instantaneous_modulus(self):
        # w->inf: every arm is glassy, so G* -> G_inf+sum(G_i), K* -> K_inf+sum(K_i),
        # and E* -> the instantaneous E_0 from those, real.
        G, K = (2000.0,), (1000.0,)
        mat = self._mat(G, K, (1.0,))
        G_inf = self.YOUNGS / (2 * (1 + self.POISSON))
        K_inf = self.YOUNGS / (3 * (1 - 2 * self.POISSON))
        G0, K0 = G_inf + sum(G), K_inf + sum(K)
        E0 = 9 * K0 * G0 / (3 * K0 + G0)
        E = mat.complex_modulus(np.array([1e12]))
        np.testing.assert_allclose(E.real, [E0], rtol=1e-6)
        np.testing.assert_allclose(E.imag, [0.0], atol=1e-3)

    def test_loss_is_nonnegative_and_vanishes_at_extremes(self):
        freqs = np.logspace(-6, 6, 25) / (2 * np.pi)  # w spans 1e-6 .. 1e6
        E = self._mat().complex_modulus(freqs)
        self.assertTrue(np.all(E.imag >= -1e-9))
        self.assertLess(E.imag[0], 1e-3 * E.real[0])
        self.assertLess(E.imag[-1], 1e-3 * E.real[-1])
        self.assertGreater(E.imag.max(), 1.0)  # a genuine loss peak in between

    def test_single_term_peaks_near_wtau_unity(self):
        tau = 1.0
        freqs = np.logspace(-3, 3, 601) / (2 * np.pi)  # w spans 1e-3 .. 1e3
        E = self._mat(G=(2000.0,), K=(1000.0,), tau=(tau,)).complex_modulus(freqs)
        w_peak = 2 * np.pi * freqs[np.argmax(E.imag)]
        # the E* recombination shifts the peak modestly off w*tau == 1, but keeps it O(1)
        self.assertGreater(w_peak * tau, 0.2)
        self.assertLess(w_peak * tau, 5.0)

    def test_matches_hand_computed_single_term(self):
        # single Maxwell arm at w*tau == 1: (j w tau)/(1 + j w tau) == j/(1+j).
        mat = self._mat(G=(2000.0,), K=(1000.0,), tau=(1.0,))
        f = 1.0 / (2 * np.pi)  # -> w = 1
        G_inf = self.YOUNGS / (2 * (1 + self.POISSON))
        K_inf = self.YOUNGS / (3 * (1 - 2 * self.POISSON))
        frac = 1j / (1 + 1j)
        Gs = G_inf + 2000.0 * frac
        Ks = K_inf + 1000.0 * frac
        expected = 9 * Ks * Gs / (3 * Ks + Gs)
        got = mat.complex_modulus(np.array([f]))[0]
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_no_prony_terms_is_flat_elastic(self):
        # no arms -> reduces to Elastic (frequency-flat elastic youngs).
        E = self._mat(G=(), K=(), tau=()).complex_modulus(np.array([1e-3, 1e0, 1e3]))
        np.testing.assert_allclose(E, [self.YOUNGS + 0j] * 3, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
