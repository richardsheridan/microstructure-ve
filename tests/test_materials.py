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
    ArrudaBoyce,
    NeoHookean,
    Plastic,
    Polynomial,
    PronyViscoelastic,
    ReducedPolynomial,
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


class HyperelasticMu0AndYoungs(unittest.TestCase):
    """Small-strain shear modulus mu0 and Young's modulus for the hyperelastic responses.

    mu0 is the linearised (Cauchy) shear modulus at zero strain; it differs between models:
      ReducedPolynomial: mu0 = 2*C10
      Polynomial:        mu0 = 2*(C10 + C01)
      ArrudaBoyce:       mu0 = mu * (1 + 3/(5*lm**2) + 99/(175*lm**4)
                                       + 513/(875*lm**6) + 42039/(67375*lm**8))
    The Young's modulus at small strain is youngs = 2*mu0*(1+poisson) (isotropic elasticity).
    """

    def test_reduced_polynomial_mu0(self):
        # For ReducedPolynomial with a single term, mu0 == 2*C10.
        C10 = 1.23
        rp = ReducedPolynomial(c=[C10], poisson=0.45)
        self.assertAlmostEqual(rp.mu0, 2 * C10, places=12)

    def test_reduced_polynomial_youngs(self):
        C10, nu = 1.23, 0.45
        rp = ReducedPolynomial(c=[C10], poisson=nu)
        self.assertAlmostEqual(rp.youngs, 2 * (2 * C10) * (1 + nu), places=12)

    def test_polynomial_mu0(self):
        # For Polynomial (N=1), c == [C10, C01] and mu0 == 2*(C10 + C01).
        C10, C01 = 0.8, 0.3
        poly = Polynomial(c=[C10, C01], poisson=0.4)
        self.assertAlmostEqual(poly.mu0, 2 * (C10 + C01), places=12)

    def test_polynomial_youngs(self):
        C10, C01, nu = 0.8, 0.3, 0.4
        poly = Polynomial(c=[C10, C01], poisson=nu)
        self.assertAlmostEqual(poly.youngs, 2 * (2 * (C10 + C01)) * (1 + nu), places=12)

    def test_arruda_boyce_mu0(self):
        # mu0 is the linearised shear from the Arruda-Boyce eight-chain model's Taylor series
        # in 1/lm**2 (lm = chain locking stretch).
        mu, lm = 2.5, 3.0
        ab = ArrudaBoyce(mu=mu, lm=lm, poisson=0.49)
        expected_mu0 = mu * (
            1
            + 3 / (5 * lm**2)
            + 99 / (175 * lm**4)
            + 513 / (875 * lm**6)
            + 42039 / (67375 * lm**8)
        )
        self.assertAlmostEqual(ab.mu0, expected_mu0, places=12)

    def test_arruda_boyce_youngs(self):
        mu, lm, nu = 2.5, 3.0, 0.49
        ab = ArrudaBoyce(mu=mu, lm=lm, poisson=nu)
        mu0 = mu * (
            1
            + 3 / (5 * lm**2)
            + 99 / (175 * lm**4)
            + 513 / (875 * lm**6)
            + 42039 / (67375 * lm**8)
        )
        self.assertAlmostEqual(ab.youngs, 2 * mu0 * (1 + nu), places=12)


class HyperelasticDCoefficients(unittest.TestCase):
    """Poisson-ratio-to-D derivation and explicit override.

    When d is not given, compressibility is encoded via the bulk modulus coefficient D1:
        K0 = 2*mu0*(1+nu) / (3*(1-2*nu))
        D1 = 2 / K0 = 3*(1-2*nu) / (mu0*(1+nu))
    For ReducedPolynomial/Polynomial with N>1 the effective list is [D1, 0, ..., 0] (length N).
    For ArrudaBoyce effective d is [D1] (length 1).
    The explicit d override uses the user-supplied list verbatim.
    """

    def _D1(self, mu0, nu):
        return 3 * (1 - 2 * nu) / (mu0 * (1 + nu))

    def test_reduced_polynomial_n1_derived_d(self):
        C10, nu = 1.23, 0.40
        rp = ReducedPolynomial(c=[C10], poisson=nu)
        mu0 = 2 * C10
        expected = [self._D1(mu0, nu)]
        self.assertEqual(rp.d_coeffs, expected)

    def test_reduced_polynomial_n2_derived_d_higher_zero(self):
        # N=2: d_coeffs is [D1, 0.0] -- higher Di are set to zero.
        C10, C20, nu = 1.0, 0.5, 0.45
        rp = ReducedPolynomial(c=[C10, C20], poisson=nu)
        mu0 = 2 * C10
        D1 = self._D1(mu0, nu)
        self.assertEqual(len(rp.d_coeffs), 2)
        self.assertAlmostEqual(rp.d_coeffs[0], D1, places=12)
        self.assertAlmostEqual(rp.d_coeffs[1], 0.0, places=12)

    def test_polynomial_n2_derived_d_higher_zero(self):
        # N=2 Polynomial: mu0 = 2*(C10+C01); d_coeffs is [D1, 0.0].
        C10, C01, C20, C11, C02 = 0.8, 0.3, 0.1, 0.05, 0.02
        nu = 0.40
        poly = Polynomial(c=[C10, C01, C20, C11, C02], poisson=nu)
        mu0 = 2 * (C10 + C01)
        D1 = self._D1(mu0, nu)
        self.assertEqual(len(poly.d_coeffs), 2)
        self.assertAlmostEqual(poly.d_coeffs[0], D1, places=12)
        self.assertAlmostEqual(poly.d_coeffs[1], 0.0, places=12)

    def test_arruda_boyce_derived_d(self):
        mu, lm, nu = 2.5, 3.0, 0.49
        ab = ArrudaBoyce(mu=mu, lm=lm, poisson=nu)
        mu0 = mu * (
            1 + 3 / (5 * lm**2) + 99 / (175 * lm**4)
            + 513 / (875 * lm**6) + 42039 / (67375 * lm**8)
        )
        D1 = self._D1(mu0, nu)
        self.assertEqual(len(ab.d_coeffs), 1)
        self.assertAlmostEqual(ab.d_coeffs[0], D1, places=12)

    def test_explicit_d_override_reduced_polynomial(self):
        # Explicit d of correct length is stored verbatim.
        rp = ReducedPolynomial(c=[1.0, 0.5], poisson=0.45, d=[0.1, 0.2])
        self.assertEqual(rp.d_coeffs, [0.1, 0.2])

    def test_explicit_d_override_polynomial(self):
        poly = Polynomial(c=[0.8, 0.3], poisson=0.4, d=[0.05])
        self.assertEqual(poly.d_coeffs, [0.05])

    def test_explicit_d_override_arruda_boyce(self):
        ab = ArrudaBoyce(mu=2.5, lm=3.0, poisson=0.49, d=[0.01])
        self.assertEqual(ab.d_coeffs, [0.01])

    def test_wrong_d_length_reduced_polynomial_raises(self):
        # d must match N (number of c terms) for ReducedPolynomial.
        with self.assertRaises(ValueError):
            ReducedPolynomial(c=[1.0, 0.5], poisson=0.45, d=[0.1])  # need length 2

    def test_wrong_d_length_polynomial_raises(self):
        # For Polynomial(N=2), d must have length 2.
        with self.assertRaises(ValueError):
            Polynomial(c=[0.8, 0.3, 0.1, 0.05, 0.02], poisson=0.40, d=[0.1, 0.2, 0.3])

    def test_wrong_d_length_arruda_boyce_raises(self):
        # ArrudaBoyce d must be length 1.
        with self.assertRaises(ValueError):
            ArrudaBoyce(mu=2.5, lm=3.0, poisson=0.49, d=[0.01, 0.02])


class HyperelasticValidation(unittest.TestCase):
    """Input validation for all three hyperelastic forms.

    poisson >= 0.5 (fully incompressible) is rejected because fully incompressible
    materials require hybrid (mixed-formulation) elements, which are out of scope.
    poisson is required even when d is given because it feeds the duck-typed backend
    interface (youngs, complex_modulus).
    Coefficient list length is validated against the allowed polynomial orders.
    """

    def test_incompressible_poisson_reduced_polynomial(self):
        with self.assertRaises(ValueError):
            ReducedPolynomial(c=[1.0], poisson=0.5)

    def test_incompressible_poisson_polynomial(self):
        with self.assertRaises(ValueError):
            Polynomial(c=[1.0, 0.0], poisson=0.5)

    def test_incompressible_poisson_arruda_boyce(self):
        with self.assertRaises(ValueError):
            ArrudaBoyce(mu=1.0, lm=3.0, poisson=0.5)

    def test_over_incompressible_poisson_raises(self):
        with self.assertRaises(ValueError):
            ReducedPolynomial(c=[1.0], poisson=0.6)

    def test_reduced_polynomial_invalid_c_length_zero(self):
        with self.assertRaises(ValueError):
            ReducedPolynomial(c=[], poisson=0.4)

    def test_reduced_polynomial_invalid_c_length_too_long(self):
        with self.assertRaises(ValueError):
            ReducedPolynomial(c=[1.0] * 7, poisson=0.4)

    def test_polynomial_invalid_c_length(self):
        # Valid c lengths for Polynomial are N*(N+3)/2 for N=1..6: 2,5,9,14,20,27.
        # Length 3 is not a valid count.
        with self.assertRaises(ValueError):
            Polynomial(c=[1.0, 0.0, 0.5], poisson=0.4)

    def test_polynomial_valid_lengths(self):
        # N=1: 2 terms, N=2: 5 terms -- smoke-test that construction succeeds.
        Polynomial(c=[1.0, 0.0], poisson=0.4)
        Polynomial(c=[1.0, 0.0, 0.5, 0.0, 0.0], poisson=0.4)

    def test_reduced_polynomial_n_property(self):
        rp = ReducedPolynomial(c=[1.0, 0.5, 0.1], poisson=0.4)
        self.assertEqual(rp.n, 3)

    def test_polynomial_n_property(self):
        # N=2 has 5 coefficients (C10,C01,C20,C11,C02).
        poly = Polynomial(c=[1.0, 0.0, 0.5, 0.0, 0.0], poisson=0.4)
        self.assertEqual(poly.n, 2)


class HyperelasticComplexModulus(unittest.TestCase):
    """complex_modulus(freqs) is frequency-flat and real == youngs.

    Hyperelastic materials have no frequency-dependent dissipation (the finite-strain
    elasticity is instantaneous), so the complex modulus is simply ``youngs + 0j``
    at every frequency -- mirroring the Plastic / Elastic pattern.
    """

    def test_reduced_polynomial_complex_modulus_flat(self):
        rp = ReducedPolynomial(c=[1.5], poisson=0.3)
        E = rp.complex_modulus(np.array([1e0, 1e6]))
        np.testing.assert_array_equal(E, [rp.youngs + 0j, rp.youngs + 0j])

    def test_polynomial_complex_modulus_flat(self):
        poly = Polynomial(c=[1.2, 0.4], poisson=0.35)
        E = poly.complex_modulus(np.array([1e-3, 1e9]))
        np.testing.assert_array_equal(E, [poly.youngs + 0j, poly.youngs + 0j])

    def test_arruda_boyce_complex_modulus_flat(self):
        ab = ArrudaBoyce(mu=2.0, lm=4.0, poisson=0.48)
        E = ab.complex_modulus(np.array([0.1, 1e5]))
        np.testing.assert_array_equal(E, [ab.youngs + 0j, ab.youngs + 0j])

    def test_complex_modulus_dtype_is_complex(self):
        rp = ReducedPolynomial(c=[1.0], poisson=0.3)
        E = rp.complex_modulus(np.array([1.0]))
        self.assertTrue(np.iscomplexobj(E))


class HyperelasticNeoHookeEquivalence(unittest.TestCase):
    """ReducedPolynomial(N=1) and Polynomial(N=1 with C01=0) are both Neo-Hookean.

    The Neo-Hooke model is the simplest hyperelastic form: W = C10*(I1-3). It is
    recovered by ReducedPolynomial([C10]) and by Polynomial([C10, 0.0]). Both must
    produce identical mu0, youngs, and d_coeffs.
    """

    def test_neo_hooke_youngs_equivalent(self):
        C10, nu = 0.9, 0.45
        rp = ReducedPolynomial(c=[C10], poisson=nu)
        poly = Polynomial(c=[C10, 0.0], poisson=nu)
        self.assertAlmostEqual(rp.youngs, poly.youngs, places=12)

    def test_neo_hooke_d_coeffs_equivalent(self):
        C10, nu = 0.9, 0.45
        rp = ReducedPolynomial(c=[C10], poisson=nu)
        poly = Polynomial(c=[C10, 0.0], poisson=nu)
        self.assertEqual(len(rp.d_coeffs), len(poly.d_coeffs))
        for a, b in zip(rp.d_coeffs, poly.d_coeffs):
            self.assertAlmostEqual(a, b, places=12)


class NeoHookeanTests(unittest.TestCase):
    """The coupled compressible Neo-Hookean response (Mechanical-MNIST / FEniCS-demo form).

    psi = mu/2*(I1 - 3) - mu*ln(J) + lam/2*((J**2 - 1)/2 - ln(J)) with the FULL first
    invariant I1 = tr(F^T F) -- not ABAQUS's isochoric I1bar. The class carries the
    small-strain constants directly: mu0 and lam are the classical Lame parameters of
    (youngs, poisson), which the coupled form linearizes to exactly.
    """

    def test_mu0_is_lame_mu(self):
        nh = NeoHookean(poisson=0.3, youngs=100.0)
        self.assertAlmostEqual(nh.mu0, 100.0 / (2 * 1.3), places=12)

    def test_lam_is_lame_lambda(self):
        nh = NeoHookean(poisson=0.3, youngs=100.0)
        self.assertAlmostEqual(nh.lam, 100.0 * 0.3 / (1.3 * 0.4), places=12)

    def test_complex_modulus_flat(self):
        nh = NeoHookean(poisson=0.3, youngs=42.0)
        E = nh.complex_modulus(np.array([1e-2, 1e4]))
        np.testing.assert_array_equal(E, [42.0 + 0j, 42.0 + 0j])
        self.assertTrue(np.iscomplexobj(E))

    def test_incompressible_poisson_raises(self):
        with self.assertRaises(ValueError):
            NeoHookean(poisson=0.5, youngs=1.0)


if __name__ == "__main__":
    unittest.main()
