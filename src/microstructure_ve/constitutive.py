"""Constitutive responses: the backend-neutral *physics* a ``Material`` has-a.

Each response is a small, standalone dataclass carrying the elastic constants it needs
(``poisson``/``youngs``, the long-term / relaxed moduli) plus the one query both backends
consume: ``complex_modulus(freqs)`` -> the complex Young's modulus E*(f). There is no
inheritance between responses -- a ``Material`` composes one of them (``Material.response``),
so adding a new physics is a new dataclass here plus one ABAQUS emit registration, and
touches no other backend. The ABAQUS backend also reads the tabular helpers
(``apply_shift`` / ``normalize_constant_nu_modulus``) to build its ``*Viscoelastic`` table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Hyperelastic helpers
# ---------------------------------------------------------------------------

def _d1_from_poisson(mu0: float, nu: float) -> float:
    """Bulk compressibility coefficient D1 from the small-strain shear modulus and Poisson's ratio.

    The isotropic small-strain bulk modulus is K0 = 2*mu0*(1+nu) / (3*(1-2*nu)); ABAQUS
    encodes compressibility for hyperelastic models through the coefficient D1 = 2/K0::

        D1 = 3*(1-2*nu) / (mu0*(1+nu))

    Note that the ABAQUS ``*HYPERELASTIC`` keyword's ``POISSON`` parameter is *illegal* when
    coefficients are given on data lines.  Compressibility must therefore be expressed via the
    Di data-line entries.  ``poisson`` here is converted to D1 (and Di>1 set to zero) unless
    the caller supplies ``d`` explicitly.  ``poisson``/``youngs`` also serve the small-strain
    duck-typed backend interface (DOLFINx / complex_modulus).
    """
    return 3.0 * (1.0 - 2.0 * nu) / (mu0 * (1.0 + nu))


@dataclass
class Elastic:
    """A linear-elastic (frequency-flat) response.

    ``poisson`` is Poisson's ratio and ``youngs`` the Young's modulus in MPa. The complex
    modulus is frequency-independent, so every entry is just ``youngs`` as a complex number,
    shaped like ``freqs``:

    >>> import numpy as np
    >>> from microstructure_ve.constitutive import Elastic
    >>> Elastic(poisson=0.3, youngs=5.0).complex_modulus(np.array([1e0, 1e3, 1e6]))
    array([5.+0.j, 5.+0.j, 5.+0.j])
    """

    poisson: float
    youngs: float  # MPa, long term, low freq modulus

    def complex_modulus(self, freqs):
        """Complex Young's modulus E*(f); frequency-flat, so ``youngs`` everywhere."""
        return np.full(np.shape(freqs), self.youngs, dtype=complex)


@dataclass
class Plastic:
    """A rate-independent, isotropic-hardening, von-Mises (J2) plastic response.

    The elastic part is the linear-elastic ``youngs``/``poisson`` (the ``*Elastic`` block);
    plasticity adds a hardening table of (yield stress, plastic strain) pairs that ABAQUS reads
    as a ``*Plastic`` block. The two lists are the table columns and must be the same length.
    Plasticity is amplitude/path-dependent, not frequency-dependent, so ``complex_modulus`` is
    the frequency-flat ``youngs`` (same as ``Elastic``).
    """

    poisson: float
    youngs: float  # MPa, long term, low freq modulus
    yield_stress: list  # MPa, von-Mises yield stress at each hardening point
    plastic_strain: list  # dimensionless equivalent plastic strain (first point usually 0)

    def __post_init__(self):
        if len(self.yield_stress) != len(self.plastic_strain):
            raise ValueError(
                f"yield_stress and plastic_strain must be the same length; got "
                f"{len(self.yield_stress)} and {len(self.plastic_strain)}"
            )

    def complex_modulus(self, freqs):
        """Complex Young's modulus E*(f); frequency-flat, so ``youngs`` everywhere."""
        return np.full(np.shape(freqs), self.youngs, dtype=complex)


@dataclass
class TabularViscoelastic:
    """A tabulated frequency-domain viscoelastic response.

    ``youngs`` is the long-term (relaxed) modulus feeding the ABAQUS ``*Elastic`` keyword;
    ``freq``/``youngs_cplx`` are the measured complex Young's modulus table, and
    ``shift``/``left_broadening``/``right_broadening`` shift and broaden it about the tan(delta)
    peak. Assumes a frequency-independent Poisson ratio.
    """

    poisson: float
    youngs: float  # MPa, long term, low freq modulus
    freq: np.ndarray  # excitation freq in Hz
    youngs_cplx: np.ndarray  # complex youngs modulus
    shift: float = 0.0  # frequency shift induced relative to nominal properties
    left_broadening: float = 1.0  # 1 is no broadening
    right_broadening: float = 1.0  # 1 is no broadening

    def apply_shift(self):
        """Apply shift and broadening factors to frequency.

        left and right refer to frequencies below and above tand peak"""
        freq = np.log10(self.freq) - self.shift

        # shift relative to tand peak; guard the division so a zero storage modulus
        # can't inject nan/inf and silently pin the "peak" at index 0
        tan_delta = np.divide(
            self.youngs_cplx.imag,
            self.youngs_cplx.real,
            out=np.zeros_like(self.youngs_cplx.real),
            where=self.youngs_cplx.real != 0,
        )
        i = int(np.argmax(tan_delta))
        f = freq[i]

        freq[:i] = self.left_broadening * (freq[:i] - f) + f
        freq[i:] = self.right_broadening * (freq[i:] - f) + f
        return 10**freq

    def normalize_constant_nu_modulus(self):
        # special normalized bulk modulus used by abaqus
        # if poisson's ratio is frequency-independent, then
        # youngs=shear=bulk when normalized
        wgstar = np.empty_like(self.youngs_cplx)
        youngs_inf = self.youngs_cplx[0].real
        wgstar.real = self.youngs_cplx.imag / youngs_inf
        wgstar.imag = 1 - self.youngs_cplx.real / youngs_inf

        return wgstar, wgstar

    def complex_modulus(self, freqs):
        """E*(f) reconstructed exactly as ABAQUS rebuilds it from the
        ``*VISCOELASTIC, FREQUENCY=TABULAR`` table this response emits.

        The normalized loss/storage (omega-g-star) table is written at the
        shifted/broadened frequencies ``apply_shift()``; ABAQUS reads it relative to
        the ``*Elastic`` modulus (``self.youngs``) and interpolates between table rows.
        We mirror that: interpolate omega-g-star onto ``freqs`` and rescale by
        ``self.youngs``. Interpolation is in log10(frequency) since the table spans
        many decades. Inverts ``normalize_constant_nu_modulus``.
        """
        wgstar, _ = self.normalize_constant_nu_modulus()
        table_freq = self.apply_shift()
        order = np.argsort(table_freq)
        log_f = np.log10(freqs)
        log_table = np.log10(table_freq[order])
        wgr = np.interp(log_f, log_table, wgstar.real[order])
        wgi = np.interp(log_f, log_table, wgstar.imag[order])
        # invert normalize_constant_nu_modulus: E_storage = (1 - wgi)*E0, E_loss = wgr*E0
        return self.youngs * ((1.0 - wgi) + 1j * wgr)


@dataclass
class PronyViscoelastic:
    """A generalized-Maxwell (Prony series) viscoelastic response.

    ``youngs``/``poisson`` are the **long-term (relaxed)** elastic constants; the series adds
    one Maxwell arm per entry: ``shear_modulus_coefficients`` (G_i) and
    ``bulk_modulus_coefficients`` (K_i) are the *absolute* moduli of each arm and
    ``relaxation_times`` (tau_i) their time constants -- the three arrays share one length.
    The ABAQUS backend emits these as a ``*Viscoelastic, frequency=PRONY`` block (ratios
    relative to the instantaneous moduli, with ``*Elastic, moduli=LONG TERM``); the DOLFINx
    backend consumes ``complex_modulus`` below.

    Note: the DOLFINx backend assumes a **frequency-independent Poisson ratio** (as its
    tabular path does), so it faithfully solves only constant-nu Prony materials -- those
    whose bulk arms are proportional to the shear arms (``K_i/K_inf == G_i/G_inf``). ABAQUS
    handles the general (frequency-dependent-nu) case.
    """

    poisson: float
    youngs: float  # MPa, long term, low freq modulus
    shear_modulus_coefficients: np.ndarray
    bulk_modulus_coefficients: np.ndarray
    relaxation_times: np.ndarray

    def complex_modulus(self, freqs):
        """Complex Young's modulus E*(f) of a generalized-Maxwell (Prony) solid.

        Backend-neutral material query (used by the DOLFINx backend; the ABAQUS path
        emits a ``*Viscoelastic, frequency=PRONY`` block instead). The shear and bulk
        relaxation moduli are Prony series about the long-term (relaxed) elastic moduli
        ``G_inf = youngs/(2(1+nu))`` and ``K_inf = youngs/(3(1-2nu))``::

            G*(w) = G_inf + sum_i G_i (j w tau_i)/(1 + j w tau_i)

        (same for ``K*`` with the bulk coefficients), evaluated at angular frequency
        ``w = 2 pi f``, then recombined into ``E* = 9 K* G* / (3 K* + G*)``. With no
        Prony terms this reduces to the frequency-flat elastic ``youngs``. Returns a
        complex array shaped like ``freqs``.

        >>> import numpy as np
        >>> from microstructure_ve.constitutive import PronyViscoelastic
        >>> resp = PronyViscoelastic(
        ...     poisson=0.3, youngs=3000.0,
        ...     shear_modulus_coefficients=np.array([2000.0]),
        ...     bulk_modulus_coefficients=np.array([1000.0]),
        ...     relaxation_times=np.array([1.0]))
        >>> E = resp.complex_modulus(np.array([1e-9, 1e9]))  # relaxed, then glassy
        >>> np.round(E.real, 3)
        array([3000.   , 7276.056])
        >>> bool(np.allclose(E.imag, 0.0, atol=1e-3))  # no loss at either extreme
        True
        """
        freqs = np.asarray(freqs, dtype=float)
        w = 2 * np.pi * freqs
        g_inf = self.youngs / (2 * (1 + self.poisson))
        k_inf = self.youngs / (3 * (1 - 2 * self.poisson))
        tau = np.asarray(self.relaxation_times, dtype=float)
        jwt = 1j * w[..., np.newaxis] * tau  # (..., n_terms)
        arms = jwt / (1 + jwt)
        g_star = g_inf + arms @ np.asarray(self.shear_modulus_coefficients, dtype=float)
        k_star = k_inf + arms @ np.asarray(self.bulk_modulus_coefficients, dtype=float)
        return 9 * k_star * g_star / (3 * k_star + g_star)


# ---------------------------------------------------------------------------
# Hyperelastic constitutive responses
# ---------------------------------------------------------------------------

# Valid coefficient counts for a Polynomial of order N=1..6.
# The number of distinct Cij with i>=0, j>=0, 1<=i+j<=N is N*(N+3)/2.
_POLYNOMIAL_VALID_LENGTHS = {N * (N + 3) // 2: N for N in range(1, 7)}
# {2:1, 5:2, 9:3, 14:4, 20:5, 27:6}


@dataclass
class ReducedPolynomial:
    """A reduced polynomial (Rivlin) hyperelastic response.

    The strain-energy density is W = sum_{k=1}^{N} C_{k0} * (I1 - 3)**k, where I1 is the
    first deviatoric strain invariant.  ``c`` holds [C10, C20, ..., CN0]; N is 1..6.

    **Small-strain equivalence:** mu0 = 2*C10; E_small = 2*mu0*(1+poisson).

    **Compressibility:** the ABAQUS ``*HYPERELASTIC`` keyword's ``POISSON`` parameter is
    *illegal* when coefficients are given on data lines.  Compressibility is therefore encoded
    through the Di coefficients (ABAQUS data-line column).  When ``d`` is not given, D1 is
    derived from ``poisson`` via D1 = 3*(1-2*nu) / (mu0*(1+nu)) (equivalently 2/K0 with
    K0 = 2*mu0*(1+nu) / (3*(1-2*nu))); higher Di (i>1) are set to zero, giving an
    ``effective d`` of length N = [D1, 0, ..., 0].  When ``d`` is supplied it is used
    verbatim and must have length N.

    ``poisson`` is required even when ``d`` is given: it feeds the small-strain
    duck-typed backend interface (``youngs``, ``complex_modulus``).

    ABAQUS keyword: ``*HYPERELASTIC, REDUCED POLYNOMIAL, N=<n>``
    """

    c: list  # [C10, C20, ..., CN0]; len must be 1..6
    poisson: float
    d: Optional[list] = None  # [D1, ..., DN]; if None, derived from poisson

    def __post_init__(self):
        n = len(self.c)
        if n < 1 or n > 6:
            raise ValueError(
                f"ReducedPolynomial requires 1 <= N <= 6 coefficient(s); got {n}"
            )
        if self.poisson >= 0.5:
            raise ValueError(
                f"poisson must be < 0.5 (fully incompressible materials require hybrid "
                f"elements, which are out of scope); got poisson={self.poisson}"
            )
        if self.d is not None and len(self.d) != n:
            raise ValueError(
                f"d must have length N={n} for ReducedPolynomial(N={n}); "
                f"got len(d)={len(self.d)}"
            )

    @property
    def n(self) -> int:
        """Polynomial order N (number of Ci0 terms)."""
        return len(self.c)

    @property
    def mu0(self) -> float:
        """Small-strain shear modulus: mu0 = 2*C10."""
        return 2.0 * self.c[0]

    @property
    def youngs(self) -> float:
        """Small-strain Young's modulus: E = 2*mu0*(1+nu)."""
        return 2.0 * self.mu0 * (1.0 + self.poisson)

    @property
    def d_coeffs(self) -> list:
        """Effective compressibility coefficients [D1, ..., DN].

        If ``d`` was supplied at construction it is returned verbatim; otherwise D1 is
        derived from ``poisson`` and higher Di are zero.
        """
        if self.d is not None:
            return list(self.d)
        D1 = _d1_from_poisson(self.mu0, self.poisson)
        return [D1] + [0.0] * (self.n - 1)

    def complex_modulus(self, freqs):
        """Complex Young's modulus E*(f); frequency-flat, so ``youngs`` everywhere."""
        return np.full(np.shape(freqs), self.youngs, dtype=complex)


@dataclass
class Polynomial:
    """A full polynomial (Rivlin) hyperelastic response.

    The strain-energy density is W = sum_{i+j>=1, i+j<=N} Cij*(I1-3)**i*(I2-3)**j, where
    I1, I2 are the first and second deviatoric strain invariants.  ``c`` holds the
    coefficients in ABAQUS data-line order: for each k = i+j from 1 to N with i decreasing,
    e.g. for N=2: [C10, C01, C20, C11, C02].  The total number of coefficients for order N
    is N*(N+3)/2; valid lengths for N=1..6 are 2, 5, 9, 14, 20, 27.

    **Small-strain equivalence:** mu0 = 2*(C10 + C01); E_small = 2*mu0*(1+poisson).

    **Compressibility:** same convention as ``ReducedPolynomial`` -- the ABAQUS ``POISSON``
    parameter is illegal with data lines; compressibility is encoded through Di.  When ``d``
    is not given, D1 = 3*(1-2*nu) / (mu0*(1+nu)) and higher Di are zero (length N).  When
    ``d`` is supplied it must have length N.

    ``poisson`` is required even when ``d`` is given: it feeds ``youngs``/``complex_modulus``.

    ABAQUS keyword: ``*HYPERELASTIC, POLYNOMIAL, N=<n>``
    """

    c: list  # ABAQUS data-line order; len must be in {2,5,9,14,20,27}
    poisson: float
    d: Optional[list] = None  # [D1, ..., DN]; if None, derived from poisson

    def __post_init__(self):
        if len(self.c) not in _POLYNOMIAL_VALID_LENGTHS:
            valid = sorted(_POLYNOMIAL_VALID_LENGTHS)
            raise ValueError(
                f"Polynomial c list length must be one of {valid} (= N*(N+3)/2 for N=1..6); "
                f"got {len(self.c)}"
            )
        n = _POLYNOMIAL_VALID_LENGTHS[len(self.c)]
        if self.poisson >= 0.5:
            raise ValueError(
                f"poisson must be < 0.5 (fully incompressible materials require hybrid "
                f"elements, which are out of scope); got poisson={self.poisson}"
            )
        if self.d is not None and len(self.d) != n:
            raise ValueError(
                f"d must have length N={n} for Polynomial(N={n}); "
                f"got len(d)={len(self.d)}"
            )

    @property
    def n(self) -> int:
        """Polynomial order N."""
        return _POLYNOMIAL_VALID_LENGTHS[len(self.c)]

    @property
    def mu0(self) -> float:
        """Small-strain shear modulus: mu0 = 2*(C10 + C01).

        In ABAQUS data-line order the first two entries are always C10 and C01.
        """
        return 2.0 * (self.c[0] + self.c[1])

    @property
    def youngs(self) -> float:
        """Small-strain Young's modulus: E = 2*mu0*(1+nu)."""
        return 2.0 * self.mu0 * (1.0 + self.poisson)

    @property
    def d_coeffs(self) -> list:
        """Effective compressibility coefficients [D1, ..., DN].

        If ``d`` was supplied at construction it is returned verbatim; otherwise D1 is
        derived from ``poisson`` and higher Di are zero.
        """
        if self.d is not None:
            return list(self.d)
        D1 = _d1_from_poisson(self.mu0, self.poisson)
        return [D1] + [0.0] * (self.n - 1)

    def complex_modulus(self, freqs):
        """Complex Young's modulus E*(f); frequency-flat, so ``youngs`` everywhere."""
        return np.full(np.shape(freqs), self.youngs, dtype=complex)


@dataclass
class ArrudaBoyce:
    """An Arruda-Boyce eight-chain hyperelastic response.

    The Arruda-Boyce model captures the finite extensibility of polymer chains.  ``mu`` is
    the initial shear modulus parameter (MPa) and ``lm`` is the locking stretch (chain
    extensibility limit, dimensionless > 0).

    **Small-strain shear modulus** (Taylor expansion of the Langevin-based strain energy in
    1/lm**2, truncated at the 5th term)::

        mu0 = mu * (1 + 3/(5*lm**2) + 99/(175*lm**4)
                      + 513/(875*lm**6) + 42039/(67375*lm**8))

    Small-strain Young's modulus: E = 2*mu0*(1+poisson).

    **Compressibility:** the ABAQUS ``POISSON`` parameter is illegal with data lines.
    When ``d`` is not given, D1 = 3*(1-2*nu) / (mu0*(1+nu)) and the effective d is [D1]
    (single entry; ArrudaBoyce always has N=1 for the volumetric term).  When ``d`` is
    supplied it must be a single-element list.

    ``poisson`` is required even when ``d`` is given.

    ABAQUS keyword: ``*HYPERELASTIC, ARRUDA-BOYCE``
    """

    mu: float    # initial shear modulus parameter, MPa
    lm: float    # locking stretch (chain extensibility)
    poisson: float
    d: Optional[list] = None  # [D1]; if None, derived from poisson

    def __post_init__(self):
        if self.poisson >= 0.5:
            raise ValueError(
                f"poisson must be < 0.5 (fully incompressible materials require hybrid "
                f"elements, which are out of scope); got poisson={self.poisson}"
            )
        if self.d is not None and len(self.d) != 1:
            raise ValueError(
                f"ArrudaBoyce d must be a single-element list; got len(d)={len(self.d)}"
            )

    @property
    def mu0(self) -> float:
        """Small-strain shear modulus from the 5-term Langevin expansion in 1/lm**2."""
        lm2 = self.lm ** 2
        return self.mu * (
            1.0
            + 3.0 / (5.0 * lm2)
            + 99.0 / (175.0 * lm2**2)
            + 513.0 / (875.0 * lm2**3)
            + 42039.0 / (67375.0 * lm2**4)
        )

    @property
    def youngs(self) -> float:
        """Small-strain Young's modulus: E = 2*mu0*(1+nu)."""
        return 2.0 * self.mu0 * (1.0 + self.poisson)

    @property
    def d_coeffs(self) -> list:
        """Effective compressibility coefficient [D1].

        If ``d`` was supplied at construction it is returned verbatim; otherwise D1 is
        derived from ``poisson``.
        """
        if self.d is not None:
            return list(self.d)
        return [_d1_from_poisson(self.mu0, self.poisson)]

    def complex_modulus(self, freqs):
        """Complex Young's modulus E*(f); frequency-flat, so ``youngs`` everywhere."""
        return np.full(np.shape(freqs), self.youngs, dtype=complex)
