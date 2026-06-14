"""Material models and their backend-neutral physics queries.

The classes hold material data plus *queries* both backends consume: ``complex_modulus``
(E*(f) for the FE backend) and the ABAQUS normalization helpers (``normalize_*``,
``apply_shift``) the ABAQUS emitter uses to build its tabular table. Emission of the
``*Material`` keyword blocks lives in the ABAQUS backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import ElementSet


@dataclass
class Material:
    elset: ElementSet
    density: float  # kg/micron^3
    poisson: float
    youngs: float  # MPa, long term, low freq modulus

    def complex_modulus(self, freqs):
        """Complex Young's modulus E*(f) at each frequency in ``freqs``.

        Backend-neutral material query (used by the DOLFINx backend; the ABAQUS
        path emits the table instead). Plain elastic materials are frequency-flat,
        so every entry is just ``youngs`` as a complex number, shaped like ``freqs``:

        >>> import numpy as np
        >>> from microstructure_ve.core import ElementSet
        >>> from microstructure_ve.materials import Material
        >>> mat = Material(ElementSet(1, np.array([1])), density=1.0,
        ...                poisson=0.3, youngs=5.0)
        >>> mat.complex_modulus(np.array([1e0, 1e3, 1e6]))
        array([5.+0.j, 5.+0.j, 5.+0.j])
        """
        return np.full(np.shape(freqs), self.youngs, dtype=complex)


@dataclass
class TabularViscoelasticMaterial(Material):
    freq: np.ndarray  # excitation freq in Hz
    youngs_cplx: np.ndarray  # complex youngs modulus
    shift: float = 0.0  # frequency shift induced relative to nominal properties
    left_broadening: float = 1.0  # 1 is no broadening
    right_broadening: float = 1.0  # 1 is no broadening

    def apply_shift(self):
        """Apply shift and broadening factors to frequency.

        left and right refer to frequencies below and above tand peak"""
        freq = np.log10(self.freq) - self.shift

        # shift relative to tand peak
        i = np.argmax(self.youngs_cplx.imag / self.youngs_cplx.real)
        f = freq[i]

        freq[:i] = self.left_broadening * (freq[:i] - f) + f
        freq[i:] = self.right_broadening * (freq[i:] - f) + f
        return 10**freq

    def normalize_modulus(self):
        """Convert to abaqus's preferred normalized moduli"""
        # Only works with frequency-dependent poisson's ratio
        shear_cplx = self.youngs_cplx / (2 * (1 + self.poisson))
        bulk_cplx = self.youngs_cplx / (3 * (1 - 2 * self.poisson))

        # special normalized shear modulus used by abaqus
        wgstar = np.empty_like(shear_cplx)
        shear_inf = shear_cplx[0].real
        wgstar.real = shear_cplx.imag / shear_inf
        wgstar.imag = 1 - shear_cplx.real / shear_inf

        # special normalized bulk modulus used by abaqus
        wkstar = np.empty_like(shear_cplx)
        bulk_inf = bulk_cplx[0].real
        wkstar.real = bulk_cplx.imag / bulk_inf
        wkstar.imag = 1 - bulk_cplx.real / bulk_inf

        return wgstar, wkstar

    def normalize_constant_bulk_modulus(self):
        # assume bulk modulus of glassy system
        bulk_inf = self.youngs_cplx[-1].real / (3 * (1 - 2 * self.poisson))
        shear_cplx = 3 * bulk_inf * self.youngs_cplx / (9 * bulk_inf - self.youngs_cplx)

        # special normalized shear modulus used by abaqus
        wgstar = np.empty_like(shear_cplx)
        shear_inf = shear_cplx[0].real
        wgstar.real = shear_cplx.imag / shear_inf
        wgstar.imag = 1 - shear_cplx.real / shear_inf

        # special normalized bulk modulus used by abaqus
        # if bulk_cplx = bulk_inf, wgstar is all zeros
        wkstar = np.zeros_like(shear_cplx)

        return wgstar, wkstar

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
        ``*VISCOELASTIC, FREQUENCY=TABULAR`` table this material emits.

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
class PronyViscoelasticMaterial(Material):
    shear_modulus_coefficients: np.ndarray
    bulk_modulus_coefficients: np.ndarray
    relaxation_times: np.ndarray
