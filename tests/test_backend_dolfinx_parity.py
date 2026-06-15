"""FE-vs-ABAQUS parity for the DOLFINx backend (requires the fenicsx env).

Each committed oracle under ``tests/data/`` is the ABAQUS reaction-force tsv for a small
synthetic RVE; the FE backend solves the same Simulation (ignoring the corner BCs and
applying the loading via its own periodic constraints) at the oracle's frequencies, and
the homogenized x-response (storage RF_Real1, loss RF_Imag1, drive U1) must agree. The
oracles were generated once with ABAQUS CPE4 / C3D8; see tests/_helpers oracle builders.
"""
import pathlib
import unittest

import numpy as np

from tests._helpers import oracle_simulation_2d, oracle_simulation_3d

try:
    import dolfinx  # noqa: F401
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

needs_dolfinx = unittest.skipUnless(HAS_DOLFINX, "needs the fenicsx env (dolfinx)")
DATA = pathlib.Path(__file__).resolve().parent / "data"


@needs_dolfinx
class AbaqusParityTests(unittest.TestCase):
    def _check(self, sim, oracle_name, dim, rtol):
        oracle = np.loadtxt(DATA / oracle_name, skiprows=1)
        oracle = np.atleast_2d(oracle)
        oracle = oracle[np.argsort(oracle[:, 0])]
        from microstructure_ve.backends.dolfinx import _run as run

        # frequencies (the Dynamic sweep) and the lateral traction are read from the sim;
        # the oracle was generated from the same Dynamic subsection, so they line up.
        fe = run.run(sim)
        fe = fe[np.argsort(fe[:, 0])]
        scale = np.max(np.abs(oracle[:, 1]))  # storage magnitude sets the absolute floor
        # storage RF_Real1 (col 1), loss RF_Imag1 (col 1+dim), drive U1 (col 1+2*dim)
        np.testing.assert_allclose(fe[:, 1], oracle[:, 1], rtol=rtol, atol=scale * rtol)
        np.testing.assert_allclose(fe[:, 1 + dim], oracle[:, 1 + dim], rtol=rtol, atol=scale * rtol)
        np.testing.assert_allclose(fe[:, 1 + 2 * dim], oracle[:, 1 + 2 * dim], rtol=1e-6, atol=1e-12)

    def test_2d_confined_cpe4(self):
        self._check(oracle_simulation_2d("confined"),
                    "oracle_2d_confined.tsv", dim=2, rtol=2e-3)

    def test_2d_free_lateral_cpe4(self):
        self._check(oracle_simulation_2d("free"),
                    "oracle_2d_free.tsv", dim=2, rtol=2e-3)

    def test_3d_elastic_c3d8_machine_precision(self):
        # elastic -> no tabular interpolation -> near machine precision
        self._check(oracle_simulation_3d(),
                    "oracle_3d_elastic.tsv", dim=3, rtol=1e-5)

    def test_confined_stiffer_than_free(self):
        # cross-check inside the FE backend alone: free-lateral is softer (Poisson relief)
        from microstructure_ve.backends.dolfinx import _run as run

        sim_c = oracle_simulation_2d("confined")
        sim_f = oracle_simulation_2d("free")
        rf_c = run.run(sim_c)[0, 1]
        rf_f = run.run(sim_f)[0, 1]
        self.assertLess(rf_f, rf_c)  # free modulus < confined modulus


if __name__ == "__main__":
    unittest.main()
