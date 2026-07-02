"""Golden-file regression on the ABAQUS ``.inp`` emission.

Builds a small synthetic multi-material RVE (mesh, elastic + tabular-viscoelastic
materials, periodic BCs, drive, dynamic step) and asserts its emitted ``.inp`` is
byte-identical to a committed golden. This is the automated guard that the package
split and (later) the emission extraction do not perturb output. The big
``example.inp``/``ms.npy`` byte-check stays a manual driver-level compare; no test
loads ``ms.npy``.
"""
import io
import pathlib
import unittest

import numpy as np

from microstructure_ve.backends.abaqus import write_inp
from microstructure_ve.backends.abaqus._inp import emit
from microstructure_ve.constitutive import Plastic
from microstructure_ve.core import ElementSet
from microstructure_ve.materials import Material

from tests._helpers import synthetic_simulation

GOLDEN = pathlib.Path(__file__).resolve().parent / "data" / "synthetic.inp"


class SyntheticInpRegression(unittest.TestCase):
    def test_emission_matches_golden(self):
        buf = io.StringIO()
        write_inp(synthetic_simulation(), buf)
        emitted = buf.getvalue()
        golden = GOLDEN.read_text(encoding="ascii")
        if emitted != golden:
            # show the first divergence to make a real regression debuggable
            e_lines, g_lines = emitted.splitlines(), golden.splitlines()
            for i, (a, b) in enumerate(zip(e_lines, g_lines)):
                if a != b:
                    self.fail(
                        f"emission diverges from golden at line {i}:\n"
                        f"  emitted: {a!r}\n  golden:  {b!r}"
                    )
            self.fail(
                f"emission length differs: emitted {len(e_lines)} lines, "
                f"golden {len(g_lines)} lines"
            )


class PlasticEmission(unittest.TestCase):
    def test_plastic_block_follows_elastic(self):
        mat = Material(
            ElementSet(3, np.array([1, 2])), density=2.65e-15,
            response=Plastic(poisson=0.15, youngs=5.0e5,
                             yield_stress=[250.0, 300.0], plastic_strain=[0.0, 0.02]),
        )
        buf = io.StringIO()
        emit(mat, buf)
        lines = buf.getvalue().splitlines()
        # the *Plastic block opens immediately after the *Elastic constants line
        ei = lines.index("*Elastic")
        self.assertEqual(lines[ei + 2], "*Plastic")
        self.assertEqual(lines[ei + 3], "2.500000e+02, 0.000000e+00")
        self.assertEqual(lines[ei + 4], "3.000000e+02, 2.000000e-02")


if __name__ == "__main__":
    unittest.main()
