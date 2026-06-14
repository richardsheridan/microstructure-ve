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

from microstructure_ve.backends.abaqus import write_inp

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


if __name__ == "__main__":
    unittest.main()
