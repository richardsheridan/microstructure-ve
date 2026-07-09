"""Exact-block-text assertions for hyperelastic material emission and nlgeom Step flag.

Part A of the hyperelastic TDD suite. These tests must run RED before the implementation
lands (Part B). Each test emits a single Material or Step into an in-memory buffer and
checks exact keyword/data-line text.
"""
import io
import unittest

import numpy as np

from microstructure_ve.backends.abaqus._inp import emit
from microstructure_ve.constitutive import ArrudaBoyce, Polynomial, ReducedPolynomial
from microstructure_ve.core import ElementSet
from microstructure_ve.materials import Material
from microstructure_ve.steps import Static, Step


def _emit_mat(response):
    """Emit a single Material with the given response into a string."""
    elset = ElementSet(1, np.array([1, 2]))
    mat = Material(elset, density=1.2e-15, response=response)
    buf = io.StringIO()
    emit(mat, buf)
    return buf.getvalue()


def _emit_step(step):
    """Emit a Step into a string."""
    buf = io.StringIO()
    emit(step, buf)
    return buf.getvalue()


class ArrudaBoyceEmissionTests(unittest.TestCase):
    def setUp(self):
        self.response = ArrudaBoyce(mu=1000.0, lm=2.0, poisson=0.45)
        self.text = _emit_mat(self.response)
        self.lines = self.text.splitlines()

    def test_hyperelastic_keyword_present(self):
        self.assertIn("*Hyperelastic, arruda-boyce", self.text)

    def test_no_elastic_keyword(self):
        # hyperelastic materials must NOT emit *Elastic
        self.assertNotIn("*Elastic", self.text)

    def test_no_poisson_keyword(self):
        # POISSON parameter on *Hyperelastic is illegal with data lines
        self.assertNotIn("POISSON", self.text.upper().replace("POISSON", "").upper(),
                         msg="POISSON should not appear anywhere in the emitted text")
        # simpler check
        for line in self.lines:
            self.assertNotIn("POISSON", line.upper(),
                             msg=f"POISSON found on line: {line!r}")

    def test_data_line_format(self):
        """One data line: mu, lm, D1 in %.6e format with ', ' separator."""
        r = self.response
        d1 = r.d_coeffs[0]
        expected = f"{r.mu:.6e}, {r.lm:.6e}, {d1:.6e}"
        self.assertIn(expected, self.text)

    def test_exactly_one_data_line_after_keyword(self):
        """Exactly one data line between the *Hyperelastic keyword and the next keyword."""
        kw_idx = next(
            i for i, l in enumerate(self.lines) if "*Hyperelastic, arruda-boyce" in l
        )
        # first non-blank line after keyword is the data line
        data_line = self.lines[kw_idx + 1]
        # next line after that should start a new keyword or end the block
        next_line = self.lines[kw_idx + 2] if kw_idx + 2 < len(self.lines) else ""
        self.assertFalse(
            data_line.startswith("*"),
            msg=f"Expected data line after keyword, got: {data_line!r}",
        )
        self.assertTrue(
            next_line == "" or next_line.startswith("*"),
            msg=f"Expected keyword or end after single data line, got: {next_line!r}",
        )


class ReducedPolynomialN1EmissionTests(unittest.TestCase):
    def setUp(self):
        self.response = ReducedPolynomial(c=[500.0], poisson=0.45)
        self.text = _emit_mat(self.response)
        self.lines = self.text.splitlines()

    def test_hyperelastic_keyword_present(self):
        self.assertIn("*Hyperelastic, reduced polynomial, n=1", self.text)

    def test_no_elastic_keyword(self):
        self.assertNotIn("*Elastic", self.text)

    def test_no_poisson_in_text(self):
        for line in self.lines:
            self.assertNotIn("POISSON", line.upper(),
                             msg=f"POISSON found on line: {line!r}")

    def test_data_line_c10_d1(self):
        """One data line: C10, D1 in %.6e format."""
        r = self.response
        c10 = r.c[0]
        d1 = r.d_coeffs[0]
        expected = f"{c10:.6e}, {d1:.6e}"
        self.assertIn(expected, self.text)


class PolynomialN2EmissionTests(unittest.TestCase):
    def setUp(self):
        # N=2: 5 coefficients [C10, C01, C20, C11, C02]
        self.response = Polynomial(c=[400.0, 100.0, 20.0, 10.0, 5.0], poisson=0.45)
        self.text = _emit_mat(self.response)
        self.lines = self.text.splitlines()

    def test_hyperelastic_keyword_present(self):
        self.assertIn("*Hyperelastic, polynomial, n=2", self.text)

    def test_no_elastic_keyword(self):
        self.assertNotIn("*Elastic", self.text)

    def test_no_poisson_in_text(self):
        for line in self.lines:
            self.assertNotIn("POISSON", line.upper(),
                             msg=f"POISSON found on line: {line!r}")

    def test_data_line_7_values(self):
        """One data line of 7 values: C10, C01, C20, C11, C02, D1, D2."""
        r = self.response
        vals = list(r.c) + r.d_coeffs  # 5 + 2 = 7 values
        expected = ", ".join(f"{v:.6e}" for v in vals)
        self.assertIn(expected, self.text)


class PolynomialN3WrappingTests(unittest.TestCase):
    def setUp(self):
        # N=3: 9 coefficients [C10, C01, C20, C11, C02, C30, C21, C12, C03]
        self.response = Polynomial(
            c=[400.0, 100.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25],
            poisson=0.45,
        )
        self.text = _emit_mat(self.response)
        self.lines = self.text.splitlines()

    def test_hyperelastic_keyword_present(self):
        self.assertIn("*Hyperelastic, polynomial, n=3", self.text)

    def test_exactly_two_data_lines(self):
        """9 Cij + 3 Di = 12 values -> 8 on line 1, 4 on line 2."""
        kw_idx = next(
            i for i, l in enumerate(self.lines) if "*Hyperelastic, polynomial, n=3" in l
        )
        line1 = self.lines[kw_idx + 1]
        line2 = self.lines[kw_idx + 2]
        # third line should be a keyword or end
        next_after = self.lines[kw_idx + 3] if kw_idx + 3 < len(self.lines) else ""
        self.assertFalse(line1.startswith("*"), msg=f"Expected data, got: {line1!r}")
        self.assertFalse(line2.startswith("*"), msg=f"Expected data, got: {line2!r}")
        self.assertTrue(
            next_after == "" or next_after.startswith("*"),
            msg=f"Expected keyword after 2 data lines, got: {next_after!r}",
        )

    def test_first_line_8_values(self):
        """First data line has exactly 8 values."""
        r = self.response
        vals = list(r.c) + r.d_coeffs  # 9 + 3 = 12 values
        expected_line1 = ", ".join(f"{v:.6e}" for v in vals[:8])
        self.assertIn(expected_line1, self.text)

    def test_second_line_remaining_values(self):
        """Second data line has the remaining 4 values: C03, D1, D2, D3."""
        r = self.response
        vals = list(r.c) + r.d_coeffs  # 12 total
        expected_line2 = ", ".join(f"{v:.6e}" for v in vals[8:])
        self.assertIn(expected_line2, self.text)


class StepNlgeomTests(unittest.TestCase):
    def test_nlgeom_true_emits_yes(self):
        step = Step(subsections=[Static()], nlgeom=True)
        text = _emit_step(step)
        step_line = next(l for l in text.splitlines() if l.startswith("*STEP"))
        self.assertIn("nlgeom=YES", step_line)

    def test_nlgeom_false_default_no_nlgeom(self):
        step = Step(subsections=[Static()])
        text = _emit_step(step)
        step_line = next(l for l in text.splitlines() if l.startswith("*STEP"))
        self.assertNotIn("nlgeom", step_line.lower())

    def test_nlgeom_false_explicit_no_nlgeom(self):
        step = Step(subsections=[Static()], nlgeom=False)
        text = _emit_step(step)
        step_line = next(l for l in text.splitlines() if l.startswith("*STEP"))
        self.assertNotIn("nlgeom", step_line.lower())

    def test_nlgeom_with_perturbation(self):
        """nlgeom=YES composes correctly with PERTURBATION."""
        step = Step(subsections=[Static()], perturbation=True, nlgeom=True)
        text = _emit_step(step)
        step_line = next(l for l in text.splitlines() if l.startswith("*STEP"))
        self.assertIn("PERTURBATION", step_line)
        self.assertIn("nlgeom=YES", step_line)


if __name__ == "__main__":
    unittest.main()
