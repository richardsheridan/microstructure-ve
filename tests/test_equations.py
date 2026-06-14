"""*Equation constraint dataclasses: dependent enumeration and emitted text.

The emitted-text checks are characterization: they lock the term order and coefficients
(slave = nsets[0] with coeff +1), which the DOLFINx MPC mapping relies on.
"""
import io
import unittest

import numpy as np

from microstructure_ve import (
    DriveEquation,
    EqualityEquation,
    NodeSet,
    SequentialDifferenceEquation,
)


def _emit(obj):
    buf = io.StringIO()
    obj.to_inp(buf)
    return buf.getvalue()


class SequentialDifferenceEquationTests(unittest.TestCase):
    def test_unequal_node_counts_raise(self):
        a = NodeSet("A", np.array([1, 2, 3]))
        b = NodeSet("B", np.array([4, 5]))
        with self.assertRaises(ValueError):
            SequentialDifferenceEquation([a, b, 10, 11], dof=1)

    def test_dependent_dofs_is_first_nset(self):
        a = NodeSet("A", np.array([1, 2]))
        b = NodeSet("B", np.array([3, 4]))
        eq = SequentialDifferenceEquation([a, b, 10, 11], dof=2)
        (inds, dof), = list(eq.dependent_dofs())
        np.testing.assert_array_equal(inds, [1, 2])
        self.assertEqual(dof, 2)

    def test_emits_four_term_equation_per_node_pair(self):
        a = NodeSet("A", np.array([1, 2]))
        b = NodeSet("B", np.array([3, 4]))
        text = _emit(SequentialDifferenceEquation([a, b, 10, 11], dof=1))
        self.assertEqual(text.count("*Equation"), 2)
        self.assertIn("1, 1, 1.", text)    # dependent term, coeff +1
        self.assertIn("3, 1, -1.", text)   # image term, coeff -1
        self.assertIn("10, 1, -1.", text)  # refHi
        self.assertIn("11, 1, 1.", text)   # refLo


class EqualityEquationTests(unittest.TestCase):
    def test_dependent_and_text(self):
        eq = EqualityEquation([5, 6], dof=2)
        (inds, dof), = list(eq.dependent_dofs())
        np.testing.assert_array_equal(inds, [5])
        self.assertEqual(dof, 2)
        text = _emit(eq)
        self.assertIn("5, 2, 1.", text)
        self.assertIn("6, 2, -1.", text)


class DriveEquationTests(unittest.TestCase):
    def test_three_term_with_drive_node(self):
        eq = DriveEquation([5, 6], dof=1, drive_node=99)
        text = _emit(eq)
        self.assertIn("5, 1, 1.", text)
        self.assertIn("6, 1, -1.", text)
        self.assertIn("99, 1, 1.", text)


if __name__ == "__main__":
    unittest.main()
