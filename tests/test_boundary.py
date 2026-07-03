"""Boundary conditions, the periodic constraint, and over-constraint validation.

``validate_constraints`` (positive + both negative cases) and the periodic-equation
invariants are correctness assertions — they are the construction-time guarantees the
corner-driven PBC formulation depends on. The emitted-equation checks also lock the
term order and coefficients (dependent = first term with coeff +1), which the DOLFINx
MPC mapping relies on.
"""
import io
import unittest

import numpy as np

from microstructure_ve.backends.abaqus._inp import emit
from microstructure_ve.boundary import (
    BoundaryCondition,
    Fixed,
    PeriodicBoundaryConstraint,
    Prescribed,
    validate_constraints,
)
from microstructure_ve.core import GridNodes, NodeSet


def _parse_equations(text):
    """Yield [(token, dof, coeff), ...] for each *Equation block in emitted text."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().upper() == "*EQUATION":
            n = int(lines[i + 1])
            terms = []
            for j in range(i + 2, i + 2 + n):
                tok, dof, coeff = (s.strip() for s in lines[j].split(","))
                terms.append((tok, int(dof), float(coeff)))
            yield terms
            i += 2 + n
        else:
            i += 1


def _emit(obj):
    buf = io.StringIO()
    emit(obj, buf)
    return buf.getvalue()


def _corner_driven_bcs(nodes):
    pbc = PeriodicBoundaryConstraint(nodes=nodes)
    origin = nodes.nsets["X0Y0"]
    x_macro = nodes.nsets["X1Y0"]
    y_macro = nodes.nsets["X0Y1"]
    return pbc, [
        pbc,
        BoundaryCondition(origin, Fixed(dofs=[1, 2])),
        BoundaryCondition(x_macro, Fixed(dofs=[2])),
        BoundaryCondition(y_macro, Fixed(dofs=[1])),
    ]


class ValidateConstraintsTests(unittest.TestCase):
    def setUp(self):
        self.nodes = GridNodes(np.array([5, 5]), 1.0)

    def test_valid_corner_driven_passes(self):
        _, bcs = _corner_driven_bcs(self.nodes)
        validate_constraints(bcs)  # must not raise

    def test_empty_is_ok(self):
        validate_constraints([])

    def test_double_dependent_equation_raises(self):
        # two periodic constraints over the same grid duplicate every dependent dof
        a = PeriodicBoundaryConstraint(nodes=self.nodes)
        b = PeriodicBoundaryConstraint(nodes=self.nodes)
        with self.assertRaises(ValueError) as cm:
            validate_constraints([a, b])
        self.assertIn("more than one *Equation", str(cm.exception))

    def test_equation_dependent_also_prescribed_raises(self):
        # X1 edge is the dependent term of a periodic equation (dof 1); pinning it
        # with a *Boundary over-constrains that dof.
        pbc, bcs = _corner_driven_bcs(self.nodes)
        bcs.append(BoundaryCondition(self.nodes.nsets["X1"], Fixed(dofs=[1])))
        with self.assertRaises(ValueError) as cm:
            validate_constraints(bcs)
        self.assertIn("*Boundary", str(cm.exception))

    def test_double_boundary_tolerated(self):
        # baseline + step displacement on the same (free) dof is normal, not an error
        _, bcs = _corner_driven_bcs(self.nodes)
        x_macro = self.nodes.nsets["X1Y0"]
        bcs.append(BoundaryCondition(x_macro, Prescribed([1], 0.0)))
        bcs.append(BoundaryCondition(x_macro, Prescribed([1], 0.005)))
        validate_constraints(bcs)  # must not raise


class PeriodicEquationInvariants(unittest.TestCase):
    def _check(self, nodes):
        pbc = PeriodicBoundaryConstraint(nodes=nodes)
        dim = nodes.dim
        # one dependent-dof group per (pair, dof)
        self.assertEqual(len(list(pbc.dependent_dofs())), len(pbc.node_pairs) * dim)

        ref_nums = {
            str(nodes.nsets[c].node_inds[0])
            for c in ("X0Y0", "X1Y0", "X0Y1")
        }
        seen_dep = set()
        for terms in _parse_equations(_emit(pbc)):
            # 4-term u_dep - u_img = u_refHi - u_refLo with coefficients +1, -1, -1, +1
            self.assertEqual([t[2] for t in terms], [1.0, -1.0, -1.0, 1.0])
            self.assertAlmostEqual(sum(t[2] for t in terms), 0.0, places=9)
            dep = (terms[0][0], terms[0][1])
            self.assertNotIn(dep, seen_dep, "duplicate dependent term")
            seen_dep.add(dep)
            self.assertNotIn(terms[0][0], ref_nums, "reference corner used as dependent")

    def test_2d_invariants(self):
        self._check(GridNodes(np.array([5, 5]), 1.0))

    def test_2d_pair_and_equation_counts(self):
        nodes = GridNodes(np.array([5, 5]), 1.0)
        pbc = PeriodicBoundaryConstraint(nodes=nodes)
        self.assertEqual(len(pbc.node_pairs), 3)  # vertex pair + 2 edge pairs
        # one emitted *Equation per boundary node per dof
        n_boundary = sum(len(pair[0].node_inds) for pair in pbc.node_pairs)
        blocks = list(_parse_equations(_emit(pbc)))
        self.assertEqual(len(blocks), n_boundary * nodes.dim)

    def test_3d_pair_and_dependent_counts(self):
        nodes = GridNodes(np.array([4, 4, 4]), 1.0)
        pbc = PeriodicBoundaryConstraint(nodes=nodes)
        self.assertEqual(len(pbc.node_pairs), 16)  # 4 vertex + 9 edge + 3 face groups
        self.assertEqual(len(list(pbc.dependent_dofs())), len(pbc.node_pairs) * 3)


class PrescribedDofsTests(unittest.TestCase):
    def test_fixed_enumerates_each_dof(self):
        ns = NodeSet("S", np.array([2, 3]))
        groups = list(BoundaryCondition(ns, Fixed(dofs=[1, 2])).prescribed_dofs())
        dofs = sorted(d for _, d in groups)
        self.assertEqual(dofs, [1, 2])
        for inds, _ in groups:
            np.testing.assert_array_equal(inds, [2, 3])

    def test_prescribed_multi_dof(self):
        groups = list(BoundaryCondition(7, Prescribed([1, 2], 0.01)).prescribed_dofs())
        self.assertEqual(sorted(d for _, d in groups), [1, 2])
        for inds, _ in groups:
            np.testing.assert_array_equal(inds, [7])


class BoundaryEmissionTests(unittest.TestCase):
    """Characterize the emitted *Boundary text (bytes the golden .inp relies on)."""

    def test_fixed_block(self):
        ns = NodeSet("X0Y0", np.array([1]))
        text = _emit(BoundaryCondition(ns, Fixed(dofs=[1, 2])))
        self.assertEqual(text, "*Boundary\nX0Y0, 1, 1\nX0Y0, 2, 2\n")

    def test_prescribed_block(self):
        ns = NodeSet("X1Y0", np.array([5]))
        text = _emit(BoundaryCondition(ns, Prescribed(dofs=[1], value=0.005)))
        self.assertEqual(text, "*Boundary, type=displacement\nX1Y0, 1, 1, 0.005\n")


if __name__ == "__main__":
    unittest.main()
