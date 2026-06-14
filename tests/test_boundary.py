"""Boundary conditions, periodic constraints, and over-constraint validation.

``validate_constraints`` (positive + both negative cases) and the periodic-equation
invariants are correctness assertions — they are the construction-time guarantees the
corner-driven PBC formulation depends on. The invariants mirror the ABAQUS-free
preflight checks that ``verify_pbc.sanity_check`` ran against the emitted ``.inp``.
"""
import io
import unittest

import numpy as np

from microstructure_ve import (
    DisplacementBoundaryCondition,
    FixedBoundaryCondition,
    GridNodes,
    NodeSet,
    OldPeriodicBoundaryCondition,
    PeriodicBoundaryCondition,
    validate_constraints,
)


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
    obj.to_inp(buf)
    return buf.getvalue()


def _corner_driven_bcs(nodes):
    pbc = PeriodicBoundaryCondition(nodes=nodes)
    origin = nodes.nsets["X0Y0"]
    x_macro = nodes.nsets["X1Y0"]
    y_macro = nodes.nsets["X0Y1"]
    return pbc, [
        pbc,
        FixedBoundaryCondition(origin, dofs=[1, 2]),
        FixedBoundaryCondition(x_macro, dofs=[2]),
        FixedBoundaryCondition(y_macro, dofs=[1]),
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
        a = PeriodicBoundaryCondition(nodes=self.nodes)
        b = PeriodicBoundaryCondition(nodes=self.nodes)
        with self.assertRaises(ValueError) as cm:
            validate_constraints([a, b])
        self.assertIn("more than one *Equation", str(cm.exception))

    def test_equation_dependent_also_prescribed_raises(self):
        # X1 edge is the dependent term of a periodic equation (dof 1); pinning it
        # with a *Boundary over-constrains that dof.
        pbc, bcs = _corner_driven_bcs(self.nodes)
        bcs.append(FixedBoundaryCondition(self.nodes.nsets["X1"], dofs=[1]))
        with self.assertRaises(ValueError) as cm:
            validate_constraints(bcs)
        self.assertIn("*Boundary", str(cm.exception))

    def test_double_boundary_tolerated(self):
        # baseline + step displacement on the same (free) dof is normal, not an error
        _, bcs = _corner_driven_bcs(self.nodes)
        x_macro = self.nodes.nsets["X1Y0"]
        bcs.append(DisplacementBoundaryCondition(x_macro, 1, 1, 0.0))
        bcs.append(DisplacementBoundaryCondition(x_macro, 1, 1, 0.005))
        validate_constraints(bcs)  # must not raise


class PeriodicEquationInvariants(unittest.TestCase):
    def _check(self, nodes):
        pbc = PeriodicBoundaryCondition(nodes=nodes)
        dim = nodes.dim
        self.assertEqual(len(pbc.equations), len(pbc.node_pairs) * dim)

        ref_nums = {
            str(nodes.nsets[c].node_inds[0])
            for c in ("X0Y0", "X1Y0", "X0Y1")
        }
        seen_dep = set()
        for terms in _parse_equations(_emit(pbc)):
            self.assertAlmostEqual(sum(t[2] for t in terms), 0.0, places=9)
            dep = (terms[0][0], terms[0][1])
            self.assertNotIn(dep, seen_dep, "duplicate dependent term")
            seen_dep.add(dep)
            self.assertNotIn(terms[0][0], ref_nums, "reference corner used as dependent")

    def test_2d_invariants(self):
        self._check(GridNodes(np.array([5, 5]), 1.0))

    def test_3d_equation_count(self):
        nodes = GridNodes(np.array([4, 4, 4]), 1.0)
        pbc = PeriodicBoundaryCondition(nodes=nodes)
        self.assertEqual(len(pbc.equations), len(pbc.node_pairs) * 3)


class PrescribedDofsTests(unittest.TestCase):
    def test_fixed_enumerates_each_dof(self):
        ns = NodeSet("S", np.array([2, 3]))
        groups = list(FixedBoundaryCondition(ns, dofs=[1, 2]).prescribed_dofs())
        dofs = sorted(d for _, d in groups)
        self.assertEqual(dofs, [1, 2])
        for inds, _ in groups:
            np.testing.assert_array_equal(inds, [2, 3])

    def test_displacement_range(self):
        groups = list(DisplacementBoundaryCondition(7, 1, 2, 0.01).prescribed_dofs())
        self.assertEqual(sorted(d for _, d in groups), [1, 2])


class OldPeriodicBoundaryConditionTests(unittest.TestCase):
    def test_constructs_and_enumerates(self):
        nodes = GridNodes(np.array([4, 4]), 1.0)
        drive = NodeSet("DRIVE", [nodes.virtual_node])
        old = OldPeriodicBoundaryCondition(
            nodes=nodes, nset=drive, first_dof=1, last_dof=1, displacement=0.0
        )
        deps = list(old.dependent_dofs())
        self.assertTrue(deps)
        # the driven dof is carried by a 3-term DriveEquation in the emitted text
        text = _emit(old)
        self.assertIn("*Equation\n3\n", text)


if __name__ == "__main__":
    unittest.main()
