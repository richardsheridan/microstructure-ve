"""Structural ``.inp`` assertions over the full feature matrix (no ABAQUS run needed).

For every matrix case (``matrix_cases()`` -- every cell x {elastic, viscoelastic} plus the
appended multi-step viscoelastic cases) we build the Simulation (which runs
``validate_constraints`` at Model construction -- a passing build is itself the over-
constraint check), emit the ``.inp`` in memory, and assert its structure: the ordered step
blocks (type + perturbation flag), the drive ``*Boundary, type=displacement`` on the right
corner/face and dof, the periodic vs standard ``*Equation`` presence, and the
traction-specific fixed DOFs.

These are parsed assertions, not per-cell byte goldens -- the committed ``synthetic.inp``
remains the formatting-stability guard.
"""
import unittest

from tests._matrix import (
    cell_expectations,
    emit_inp_text,
    matrix_cases,
    matrix_cells,
    matrix_simulation,
    parse_inp,
)


class MatrixAbaqusStructureTests(unittest.TestCase):
    def test_emitted_inp_structure(self):
        for cell, test_type in matrix_cases():
            with self.subTest(test_type=test_type, **cell):
                sim = matrix_simulation(test_type=test_type, **cell)
                parsed = parse_inp(emit_inp_text(sim))
                exp = cell_expectations(test_type=test_type, **cell)

                # ordered step blocks: [(step_type, perturbation), ...]
                self.assertEqual(parsed["steps"], exp["steps"])

                # boundary-condition family: periodic emits *Equation, standard doesn't
                self.assertEqual(parsed["has_periodic"], exp["has_periodic"])

                # the macro drive: a displacement BC on the expected nset + dof, with
                # the expected (signed) amplitude present (a zero baseline also exists)
                drives = [
                    d for d in parsed["displacement_boundary"]
                    if d[0] == exp["drive_name"] and d[1] == exp["drive_dof"] == d[2]
                ]
                self.assertTrue(
                    drives,
                    msg=f"no drive on {exp['drive_name']} dof {exp['drive_dof']}: "
                        f"{parsed['displacement_boundary']}",
                )
                self.assertTrue(
                    any(abs(d[3] - exp["drive_value"]) < 1e-12 for d in drives),
                    msg=f"drive amplitude {exp['drive_value']} not found in {drives}",
                )

                # traction encoding: each (nset, dof) is fixed iff expected
                for name, dof, should_be_fixed in exp["fixed_checks"]:
                    present = any(
                        b[0] == name and b[1] <= dof <= b[2] for b in parsed["boundary"]
                    )
                    self.assertEqual(
                        present, should_be_fixed,
                        msg=f"{name} dof {dof} fixed={present}, expected "
                            f"{should_be_fixed}; boundaries={parsed['boundary']}",
                    )

    def test_matrix_counts(self):
        # guards the pruning rule: 6 (mode,traction) pairs in 2D, 10 in 3D -> 32 cells
        cells = list(matrix_cells())
        self.assertEqual(sum(c["dim"] == 2 for c in cells), 6 * len(("periodic", "standard")))
        self.assertEqual(sum(c["dim"] == 3 for c in cells), 10 * len(("periodic", "standard")))
        self.assertEqual(len(cells), 32)


if __name__ == "__main__":
    unittest.main()
