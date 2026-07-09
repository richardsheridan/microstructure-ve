"""The convenience ``microstructure_ve.api`` namespace for ``import *`` users.

The package ``__init__`` is intentionally empty, but ``api`` re-exports the whole
solver-neutral spec surface (dataclasses + helpers) for people who want a flat namespace.
"""
import unittest

# The full public spec surface that the old flat ``from microstructure_ve import ...``
# provided -- api must expose exactly this (and define __all__ so ``import *`` works).
#
# TODO: when this software gets proper API documentation, EXPECTED (the public-surface
# contract) should be generated from that documentation, or the documentation generated
# from it -- so the docs and this test cannot drift out of sync.
EXPECTED = {
    "NodeSet", "GridNodes", "GridElements", "ElementSet",
    "Material",
    "Elastic", "Plastic", "TabularViscoelastic", "PronyViscoelastic",
    "ReducedPolynomial", "Polynomial", "ArrudaBoyce",
    "BoundaryCondition", "Fixed", "Prescribed",
    "PeriodicBoundaryConstraint", "validate_constraints",
    "Heading", "Static", "Dynamic", "Step", "Model", "Simulation",
    "in_sorted", "load_matlab_microstructure", "assign_intph",
    "periodic_assign_intph", "load_viscoelasticity",
}


class ApiNamespaceTests(unittest.TestCase):
    def test_star_import_exposes_full_spec(self):
        ns = {}
        exec("from microstructure_ve.api import *", ns)
        missing = EXPECTED - ns.keys()
        self.assertFalse(missing, f"missing from `import *`: {sorted(missing)}")

    def test_all_is_defined_and_consistent(self):
        import microstructure_ve.api as api

        self.assertTrue(hasattr(api, "__all__"), "api must define __all__")
        self.assertEqual(set(api.__all__), EXPECTED)
        for name in api.__all__:
            self.assertTrue(hasattr(api, name), f"{name} listed in __all__ but absent")

    def test_reexports_are_the_real_classes(self):
        from microstructure_ve.api import GridNodes
        from microstructure_ve.core import GridNodes as CoreGridNodes

        self.assertIs(GridNodes, CoreGridNodes)


if __name__ == "__main__":
    unittest.main()
