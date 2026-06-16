"""Stub test types: viscoelastic/transient (hyperelastic-plastic is now implemented).

The remaining stub is on the matrix but not implemented. The builder refuses it with
``NotImplementedError`` (asserted here, an active guard), and a skipped placeholder
documents the intended assertion -- the spot to fill in when the feature lands.
"""
import unittest

from tests._matrix import STUB_TEST_TYPES, matrix_simulation


class StubTestTypeGuardTests(unittest.TestCase):
    def test_builder_refuses_stub_test_types(self):
        for test_type in STUB_TEST_TYPES:
            with self.subTest(test_type=test_type):
                with self.assertRaises(NotImplementedError):
                    matrix_simulation("uniaxial_x", "confined_slip", "periodic", 2,
                                      test_type=test_type)


class StubTestTypePlaceholders(unittest.TestCase):
    @unittest.skip("stub: documents the intended viscoelastic/transient API")
    def test_viscoelastic_transient(self):
        # When implemented: a time-domain (transient) viscoelastic step; assert the
        # relaxation/creep response rather than a steady-state frequency sweep.
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
