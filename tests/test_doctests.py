"""Fold module doctests into the single ``unittest discover`` run.

Simple, usage-illustrating checks live as doctests on the objects they document;
``load_tests`` appends a ``DocTestSuite`` for each pure module so the one command
``python -m unittest discover -s tests -t .`` runs unit tests and doctests together.

Modules that are import-gated on optional/abaqus-only dependencies are excluded
(e.g. the abaqus odb reader, which imports ``odbAccess``).
"""
import doctest
import importlib
import unittest

# Pure, numpy-only modules whose docstrings carry doctests. The abaqus odb reader
# is excluded (it is import-gated on odbAccess).
DOCTEST_MODULES = [
    "microstructure_ve",
    "microstructure_ve.core",
    "microstructure_ve.constitutive",
    "microstructure_ve.boundary",
    "microstructure_ve.steps",
    "microstructure_ve.utils",
]


def load_tests(loader, tests, ignore):
    for name in DOCTEST_MODULES:
        mod = importlib.import_module(name)
        tests.addTests(doctest.DocTestSuite(mod))
    return tests


def _flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _flatten(test)
        else:
            yield test


class PackageDocstringDoctestTest(unittest.TestCase):
    def test_package_import_example_runs_as_a_doctest(self):
        """The import example in the package docstring runs in the doctest suite.

        ``DocTestSuite`` drops docstrings without examples, so a literal-block example that
        nothing executes fails here instead of passing vacuously.
        """
        suite = load_tests(unittest.TestLoader(), unittest.TestSuite(), None)
        ids = {test.id() for test in _flatten(suite)}
        self.assertIn("microstructure_ve", ids)
