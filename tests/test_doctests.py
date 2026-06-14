"""Fold module doctests into the single ``unittest discover`` run.

Simple, usage-illustrating checks live as doctests on the objects they document;
``load_tests`` appends a ``DocTestSuite`` for each pure module so the one command
``python -m unittest discover -s tests -t .`` runs unit tests and doctests together.

Modules that are import-gated on optional/abaqus-only dependencies are excluded
(e.g. the abaqus odb reader, which imports ``odbAccess``).
"""
import doctest
import importlib

# Pure, numpy-only modules whose docstrings carry doctests. After the package split
# this becomes the list of microstructure_ve submodules; today it is the monolith.
DOCTEST_MODULES = [
    "microstructure_ve",
]


def load_tests(loader, tests, ignore):
    for name in DOCTEST_MODULES:
        mod = importlib.import_module(name)
        tests.addTests(doctest.DocTestSuite(mod))
    return tests
