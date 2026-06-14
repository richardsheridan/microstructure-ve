"""Backends that consume the solver-neutral spec dataclasses.

This package is deliberately import-light: each public entry point is a thin wrapper
that imports its (heavier, possibly optional) implementation lazily in the body. So
``import microstructure_ve.backends`` pulls in nothing beyond the standard library.

- ``write_inp`` / ``write_odb_reader`` -> the ABAQUS backend (``_abaqus``).
- ``run`` -> the DOLFINx backend (``_dolfinx``), which imports dolfinx lazily.
"""
from __future__ import annotations


def write_inp(simulation, file_or_path):
    """Emit ``simulation`` as an ABAQUS ``.inp`` to a path or open text file."""
    from ._abaqus.inp import write_inp as _impl

    return _impl(simulation, file_or_path)


def write_odb_reader(dest):
    """Write a standalone, abaqus-python-compatible ``read_abaqus_odb.py`` to ``dest``."""
    from ._abaqus.odb import write_odb_reader as _impl

    return _impl(dest)


def run(simulation, *args, **kwargs):
    """Solve ``simulation`` with the DOLFINx backend (requires the fenicsx env)."""
    from ._dolfinx.run import run as _impl

    return _impl(simulation, *args, **kwargs)
