"""ABAQUS backend: ``.inp`` emission and ``.odb`` extraction.

Public API::

    from microstructure_ve.backends.abaqus import write_inp, write_odb_reader

Implementation lives in the private ``_inp`` / ``_odb`` modules (and ``_read_abaqus_odb``,
the standalone reader source, which is import-gated on ``odbAccess``). This package is
numpy-only -- it never imports dolfinx.
"""
from ._inp import write_inp
from ._odb import write_odb_reader

__all__ = ["write_inp", "write_odb_reader"]
