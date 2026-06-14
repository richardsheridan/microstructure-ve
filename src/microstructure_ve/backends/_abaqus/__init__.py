"""ABAQUS backend: ``.inp`` emission and ``.odb`` extraction.

Kept import-light. Public entry points are exposed from ``microstructure_ve.backends``;
the ``.odb`` reader (``odb.py``) is import-gated on ``odbAccess`` and is meant to be run
under ABAQUS python (or emitted as a standalone script via ``write_odb_reader``).
"""
