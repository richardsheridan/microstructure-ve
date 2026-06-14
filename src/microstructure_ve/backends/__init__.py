"""Backends that consume the solver-neutral spec dataclasses.

Each backend exposes its public API from its own subpackage; import from there directly::

    from microstructure_ve.backends.abaqus import write_inp, write_odb_reader
    from microstructure_ve.backends.dolfinx import run

This package itself re-exports nothing, so ``import microstructure_ve.backends`` pulls in
nothing beyond the standard library; the (possibly optional) dependencies are imported
only when you import the specific backend.
"""
