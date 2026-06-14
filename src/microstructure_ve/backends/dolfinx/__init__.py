"""DOLFINx/FEniCSx FE backend (concern-separated private submodules).

Public API::

    from microstructure_ve.backends.dolfinx import run   # (and build_solver)

``run`` / ``build_solver`` are resolved lazily so that importing this package -- and its
pure-numpy ``_spec`` submodule -- never pulls in dolfinx. The dolfinx-touching modules
(``_assembly``, ``_constraints``, ``_solver``, ``_homogenize``, ``_run``) import
dolfinx/petsc4py/dolfinx_mpc at module top and are only loaded when ``run`` is used, so
this package stays importable under the numpy-only msve env.
"""
from __future__ import annotations

_LAZY = ("run", "build_solver")


def __getattr__(name):
    if name in _LAZY:
        from . import _run

        return getattr(_run, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_LAZY])
