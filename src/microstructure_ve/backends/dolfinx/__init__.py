"""DOLFINx/FEniCSx FE backend (concern-separated private submodules).

Public API::

    from microstructure_ve.backends.dolfinx import run, clear_cache  # (and build_solver)
    from microstructure_ve.backends.dolfinx import Cancelled, ConvergenceError

``run`` transparently caches the mesh-only FE setup by mesh shape, so sweeping many
simulations that share a grid is fast; ``clear_cache`` releases that memory.

Every entry point is thread-safe; which calls serialize, that they run single-core, and how a
``run(workers=N)`` sweep differs is the concurrency contract in ``_run``'s module docstring.

``run`` / ``build_solver`` are resolved lazily so that importing this package -- and its pure-numpy
``_spec`` and ``_loading`` submodules -- never pulls in dolfinx. The other submodules import
dolfinx/petsc4py/dolfinx_mpc at module top and are only loaded when ``run`` is used, so this package
stays importable under the numpy-only msve env.
"""
from __future__ import annotations

_LAZY = ("run", "build_solver", "Cancelled", "ConvergenceError", "clear_cache")

# The FE stack these modules import at the top; a missing one means "wrong env".
_FE_DEPS = {"dolfinx", "dolfinx_mpc", "basix", "ufl", "petsc4py", "mpi4py", "threadpoolctl"}


def __getattr__(name):
    if name in _LAZY:
        try:
            from . import _run
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] in _FE_DEPS:
                raise ImportError(
                    "The DOLFINx backend requires the FEniCSx stack (dolfinx, "
                    "dolfinx_mpc, basix, threadpoolctl, and a complex-scalar petsc4py build), "
                    f"which is not importable in this Python environment (missing {e.name!r}). "
                    "These are typically installed via conda-forge rather than pip; run "
                    "from an environment that provides them."
                ) from e
            raise  # an unrelated missing module: surface the real error
        return getattr(_run, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_LAZY])
