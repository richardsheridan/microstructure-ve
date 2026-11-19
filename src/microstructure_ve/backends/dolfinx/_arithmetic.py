"""Which scalar arithmetic a solve needs, and the variables that name each build's interpreter.

PETSc is compiled for one scalar type, so a process holds either the real-scalar or the
complex-scalar build. A ``Dynamic`` step is a frequency-domain solve with a complex modulus and needs
the complex build; ``Static`` steps solve in either. This module imports only numpy and the spec, so
it loads without dolfinx.
"""
from __future__ import annotations

import numpy as np

from microstructure_ve.steps import Dynamic

REAL_PYTHON_VAR = "MSVE_DOLFINX_REAL_PYTHON"
"""Names a Python interpreter whose PETSc is the real-scalar build."""

COMPLEX_PYTHON_VAR = "MSVE_DOLFINX_COMPLEX_PYTHON"
"""Names a Python interpreter whose PETSc is the complex-scalar build."""


def is_complex_scalar(scalar_type):
    """True iff ``scalar_type`` (for example ``PETSc.ScalarType``) is a complex type."""
    return np.dtype(scalar_type).kind == "c"


def needs_complex_build(sim):
    """True iff any step of ``sim`` carries a ``Dynamic`` subsection, a frequency-domain solve."""
    return any(any(isinstance(s, Dynamic) for s in step.subsections) for step in sim.steps)


def peer_variable(sim, complex_build):
    """The variable naming the interpreter that solves ``sim`` in place of this build, or None.

    A complex build hands a sim with no ``Dynamic`` step to the real build; a real build hands a
    sim with any ``Dynamic`` step to the complex build. ``complex_build`` is this process's build.
    """
    if needs_complex_build(sim):
        return None if complex_build else COMPLEX_PYTHON_VAR
    return REAL_PYTHON_VAR if complex_build else None


def native_static_warning():
    """The message for a Static step a complex build solves itself, no real interpreter named."""
    return (
        "Solving a Static step in complex arithmetic because no real-scalar PETSc build is "
        f"configured. Set {REAL_PYTHON_VAR} to a Python whose PETSc is the real-scalar build to "
        "solve it there."
    )


_NEEDS_COMPLEX = (
    "A frequency-domain (Dynamic) solve needs complex arithmetic, but this Python's PETSc is the "
    "real-scalar build."
)


def refusal_message():
    """The message for a frequency-domain solve ``run`` refuses in a real-scalar build."""
    return (
        f"{_NEEDS_COMPLEX} Set {COMPLEX_PYTHON_VAR} to a Python whose PETSc is the complex-scalar "
        "build, or run from that environment."
    )


def in_process_refusal_message():
    """The message for a frequency-domain ``solve_one(f)`` refused in a real-scalar build."""
    return (
        f"{_NEEDS_COMPLEX} build_solver solves in this process, so {COMPLEX_PYTHON_VAR} does not "
        "apply to it: call it from a Python whose PETSc is the complex-scalar build, or use run, "
        f"which hands the sim to the interpreter {COMPLEX_PYTHON_VAR} names."
    )
