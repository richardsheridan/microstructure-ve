"""Which scalar arithmetic a solve needs, and the variables that name each build's interpreter.

PETSc is compiled for one scalar type, so a process holds either the real-scalar or the
complex-scalar build. A ``Dynamic`` step is a frequency-domain solve with a complex modulus and needs
the complex build; ``Static`` steps solve in either. This module imports only numpy and the spec, so
it loads without dolfinx. It reads ``os.environ`` at call time and never writes it.
"""
from __future__ import annotations

import os

import numpy as np

from microstructure_ve.steps import Dynamic

REAL_PYTHON_VAR = "MSVE_DOLFINX_REAL_PYTHON"
"""Names a Python interpreter whose PETSc is the real-scalar build."""

COMPLEX_PYTHON_VAR = "MSVE_DOLFINX_COMPLEX_PYTHON"
"""Names a Python interpreter whose PETSc is the complex-scalar build."""


def is_complex_scalar(scalar_type):
    """True iff ``scalar_type`` (for example ``PETSc.ScalarType``) is a complex type."""
    return np.dtype(scalar_type).kind == "c"


def complex_python():
    """The configured complex-build interpreter, or None when the variable is unset or empty."""
    return os.environ.get(COMPLEX_PYTHON_VAR) or None


def needs_complex(sim):
    """True iff any step of ``sim`` carries a ``Dynamic`` subsection."""
    return any(any(isinstance(s, Dynamic) for s in step.subsections) for step in sim.steps)


def refusal_message():
    """The message for a frequency-domain solve refused in a real-scalar build."""
    return (
        "A frequency-domain (Dynamic) solve needs complex arithmetic, but this Python's PETSc is "
        f"the real-scalar build. Set {COMPLEX_PYTHON_VAR} to a Python whose PETSc is the "
        "complex-scalar build, or run from that environment."
    )
