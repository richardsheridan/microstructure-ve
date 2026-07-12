"""Flat convenience namespace re-exporting the whole solver-neutral spec surface.

The package ``__init__`` deliberately re-exports nothing, so the canonical way to use the
library is to import from the focused submodules (``core``, ``materials``, ``boundary``,
``steps``, ``utils``). This module is for callers who prefer a single flat namespace::

    from microstructure_ve.api import *

It re-exports the spec dataclasses and helper functions only; the backends are imported
from ``microstructure_ve.backends.abaqus`` / ``microstructure_ve.backends.dolfinx``.
"""
from __future__ import annotations

from .boundary import (
    BoundaryCondition,
    Fixed,
    PeriodicBoundaryConstraint,
    Prescribed,
    validate_constraints,
)
from .constitutive import (
    ArrudaBoyce,
    Elastic,
    NeoHookean,
    Plastic,
    Polynomial,
    PronyViscoelastic,
    ReducedPolynomial,
    TabularViscoelastic,
)
from .core import ElementSet, GridElements, GridNodes, NodeSet
from .materials import Material
from .steps import Dynamic, Heading, Model, Simulation, Static, Step
from .utils import (
    assign_intph,
    in_sorted,
    load_matlab_microstructure,
    load_viscoelasticity,
    periodic_assign_intph,
)

__all__ = [
    # core
    "NodeSet", "GridNodes", "GridElements", "ElementSet",
    # materials
    "Material",
    # constitutive responses
    "Elastic", "Plastic", "TabularViscoelastic", "PronyViscoelastic",
    "ReducedPolynomial", "Polynomial", "ArrudaBoyce", "NeoHookean",
    # boundary
    "BoundaryCondition", "Fixed", "Prescribed",
    "PeriodicBoundaryConstraint", "validate_constraints",
    # steps
    "Heading", "Static", "Dynamic", "Step", "Model", "Simulation",
    # utils
    "in_sorted", "load_matlab_microstructure", "assign_intph",
    "periodic_assign_intph", "load_viscoelasticity",
]
