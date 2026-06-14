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
    BoundaryConditions,
    DisplacementBoundaryCondition,
    FixedBoundaryCondition,
    OldPeriodicBoundaryCondition,
    PeriodicBoundaryCondition,
    validate_constraints,
)
from .core import ElementSet, GridElements, GridNodes, NodeSet
from .equations import DriveEquation, EqualityEquation, SequentialDifferenceEquation
from .materials import Material, PronyViscoelasticMaterial, TabularViscoelasticMaterial
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
    # equations
    "SequentialDifferenceEquation", "EqualityEquation", "DriveEquation",
    # materials
    "Material", "TabularViscoelasticMaterial", "PronyViscoelasticMaterial",
    # boundary
    "BoundaryConditions", "FixedBoundaryCondition", "DisplacementBoundaryCondition",
    "PeriodicBoundaryCondition", "OldPeriodicBoundaryCondition", "validate_constraints",
    # steps
    "Heading", "Static", "Dynamic", "Step", "Model", "Simulation",
    # utils
    "in_sorted", "load_matlab_microstructure", "assign_intph",
    "periodic_assign_intph", "load_viscoelasticity",
]
