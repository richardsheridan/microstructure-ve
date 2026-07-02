"""Analysis steps and the top-level model / simulation containers (pure data).

``Model`` still runs ``validate_constraints`` at construction so the over-constraint
guarantee holds for every backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .boundary import BoundaryConditions, validate_constraints
from .core import GridElements, GridNodes, NodeSet
from .materials import Material


@dataclass
class Heading:
    text: str = ""


@dataclass
class Static:
    """Data for an static subsection of a Step"""

    long_term: bool = False


@dataclass
class Dynamic:
    """Data for a STEADY STATE DYNAMICS subsection of a Step"""

    f_initial: float
    f_final: float
    f_count: int
    bias: int


@dataclass
class Step:
    """A block representing a change in boundary conditions to induce a response.

    ``subsections`` is an ordered iterable of the step's contents -- an analysis type
    (``Static`` / ``Dynamic``) together with the step-level boundary conditions
    (e.g. the drive ``DisplacementBoundaryCondition``). ``perturbation`` adds the
    ``,PERTURBATION`` flag (used for the harmonic steady-state sweep).
    """

    subsections: Iterable
    perturbation: bool = False


@dataclass
class Model:
    nodes: GridNodes
    elements: GridElements
    materials: Iterable[Material]
    bcs: Iterable[BoundaryConditions] = ()
    nsets: Iterable[NodeSet] = ()

    def __post_init__(self):
        validate_constraints(self.bcs)


@dataclass
class Simulation:
    model: Model
    heading: Optional[Heading] = None
    steps: Iterable[Step] = ()
