"""Analysis steps and the top-level model / simulation containers (pure data).

``Model`` runs ``validate_constraints`` at construction so the over-constraint guarantee holds for
every backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Union

from .boundary import BoundaryCondition, PeriodicBoundaryConstraint, validate_constraints
from .core import GridElements, GridNodes, NodeSet
from .materials import Material


@dataclass
class Heading:
    """The free text of the deck's ``*Heading``."""

    text: str = ""


@dataclass
class Static:
    """A static analysis subsection of a Step.

    Parameters
    ----------
    long_term : bool
        Request the long-term (fully relaxed) static response.
    """

    long_term: bool = False


@dataclass
class Dynamic:
    """A STEADY STATE DYNAMICS (harmonic sweep) subsection of a Step.

    Parameters
    ----------
    f_initial, f_final : float
        The frequency range in Hz.
    f_count : int
        The number of frequency points in the range.
    bias : int
        The ABAQUS bias parameter for the spacing of those points.
    """

    f_initial: float
    f_final: float
    f_count: int
    bias: int


@dataclass
class Step:
    """A block representing a change in boundary conditions to induce a response.

    Parameters
    ----------
    subsections : iterable
        The step's contents in order: an analysis type (``Static`` / ``Dynamic``) together with the
        step-level boundary conditions (e.g. the ``Prescribed`` drive ``BoundaryCondition``).
    perturbation : bool
        Add the ``,PERTURBATION`` flag (used for the harmonic steady-state sweep).
    nlgeom : bool
        Add ``nlgeom=YES`` (required for finite-strain / hyperelastic analyses).
    """

    subsections: Iterable
    perturbation: bool = False
    nlgeom: bool = False


@dataclass
class Model:
    """The mesh, materials and model-level constraints of a simulation.

    Parameters
    ----------
    nodes : GridNodes
    elements : GridElements
    materials : iterable of Material
    bcs : iterable of BoundaryCondition or PeriodicBoundaryConstraint
        Constraints that hold in every step.
    nsets : iterable of NodeSet
        Extra node sets beyond those of ``nodes``.

    Raises
    ------
    ValueError
        If ``bcs`` over-constrain a DOF (``validate_constraints``).
    """

    nodes: GridNodes
    elements: GridElements
    materials: Iterable[Material]
    bcs: Iterable[Union[BoundaryCondition, PeriodicBoundaryConstraint]] = ()
    nsets: Iterable[NodeSet] = ()

    def __post_init__(self):
        validate_constraints(self.bcs)


@dataclass
class Simulation:
    """A model, an optional heading and the steps run on it in order."""

    model: Model
    heading: Optional[Heading] = None
    steps: Iterable[Step] = ()
