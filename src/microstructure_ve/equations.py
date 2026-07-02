"""Linear multi-point constraint equations.

Pure data: each class enumerates the dependent (eliminated) DOFs it introduces so the
over-constraint validator and both backends can reason about them. 
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

from .core import NodeSet, _node_array


@dataclass
class SequentialDifferenceEquation:
    nsets: Sequence[Union[NodeSet, int]]
    dof: int

    def __post_init__(self):
        # nsets[0] and nsets[1] pair node-by-node, so they must match
        n0 = len(self.nsets[0].node_inds)
        n1 = len(self.nsets[1].node_inds)
        if n0 != n1:
            raise ValueError(
                "paired node sets must have equal node counts", n0, n1
            )

    def dependent_dofs(self):
        # The first-listed node of each emitted equation is the dependent term.
        yield self.nsets[0].node_inds, self.dof


@dataclass
class EqualityEquation:
    nsets: Sequence[Union[NodeSet, int]]
    dof: int

    def dependent_dofs(self):
        # nsets[0] is the first-listed (dependent) term; DriveEquation keeps it first.
        yield _node_array(self.nsets[0]), self.dof


@dataclass
class DriveEquation(EqualityEquation):
    drive_node: Union[NodeSet, int]
