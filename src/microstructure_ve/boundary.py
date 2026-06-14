"""Boundary conditions, periodic constraints, and over-constraint validation.

Pure data plus the constraint bookkeeping (``dependent_dofs`` / ``prescribed_dofs``)
that ``validate_constraints`` and both backends consume. Emission lives in the ABAQUS
backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Union

import numpy as np

from .core import GridNodes, NodeSet, Sides_2d, _node_array
from .equations import DriveEquation, EqualityEquation, SequentialDifferenceEquation


class BoundaryConditions:
    """Marker base for everything that lives in a model's ``bcs`` list."""


@dataclass
class FixedBoundaryCondition(BoundaryConditions):
    """Pin (zero) the given DOFs of a node or node set.

    ``node`` is a NodeSet or a 1-indexed node number; ``dofs`` is an iterable of
    1-indexed DOFs (1=x, 2=y, 3=z). E.g. pin the origin in-plane: ``dofs=[1, 2]``.
    """

    node: Union[NodeSet, int]
    dofs: Iterable

    def prescribed_dofs(self):
        nodes = _node_array(self.node)
        for dof in self.dofs:
            yield nodes, dof


@dataclass
class DisplacementBoundaryCondition(BoundaryConditions):
    """Prescribe a displacement on DOFs ``first_dof..last_dof`` (1-indexed) of ``nset``.

    Used both as the macro drive (the applied amplitude) and, with ``displacement=0``,
    as a baseline. The DOLFINx backend reads the drive amplitude from the one placed in
    a step's ``subsections``.
    """

    nset: Union[NodeSet, int]
    first_dof: int
    last_dof: int
    displacement: float

    def prescribed_dofs(self):
        nodes = _node_array(self.nset)
        for dof in range(self.first_dof, self.last_dof + 1):
            yield nodes, dof


@dataclass
class PeriodicBoundaryCondition:
    """Periodic boundary conditions tying each face to its opposite.

    On construction it eagerly builds the constraint ``equations`` (``u_dep - u_img =
    u_refHi - u_refLo``) that couple opposite boundaries through the reference corner
    nodes (X0Y0, X1Y0, X0Y1, ...); driving those corners imposes the macro deformation.
    The equation count is grid-size-independent (6 in 2D, 48 in 3D).
    """

    nodes: GridNodes

    def __post_init__(self):
        nsets = self.nodes.nsets
        if self.nodes.dim == 2:
            # 2D boundaries
            self.node_pairs: List[List[NodeSet]] = [
            # Vertices
            [nsets["X1Y1"], nsets["X0Y1"], nsets["X1Y0"], nsets["X0Y0"]], # 2-1 = 4-3

            # Edges
            [nsets["X1"], nsets["X0"], nsets["X1Y0"], nsets["X0Y0"]], # e6-e5 = 4-3
            [nsets["Y1"], nsets["Y0"], nsets["X0Y1"], nsets["X0Y0"]], # e10-e9 = 1-3
            ]
        elif self.nodes.dim == 3:
            # 3D boundaries
            self.node_pairs: List[List[NodeSet]] = [
            # Vertices
            [nsets["X1Y1Z0"], nsets["X0Y1Z0"], nsets["X1Y0Z0"], nsets["X0Y0Z0"]], # 2-1 = 4-3
            [nsets["X1Y1Z1"], nsets["X0Y1Z1"], nsets["X1Y0Z0"], nsets["X0Y0Z0"]], # 6-5 = 4-3
            [nsets["X1Y0Z1"], nsets["X0Y0Z1"], nsets["X1Y0Z0"], nsets["X0Y0Z0"]], # 8-7 = 4-3

            [nsets["X0Y1Z1"], nsets["X0Y0Z1"], nsets["X0Y1Z0"], nsets["X0Y0Z0"]], # 5-7 = 1-3

            # Edges
            [nsets["X1Y0"], nsets["X0Y0"], nsets["X1Y0Z0"], nsets["X0Y0Z0"]], # e2-e1 = 4-3
            [nsets["X1Y1"], nsets["X0Y1"], nsets["X1Y0Z0"], nsets["X0Y0Z0"]], # e3-e4 = 4-3
            [nsets["X1Z0"], nsets["X0Z0"], nsets["X1Y0Z0"], nsets["X0Y0Z0"]], # e6-e5 = 4-3
            [nsets["X1Z1"], nsets["X0Z1"], nsets["X1Y0Z0"], nsets["X0Y0Z0"]], # e7-e8 = 4-3

            [nsets["Y1Z0"], nsets["Y0Z0"], nsets["X0Y1Z0"], nsets["X0Y0Z0"]], # e10-e9 = 1-3
            [nsets["Y1Z1"], nsets["Y0Z1"], nsets["X0Y1Z0"], nsets["X0Y0Z0"]], # e11-e12 = 1-3
            [nsets["X0Y1"], nsets["X0Y0"], nsets["X0Y1Z0"], nsets["X0Y0Z0"]], # e4-e1 = 1-3

            [nsets["X0Z1"], nsets["X0Z0"], nsets["X0Y0Z1"], nsets["X0Y0Z0"]], # e8-e5 = 7-3
            [nsets["Y0Z1"], nsets["Y0Z0"], nsets["X0Y0Z1"], nsets["X0Y0Z0"]], # e12-e9 = 7-3

            # Faces
            [nsets["X1"], nsets["X0"], nsets["X1Y0Z0"], nsets["X0Y0Z0"]], # xFront-xBack = 4-3
            [nsets["Y1"], nsets["Y0"], nsets["X0Y1Z0"], nsets["X0Y0Z0"]], # yTop-yBottom = 1-3
            [nsets["Z1"], nsets["Z0"], nsets["X0Y0Z1"], nsets["X0Y0Z0"]], # zLeft-zRight = 7-3
            ]
        else:
            raise ValueError('GridNodes has illegal number of dimensions', self.nodes.dim)

        # Build the equations eagerly so dependents are enumerable before to_inp
        # (and so SequentialDifferenceEquation's node-count check runs at construction).
        # Count is len(node_pairs) * dim (6 in 2D, 48 in 3D) -- independent of grid size.
        self.equations: List[SequentialDifferenceEquation] = [
            SequentialDifferenceEquation(node_pair, i + 1)
            for node_pair in self.node_pairs
            for i in range(self.nodes.dim)
        ]

    def dependent_dofs(self):
        for eq in self.equations:
            yield from eq.dependent_dofs()


@dataclass
class OldPeriodicBoundaryCondition(DisplacementBoundaryCondition):
    nodes: GridNodes

    def __post_init__(self):
        def make_set(name):
            return NodeSet.from_slice(name, Sides_2d[name], self.nodes)

        ndim = len(self.nodes.shape)
        self.driven_nset = NodeSet.from_slice("X1ALL", np.s_[:, -1], self.nodes)
        self.node_pairs: List[List[NodeSet]] = [
            [NodeSet.from_slice("X0ALL", np.s_[:, 0], self.nodes), self.driven_nset],
            [make_set("Y0"), make_set("Y1")],
            [make_set("X1Y0"), make_set("X1Y1")],
        ]
        # Displacement at any surface node is equal to the opposing surface
        # node in both degrees of freedom unless one of the surfaces is a driver.
        # in that case, add the avg displacement from the drive node
        self.eq_pairs: List[List[EqualityEquation]] = [
            [EqualityEquation(p, x + 1) for x in range(ndim)]
            if (self.driven_nset not in p)
            else [
                DriveEquation(p, x + 1, drive_node=self.nset)
                if x in range(self.first_dof - 1, self.last_dof)
                else EqualityEquation(p, x + 1)
                for x in range(ndim)
            ]
            for p in self.node_pairs
        ]

    def dependent_dofs(self):
        # Aggregate the dependents of every equation; prescribed_dofs (the driven
        # nset) is inherited from DisplacementBoundaryCondition.
        for eq_pair in self.eq_pairs:
            for eq in eq_pair:
                yield from eq.dependent_dofs()


def validate_constraints(bcs):
    """Raise ValueError if a dependent (eliminated) DOF is over-constrained.

    ABAQUS eliminates the first-listed (node, dof) of every *Equation. That dependent
    DOF must not also be (a) the dependent term of another *Equation, nor (b) prescribed
    by a *Boundary -- either is an over-constraint ABAQUS rejects. Double-*Boundary on a
    DOF is tolerated (a model-level baseline plus a step displacement is normal) and is
    not flagged. Catches the error at Model construction instead of at solve time.
    """
    # Each constraint yields (node_inds, scalar dof) groups it eliminates / prescribes.
    def gather(attr):
        groups = []
        for bc in bcs:
            method = getattr(bc, attr, None)
            if method is not None:
                for inds, dof in method():
                    groups.append((bc, np.asarray(inds), dof))
        return groups

    dep = gather("dependent_dofs")
    if not dep:
        return
    pre = gather("prescribed_dofs")
    # Encode each (node, dof) as one int64 key; dof is small so node * mult + dof is exact.
    mult = max(dof for _, _, dof in dep + pre) + 1

    def encode(groups):
        if not groups:
            return np.empty(0, np.int64), [], np.empty(0, dtype=int)
        keys = np.concatenate([inds * mult + dof for _, inds, dof in groups])
        owners = [bc for bc, _, _ in groups]
        owner_of_key = np.repeat(np.arange(len(groups)), [inds.size for _, inds, _ in groups])
        return keys, owners, owner_of_key

    dep_keys, dep_owners, dep_oidx = encode(dep)
    pre_keys, pre_owners, pre_oidx = encode(pre)

    # (a) the same (node, dof) eliminated by more than one *Equation
    uniq, counts = np.unique(dep_keys, return_counts=True)
    dup = uniq[counts > 1]
    if dup.size:
        key = int(dup[0])
        rows = np.where(dep_keys == key)[0]
        names = sorted({type(dep_owners[dep_oidx[i]]).__name__ for i in rows})
        raise ValueError(
            f"over-constrained: node {key // mult} dof {key % mult} is the dependent "
            f"term of more than one *Equation ({', '.join(names)})"
        )

    # (b) the same (node, dof) eliminated by an *Equation and prescribed by a *Boundary
    both = np.intersect1d(dep_keys, pre_keys)
    if both.size:
        key = int(both[0])
        d = type(dep_owners[dep_oidx[np.where(dep_keys == key)[0][0]]]).__name__
        p = type(pre_owners[pre_oidx[np.where(pre_keys == key)[0][0]]]).__name__
        raise ValueError(
            f"over-constrained: node {key // mult} dof {key % mult} is eliminated by an "
            f"*Equation in {d} but also prescribed by a *Boundary in {p}"
        )
