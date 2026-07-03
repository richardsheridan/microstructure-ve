"""Boundary conditions, the periodic constraint, and over-constraint validation.

Boundary conditions are "has-a" containers mirroring ``Material``: a ``BoundaryCondition``
binds a constraint component (``Fixed`` -- pin to zero -- or ``Prescribed`` -- drive to a
value) to a target node or node set. The whole-grid periodic face coupling is *not* a
per-node boundary condition, so it stands beside them as ``PeriodicBoundaryConstraint``.
All are pure data plus the constraint bookkeeping (``dependent_dofs`` /
``prescribed_dofs``) that ``validate_constraints`` and both backends consume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Union

import numpy as np

from .core import GridNodes, NodeSet, _node_array


@dataclass
class Fixed:
    """Pin (zero) the given DOFs of the target.

    ``dofs`` is an iterable of 1-indexed DOFs (1=x, 2=y, 3=z). E.g. pin a node
    in-plane: ``Fixed(dofs=[1, 2])``.
    """

    dofs: Iterable


@dataclass
class Prescribed:
    """Prescribe displacement ``value`` on the given DOFs of the target.

    ``dofs`` is an iterable of 1-indexed DOFs (1=x, 2=y, 3=z). Used both as the macro
    drive (the applied amplitude, placed in a step's ``subsections``) and, with
    ``value=0``, as a model-level baseline (the ABAQUS initial-state convention).
    """

    dofs: Iterable
    value: float


@dataclass
class BoundaryCondition:
    """A ``constraint`` (``Fixed`` / ``Prescribed``) applied to a set of nodes.

    ``target`` is a NodeSet or a 1-indexed node number; the physics of the constraint
    lives on ``constraint`` (has-a, like ``Material.response``)::

        BoundaryCondition(origin, Fixed(dofs=[1, 2]))
        BoundaryCondition(drive_corner, Prescribed(dofs=[1], value=0.005))
    """

    target: Union[NodeSet, int]
    constraint: Union[Fixed, Prescribed]

    def prescribed_dofs(self):
        nodes = _node_array(self.target)
        for dof in self.constraint.dofs:
            yield nodes, dof


@dataclass
class PeriodicBoundaryConstraint:
    """Whole-grid periodic coupling tying each boundary face to its opposite.

    Not a per-node boundary condition -- it couples every boundary node pair
    (``u_dep - u_img = u_refHi - u_refLo``) through the reference corner nodes
    (X0Y0, X1Y0, X0Y1, ...); driving those corners imposes the macro deformation.
    On construction it builds ``node_pairs`` -- ``[dependent, image, refHi, refLo]``
    per boundary group -- which the ABAQUS backend emits as ``*Equation`` blocks
    (the DOLFINx backend applies its own equivalent MPC pairing).
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

        # The dependent and image faces pair node-by-node; check at construction so a
        # broken pairing table fails here, not at emission/solve time.
        for dep, img, *_ in self.node_pairs:
            n0, n1 = len(dep.node_inds), len(img.node_inds)
            if n0 != n1:
                raise ValueError(
                    "paired node sets must have equal node counts", n0, n1
                )

    def dependent_dofs(self):
        # The first-listed nset of each pair is the dependent (eliminated) term of the
        # emitted *Equation, one group per dof (pair-major, dof-minor).
        for pair in self.node_pairs:
            for i in range(self.nodes.dim):
                yield pair[0].node_inds, i + 1


def validate_constraints(bcs):
    """Raise ValueError if a dependent (eliminated) DOF is over-constrained.

    """
    # ABAQUS eliminates the first-listed (node, dof) of every *Equation. That dependent
    # DOF must not also be (a) the dependent term of another *Equation, nor (b) prescribed
    # by a *Boundary -- either is an over-constraint ABAQUS rejects. Double-*Boundary on a
    # DOF is tolerated (a model-level baseline plus a step displacement is normal) and is
    # not flagged. Catches the error at Model construction instead of at solve time.

    # Each constraint yields (node_inds, scalar dof) groups it eliminates / prescribes.
    def gather(attr):
        groups = []
        for bc in bcs:
            method = getattr(bc, attr, None)
            if method is not None:
                for inds, dof in method():
                    groups.append((bc, np.asarray(inds), dof))
        return groups

    def describe(bc):
        constraint = getattr(bc, "constraint", None)
        if constraint is not None:
            return f"{type(bc).__name__}({type(constraint).__name__})"
        return type(bc).__name__

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
        names = sorted({describe(dep_owners[dep_oidx[i]]) for i in rows})
        raise ValueError(
            f"over-constrained: node {key // mult} dof {key % mult} is the dependent "
            f"term of more than one *Equation ({', '.join(names)})"
        )

    # (b) the same (node, dof) eliminated by an *Equation and prescribed by a *Boundary
    both = np.intersect1d(dep_keys, pre_keys)
    if both.size:
        key = int(both[0])
        d = describe(dep_owners[dep_oidx[np.where(dep_keys == key)[0][0]]])
        p = describe(pre_owners[pre_oidx[np.where(pre_keys == key)[0][0]]])
        raise ValueError(
            f"over-constrained: node {key // mult} dof {key % mult} is eliminated by an "
            f"*Equation in {d} but also prescribed by a *Boundary in {p}"
        )
