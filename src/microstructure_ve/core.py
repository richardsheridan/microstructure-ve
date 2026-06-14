"""Geometry and mesh primitives: node grids, node sets, elements.

These are pure solver-neutral data: they carry the structured-grid geometry and the
boundary node sets. Serialization to an ABAQUS ``.inp`` lives in the ABAQUS backend
(``microstructure_ve.backends._abaqus``), not on these classes.
"""
from __future__ import annotations

from functools import partial
from itertools import product
from typing import Dict, List, Literal, Union

import numpy as np
from dataclasses import dataclass, field

# NOTE: every "1 +" you see is correcting the array indexing mismatch between
# python and abaqus (python has zero-indexed, abaqus has one-indexed arrays)
# however "+ 1" actually indicates an extra step


Sides_3d = {
    # Abaqus interprets this as [Z, Y, X]
    # Remeber that np arrays are written in [H, W, D] or [Y, X, Z]
    # FACES
    "X0": np.s_[1:-1, 1:-1, 0],   # Left (Face)
    "X1": np.s_[1:-1, 1:-1, -1],  # Right (Face)
    "Y0": np.s_[1:-1, 0, 1:-1],   # Bottom (Face)
    "Y1": np.s_[1:-1, -1, 1:-1],  # Top (Face)
    "Z0": np.s_[0, 1:-1, 1:-1],   # Back (Face)
    "Z1": np.s_[-1, 1:-1, 1:-1],  # Front (Face)

    # EDGES
    # on x axis
    "Y0Z0": np.s_[0, 0, 1:-1],    # Bottom Back (Edge)
    "Y0Z1": np.s_[-1, 0, 1:-1],   # Bottom Front (Edge)
    "Y1Z0": np.s_[0, -1, 1:-1],   # Top Back (Edge)
    "Y1Z1": np.s_[-1, -1, 1:-1],  # Top Front (Edge)

    # on y axis
    "X0Z0": np.s_[0, 1:-1, 0],    # Left Back (Edge)
    "X0Z1": np.s_[-1, 1:-1, 0],   # Left Front (Edge)
    "X1Z0": np.s_[0, 1:-1, -1],   # Right Back (Edge)
    "X1Z1": np.s_[-1, 1:-1, -1],  # Right Front (Edge)

    # on x axis
    "X0Y0": np.s_[1:-1, 0, 0],    # Left Bottom (Edge)
    "X0Y1": np.s_[1:-1, -1, 0],   # Left Top (Edge)
    "X1Y0": np.s_[1:-1, 0, -1],   # Right Bottom (Edge)
    "X1Y1": np.s_[1:-1, -1, -1],  # Right Top (Edge)

    # VERTICES
    "X0Y0Z0": np.s_[0, 0, 0],     # Left Bottom Back (Vertex)
    "X0Y1Z0": np.s_[0, -1, 0],    # Left Top Back (Vertex)
    "X1Y0Z0": np.s_[0, 0, -1],    # Right Bottom Back (Vertex)
    "X1Y1Z0": np.s_[0, -1, -1],   # Right Top Back (Vertex)

    "X0Y0Z1": np.s_[-1, 0, 0],    # Left Bottom Front (Vertex)
    "X0Y1Z1": np.s_[-1, -1, 0],   # Left Top Front (Vertex)
    "X1Y0Z1": np.s_[-1, 0, -1],   # Right Bottom Front (Vertex)
    "X1Y1Z1": np.s_[-1, -1, -1],  # Right Top Front (Vertex)
}

Sides_2d = {

    # 2D
    # EDGES
    # on x axis
    "Y0": np.s_[0, 1:-1],   # Bottom (Edge)
    "Y1": np.s_[-1, 1:-1],  # Top (Edge)

    # on y axis
    "X0": np.s_[1:-1, 0],   # Left (Edge)
    "X1": np.s_[1:-1, -1],  # Right (Edge)

    # VERTICES
    "X0Y0": np.s_[0, 0],    # Left Bottom (Vertex)
    "X0Y1": np.s_[-1, 0],   # Left Top (Vertex)
    "X1Y0": np.s_[0, -1],   # Right Bottom (Vertex)
    "X1Y1": np.s_[-1, -1],  # Right Top (Vertex)
}


@dataclass(eq=False)
class NodeSet:
    name: str
    node_inds: Union[np.ndarray, List[int]]

    @classmethod
    def from_slice(cls, name, slice_, nodes):
        """A named node set selecting the (1-indexed) nodes under a numpy slice.

        >>> import numpy as np
        >>> from microstructure_ve.core import GridNodes, NodeSet, Sides_2d
        >>> nodes = GridNodes(np.array([2, 2]), 1.0)
        >>> NodeSet.from_slice("X0Y0", Sides_2d["X0Y0"], nodes).node_inds
        array([1])
        """
        inds = np.indices(nodes.shape)
        inds_list = []
        for ind in inds:
            inds_list.append(ind[slice_].ravel())

        inds_tuple = tuple(inds_list)
        node_inds = 1 + np.ravel_multi_index(
            inds_tuple,
            dims=nodes.shape,
        )
        return cls(name, node_inds)

    def __str__(self):
        return self.name


def _node_array(token):
    """1-indexed ABAQUS node numbers for a NodeSet or a bare int, as a 1D int ndarray.

    Returns a view of NodeSet.node_inds when present (np.asarray won't copy an
    existing ndarray); a 1-element array for a bare int. Used to enumerate the DOFs
    a constraint eliminates/prescribes without materializing per-node Python lists.

    >>> from microstructure_ve.core import _node_array
    >>> _node_array(5)
    array([5])
    """
    node_inds = getattr(token, "node_inds", None)
    if node_inds is not None:
        return np.asarray(node_inds)
    return np.array([token], dtype=int)


@dataclass
class GridNodes:
    shape: np.ndarray
    scale: float
    nsets: Dict[str, NodeSet] = field(init=False)

    @classmethod
    def from_matl_img(cls, matl_img, scale):
        """Build the node grid for a material/pixel image: one more node than pixels
        per axis, scaled by ``scale``.

        >>> import numpy as np
        >>> from microstructure_ve.core import GridNodes
        >>> nodes = GridNodes.from_matl_img(np.zeros((2, 3)), scale=0.5)
        >>> nodes.shape
        array([3, 4])
        >>> nodes.virtual_node  # one past the 3*4 = 12 real nodes
        13
        """
        nodes_shape = np.array(matl_img.shape) + 1
        return cls(nodes_shape, scale)

    def __post_init__(self):
        self.node_nums = range(1, 1 + np.prod(self.shape))  # 1-indexing for ABAQUS
        self.virtual_node = self.node_nums[-1] + 1
        # create nsets
        self.nsets = {}
        make_set = partial(NodeSet.from_slice, nodes=self)
        if self.dim == 2:
            # Declare nsets using "Sides_2d" slicer dictionary
            for side, sl in Sides_2d.items():
                self.nsets[side] = make_set(side, sl)
        elif self.dim == 3:
            # Declare nsets using "Sides_3d" slicer dictionary
            for side, sl in Sides_3d.items():
                self.nsets[side] = make_set(side, sl)
        else:
            raise ValueError('GridNodes has illegal number of dimensions', self.dim)

    @property
    def dim(self):
        return len(self.shape)


@dataclass
class GridElements:
    nodes: GridNodes
    type: Literal["CPE4R", "CPS4R", "CPE4", "C3D8R", "C3D8"] = "C3D8R"

    def __post_init__(self):
        dim = self.nodes.dim
        # CPE4 / C3D8 (full integration) added for tight parity with a full-integration
        # FE backend; the *R variants are reduced-integration.
        if dim == 2:
            if self.type not in {"CPE4R", "CPS4R", "CPE4"}:
                raise ValueError("Need a 2D element type, got:", self.type)
        elif dim == 3:
            if self.type not in {"C3D8R", "C3D8"}:
                raise ValueError("Need a 3D element type, got:", self.type)
        else:
            raise ValueError('GridNodes has illegal number of dimensions', dim)
        self.element_nums = range(1, 1 + np.prod(self.nodes.shape - 1))


@dataclass
class ElementSet:
    matl_code: int
    elements: np.ndarray

    @classmethod
    def from_matl_img(cls, matl_img):
        """Produce a list of ElementSets corresponding to unique pixel values.

        Materials are ordered by the value in each of the pixels. Element numbers are
        1-indexed in raveled pixel order.

        >>> import numpy as np
        >>> from microstructure_ve.core import ElementSet
        >>> sets = ElementSet.from_matl_img(np.array([[0, 1], [1, 0]]))
        >>> [int(s.matl_code) for s in sets]
        [0, 1]
        >>> sets[0].elements  # the two pixels valued 0
        array([1, 4])
        """
        matl_img = matl_img.ravel()
        uniq = np.unique(matl_img)  # sorted!
        indices = np.arange(1, 1 + matl_img.size)

        return [cls(matl_code, indices[matl_img == matl_code]) for matl_code in uniq]
