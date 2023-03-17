import pathlib
import shutil
import subprocess
from functools import partial, cache
from itertools import product
from os import PathLike
from typing import Optional, Sequence, List, Union, TextIO, Iterable

import numpy as np
from dataclasses import dataclass, field

BASE_PATH = pathlib.Path(__file__).parent


###################
# Keyword Classes #
###################

# Each keyword class represents a specific ABAQUS keyword.
# They know what the structure of the keyword section is and what data
# are needed to fill it out. They should minimize computation outside
# the to_inp method that actually writes directly to the input file.


@dataclass
class Heading:
    text: str = ""

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*Heading
{self.text}
"""
        )


# NOTE: every "1 +" you see is correcting the array indexing mismatch between
# python and abaqus (python has zero-indexed, abaqus has one-indexed arrays)
# however "+ 1" actually indicates an extra step


@dataclass
class GridNodes:
    shape: np.ndarray
    scale: float
    dim: int = field(init = False)
    Nsets: dict = field(init = False)

    @classmethod
    def from_intph_img(cls, intph_img, scale):
        nodes_shape = np.array(intph_img.shape) + 1
        return cls(nodes_shape, scale)

    def __post_init__(self):
        self.node_nums = range(1, 1 + np.prod(self.shape))  # 1-indexing for ABAQUS
        self.virtual_node = self.node_nums[-1] + 1
        self.dim = len(self.shape)
        # create nsets
        self.Nsets = {}
        # make_set = partial(NodeSet.from_side_name, nodes=self)
        make_set = partial(NodeSet.from_slice, nodes=self)
        if self.dim == 2:
            # Declare nsets using "Sides_2d" slicer dictionary
            for side, slice in Sides_2d.items():
                self.Nsets[side] = make_set(side, slice)
        elif self.dim == 3:
            # Declare nsets using "Sides_3d" slicer dictionary
            for side, slice in Sides_3d.items():
                self.Nsets[side] = make_set(side, slice)
        else:
            raise ValueError('GridNodes has illegal number of dimensions', self.dim)


    def to_inp(self, inp_file_obj):
        pos = self.scale * np.indices(self.shape)[::-1]
        inp_file_obj.write("*Node\n")
        for node_num, *p in zip(self.node_nums, *map(np.ravel, pos)):
            inp_file_obj.write(f"{node_num:d}")
            for d in p:
                inp_file_obj.write(f",\t{d:.6e}")
            inp_file_obj.write("\n")
        # quirk: we abuse the loop variables to put another "virtual" node at the corner
        inp_file_obj.write(f"{self.virtual_node:d}")
        # noinspection PyUnboundLocalVariable
        for d in p:
            inp_file_obj.write(f",\t{d:.6e}")
        inp_file_obj.write("\n")

# Function which returns the appropriate Element Type for the given nodes
def Elements(nodes: GridNodes):
    if nodes.dim == 2:
        return RectangularElements(nodes)
    elif nodes.dim == 3:
        return CubicElements(nodes)
    else:
        raise ValueError('GridNodes has illegal number of dimensions', nodes.dim)


@dataclass
class RectangularElements:
    nodes: GridNodes
    type: str = "CPE4R"

    def __post_init__(self):
        self.element_nums = range(1, 1 + np.prod(self.nodes.shape - 1))

    def to_inp(self, inp_file_obj):
        # strategy: generate one array representing all nodes, then make slices of it
        # that represent offsets to the right, top, and topright nodes to iterate
        all_nodes = 1 + np.ravel_multi_index(
            np.indices(self.nodes.shape), self.nodes.shape
        )
        # elements are defined counterclockwise
        right_nodes = all_nodes[:-1, 1:].ravel()
        key_nodes = all_nodes[:-1, :-1].ravel()
        top_nodes = all_nodes[1:, :-1].ravel()
        topright_nodes = all_nodes[1:, 1:].ravel()
        inp_file_obj.write(f"*Element, type={self.type}\n")
        for elem_num, tn, kn, rn, trn in zip(
            self.element_nums, top_nodes, key_nodes, right_nodes, topright_nodes
        ):
            inp_file_obj.write(
                f"{elem_num:d},\t{tn:d},\t{kn:d},\t{rn:d},\t{trn:d},\t\n"
            )

@dataclass
class CubicElements:
    nodes: GridNodes
    # type: str = "CPE4R"  # CPS4R, C3D8R
    type: str = "C3D8R"  # CPS4R, C3D8R

    def __post_init__(self):
        self.element_nums = range(1, 1 + np.prod(self.nodes.shape - 1))

    def to_inp(self, inp_file_obj):
        # strategy: generate one array representing all nodes, then make slices of it
        # that represent offsets to e.g. the right or top to iterate
        all_nodes = 1 + np.ravel_multi_index(
            np.indices(self.nodes.shape), self.nodes.shape
        )
        node_slices = list(product(
            (np.s_[:-1], np.s_[1:]),
            repeat=len(self.nodes.shape)
        ))

        # elements are defined counterclockwise, but product produces zigzag
        # swapping the third and fourth elements works for squares and cubes
        node_slices[2], node_slices[3] = node_slices[3], node_slices[2]
        try:
            node_slices[6], node_slices[7] = node_slices[7], node_slices[6]
        except IndexError:
            pass  # it's 2D

        inp_file_obj.write(f"*Element, type={self.type}\n")
        for elem_num, *ns in zip(
            self.element_nums,
            *(all_nodes[slice].ravel() for slice in node_slices),
        ):
            inp_file_obj.write(f"{elem_num:d}")
            for n in ns:
                inp_file_obj.write(f",\t{n:d}")
            inp_file_obj.write("\n")


Sides_3d = {
    # Abaqus interprets this as [Z, Y, X]
    # Remeber that np arrays are written in [H, W, D] or [Y, X, Z]
    # FACES
    "X0": np.s_[1:-1, 1:-1, 0],  # Left (Face)
    "X1": np.s_[1:-1, 1:-1, -1], # Right (Face)
    "Y0": np.s_[1:-1, 0, 1:-1],  # Bottom (Face)
    "Y1": np.s_[1:-1, -1, 1:-1], # Top (Face)
    "Z0": np.s_[0, 1:-1, 1:-1],  # Back (Face)
    "Z1": np.s_[-1, 1:-1, 1:-1], # Front (Face)

    # EDGES
    # on x axis
    "Y0Z0": np.s_[0, 0, 1:-1],  # Bottom Back (Edge)
    "Y0Z1": np.s_[-1, 0, 1:-1],  # Bottom Front (Edge)
    "Y1Z0": np.s_[0, -1, 1:-1],  # Top Back (Edge)
    "Y1Z1": np.s_[-1, -1, 1:-1],  # Top Front (Edge)

    # on y axis
    "X0Z0": np.s_[0, 1:-1, 0],  # Left Back (Edge)
    "X0Z1": np.s_[-1, 1:-1, 0],  # Left Front (Edge)
    "X1Z0": np.s_[0, 1:-1, -1],  # Right Back (Edge)
    "X1Z1": np.s_[-1, 1:-1, -1],  # Right Front (Edge)

    # on x axis
    "X0Y0": np.s_[1:-1, 0, 0],  # Left Bottom (Edge)
    "X0Y1": np.s_[1:-1, -1, 0],  # Left Top (Edge)
    "X1Y0": np.s_[1:-1, 0, -1],  # Right Bottom (Edge)
    "X1Y1": np.s_[1:-1, -1, -1],  # Right Top (Edge)

    # VERTICES
    "X0Y0Z0": np.s_[0, 0, 0],   # Left Bottom Back (Vertex)
    "X0Y1Z0": np.s_[0, -1, 0],  # Left Top Back (Vertex)
    "X1Y0Z0": np.s_[0, 0, -1],  # Right Bottom Back (Vertex)
    "X1Y1Z0": np.s_[0, -1, -1], # Right Top Back (Vertex)

    "X0Y0Z1": np.s_[-1, 0, 0],   # Left Bottom Front (Vertex)
    "X0Y1Z1": np.s_[-1, -1, 0],  # Left Top Front (Vertex)
    "X1Y0Z1": np.s_[-1, 0, -1],  # Right Bottom Front (Vertex)
    "X1Y1Z1": np.s_[-1, -1, -1], # Right Top Front (Vertex)
}

Sides_2d = {

    # 2D
    # EDGES
    # on x axis
    "Y0": np.s_[0, 1:-1],  # Bottom (Edge)
    "Y1": np.s_[-1, 1:-1],  # Top (Edge)

    # on y axis
    "X0": np.s_[1:-1, 0],  # Left (Edge)
    "X1": np.s_[1:-1, -1],  # Right (Edge)

    # VERTICES
    "X0Y0": np.s_[0, 0],   # Left Bottom (Vertex)
    "X0Y1": np.s_[-1, 0],  # Left Top (Vertex)
    "X1Y0": np.s_[0, -1],  # Right Bottom (Vertex)
    "X1Y1": np.s_[-1, -1], # Right Top (Vertex)
}


@dataclass(eq=False)
class NodeSet:
    name: str
    node_inds: Union[np.ndarray, List[int]]

    @classmethod
    def from_side_name(cls, name, nodes):
        if nodes.dim == 2:
            sides = Sides_2d
        elif nodes.dim == 3:
            sides = Sides_3d
        else:
            raise ValueError('GridNodes has illegal number of dimensions', nodes.ndim)
        sl = sides[name]
        inds = np.indices(nodes.shape)
        inds_list = []
        for ind in inds:
            inds_list.append(ind[sl].ravel())

        inds_tuple = tuple(inds_list)
        node_inds = 1 + np.ravel_multi_index(
            inds_tuple,
            dims=nodes.shape,
        )
        return cls(name, node_inds)

    @classmethod
    def from_slice(cls, name, slice, nodes):
        sl = slice
        inds = np.indices(nodes.shape)
        inds_list = []
        for ind in inds:
            inds_list.append(ind[sl].ravel())

        inds_tuple = tuple(inds_list)
        node_inds = 1 + np.ravel_multi_index(
            inds_tuple,
            dims=nodes.shape,
        )
        return cls(name, node_inds)

    def __str__(self):
        return self.name

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(f"*Nset, nset={self.name}\n")
        for i in self.node_inds:
            inp_file_obj.write(f"{i:d}\n")

@dataclass
class SequentialDifferenceEquation:
    nsets: Sequence[Union[NodeSet, int]]
    dof: int

    def to_inp(self, inp_file_obj):
        for i, node0 in enumerate(self.nsets[0].node_inds):
            inp_file_obj.write(
                            f"""\
*Equation
4
{self.nsets[0].node_inds[i]}, {self.dof}, 1.
{self.nsets[1].node_inds[i]}, {self.dof}, -1.
{self.nsets[2]}, {self.dof}, -1.
{self.nsets[3]}, {self.dof}, 1.
"""
            )

@dataclass
class EqualityEquation:
    nsets: Sequence[Union[NodeSet, int]]
    dof: int

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*Equation
2
{self.nsets[0]}, {self.dof}, 1.
{self.nsets[1]}, {self.dof}, -1.
"""
        )


@dataclass
class DriveEquation(EqualityEquation):
    drive_node: Union[NodeSet, int]

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*Equation
3
{self.nsets[0]}, {self.dof}, 1.
{self.nsets[1]}, {self.dof}, -1.
{self.drive_node}, {self.dof}, 1.
"""
        )


@dataclass
class ElementSet:
    matl_code: int
    elements: np.ndarray

    @classmethod
    def from_intph_image(cls, intph_img):
        """Produce a list of ElementSets corresponding to unique pixel values.

        Materials are ordered by distance from filler
        i.e. [filler, interphase, matrix]
        """
        intph_img = intph_img.ravel()
        uniq = np.unique(intph_img)  # sorted!
        indices = np.arange(1, 1 + intph_img.size)

        return [cls(matl_code, indices[intph_img == matl_code]) for matl_code in uniq]

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(f"*Elset, elset=SET-{self.matl_code:d}\n")
        for element in self.elements:
            inp_file_obj.write(f"{element:d}\n")


#################
# Combo classes #
#################

# These represent the structure of several keywords that need to be
# ordered or depend on each other's information somehow. They create a graph
# of information for a complete conceptual component of the input file.


class BoundaryConditions:
    def to_inp(self, inp_file_obj):
        pass


@dataclass
class FixedBoundaryCondition(BoundaryConditions):
    node: Union[NodeSet, int]
    dofs: Iterable
    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*Boundary
"""
        )
        for dof in self.dofs:
            inp_file_obj.write(
            f"""\
{self.node}, {dof}, {dof}
"""
            )

@dataclass
class DisplacementBoundaryCondition(BoundaryConditions):
    nset: Union[NodeSet, int]
    first_dof: int
    last_dof: int
    displacement: float

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*Boundary, type=displacement
{self.nset}, {self.first_dof}, {self.last_dof}, {self.displacement}
"""
        )


@dataclass
class Material:
    elset: ElementSet
    density: float  # kg/micron^3
    poisson: float
    youngs: float  # MPa, long term, low freq modulus

    def to_inp(self, inp_file_obj):
        self.elset.to_inp(inp_file_obj)
        mc = self.elset.matl_code
        inp_file_obj.write(
            f"""\
*Solid Section, elset=SET-{mc:d}, material=MAT-{mc:d}
1.
*Material, name=MAT-{mc:d}
*Density
{self.density:.6e}
*Elastic
{self.youngs:.6e}, {self.poisson:.6e}
"""
        )


@dataclass
class TabularViscoelasticMaterial(Material):
    freq: np.ndarray  # excitation freq in Hz
    youngs_cplx: np.ndarray  # complex youngs modulus
    shift: float = 0.0  # frequency shift induced relative to nominal properties
    left_broadening: float = 1.0  # 1 is no broadening
    right_broadening: float = 1.0  # 1 is no broadening

    def apply_shift(self):
        """Apply shift and broadening factors to frequency.

        left and right refer to frequencies below and above tand peak"""
        freq = np.log10(self.freq) - self.shift

        # shift relative to tand peak
        i = np.argmax(self.youngs_cplx.imag / self.youngs_cplx.real)
        f = freq[i]

        freq[:i] = self.left_broadening * (freq[:i] - f) + f
        freq[i:] = self.right_broadening * (freq[i:] - f) + f
        return 10**freq

    def normalize_modulus(self):
        """Convert to abaqus's preferred normalized moduli"""
        # Only works with frequency-dependent poisson's ratio
        shear_cplx = self.youngs_cplx / (2 * (1 + self.poisson))
        bulk_cplx = self.youngs_cplx / (3 * (1 - 2 * self.poisson))

        # special normalized shear modulus used by abaqus
        wgstar = np.empty_like(shear_cplx)
        shear_inf = shear_cplx[0].real
        wgstar.real = shear_cplx.imag / shear_inf
        wgstar.imag = 1 - shear_cplx.real / shear_inf

        # special normalized bulk modulus used by abaqus
        wkstar = np.empty_like(shear_cplx)
        bulk_inf = bulk_cplx[0].real
        wkstar.real = bulk_cplx.imag / bulk_inf
        wkstar.imag = 1 - bulk_cplx.real / bulk_inf

        return wgstar, wkstar

    def normalize_constant_bulk_modulus(self):
        # assume bulk modulus of glassy system
        bulk_inf = self.youngs_cplx[-1].real / (3 * (1 - 2 * self.poisson))
        shear_cplx = 3 * bulk_inf * self.youngs_cplx / (9 * bulk_inf - self.youngs_cplx)

        # special normalized shear modulus used by abaqus
        wgstar = np.empty_like(shear_cplx)
        shear_inf = shear_cplx[0].real
        wgstar.real = shear_cplx.imag / shear_inf
        wgstar.imag = 1 - shear_cplx.real / shear_inf

        # special normalized bulk modulus used by abaqus
        # if bulk_cplx = bulk_inf, wgstar is all zeros
        wkstar = np.zeros_like(shear_cplx)

        return wgstar, wkstar

    def normalize_constant_nu_modulus(self):
        # special normalized bulk modulus used by abaqus
        # if poisson's ratio is frequency-independent, then
        # youngs=shear=bulk when normalized
        wgstar = np.empty_like(self.youngs_cplx)
        youngs_inf = self.youngs_cplx[0].real
        wgstar.real = self.youngs_cplx.imag / youngs_inf
        wgstar.imag = 1 - self.youngs_cplx.real / youngs_inf

        return wgstar, wgstar

    def to_inp(self, inp_file_obj):
        super().to_inp(inp_file_obj)
        inp_file_obj.write("*Viscoelastic, frequency=TABULAR\n")

        wgstar, wkstar = self.normalize_constant_nu_modulus()
        freq = self.apply_shift()

        for wgr, wgi, wkr, wki, f in zip(
            wgstar.real, wgstar.imag, wkstar.real, wkstar.imag, freq
        ):
            inp_file_obj.write(f"{wgr:.6e}, {wgi:.6e}, {wkr:.6e}, {wki:.6e}, {f:.6e}\n")

@dataclass
class PeriodicBoundaryCondition:
    nodes: GridNodes

    def __post_init__(self):
        Nsets = self.nodes.Nsets
        if self.nodes.dim == 2:
            # 2D boundaries
            self.node_pairs: List[List[NodeSet]] = [
            # Vertices
            [Nsets["X1Y1"], Nsets["X0Y1"], Nsets["X1Y0"], Nsets["X0Y0"]], # 2-1 = 4-3

            # Edges
            [Nsets["X1"], Nsets["X0"], Nsets["X1Y0"], Nsets["X0Y0"]], # e6-e5 = 4-3
            [Nsets["Y1"], Nsets["Y0"], Nsets["X0Y1"], Nsets["X0Y0"]], # e10-e9 = 1-3
            ]
        elif self.nodes.dim == 3:
            # 3D boundaries
            self.node_pairs: List[List[NodeSet]] = [
            # Vertices
            [Nsets["X1Y1Z0"], Nsets["X0Y1Z0"], Nsets["X1Y0Z0"], Nsets["X0Y0Z0"]], # 2-1 = 4-3
            [Nsets["X1Y1Z1"], Nsets["X0Y1Z1"], Nsets["X1Y0Z0"], Nsets["X0Y0Z0"]], # 6-5 = 4-3
            [Nsets["X1Y0Z1"], Nsets["X0Y0Z1"], Nsets["X1Y0Z0"], Nsets["X0Y0Z0"]], # 8-7 = 4-3

            [Nsets["X0Y1Z1"], Nsets["X0Y0Z1"], Nsets["X0Y1Z0"], Nsets["X0Y0Z0"]], # 5-7 = 1-3

            # Edges
            [Nsets["X1Y0"], Nsets["X0Y0"], Nsets["X1Y0Z0"], Nsets["X0Y0Z0"]], # e2-e1 = 4-3
            [Nsets["X1Y1"], Nsets["X0Y1"], Nsets["X1Y0Z0"], Nsets["X0Y0Z0"]], # e3-e4 = 4-3
            [Nsets["X1Z0"], Nsets["X0Z0"], Nsets["X1Y0Z0"], Nsets["X0Y0Z0"]], # e6-e5 = 4-3
            [Nsets["X1Z1"], Nsets["X0Z1"], Nsets["X1Y0Z0"], Nsets["X0Y0Z0"]], # e7-e8 = 4-3

            [Nsets["Y1Z0"], Nsets["Y0Z0"], Nsets["X0Y1Z0"], Nsets["X0Y0Z0"]], # e10-e9 = 1-3
            [Nsets["Y1Z1"], Nsets["Y0Z1"], Nsets["X0Y1Z0"], Nsets["X0Y0Z0"]], # e11-e12 = 1-3
            [Nsets["X0Y1"], Nsets["X0Y0"], Nsets["X0Y1Z0"], Nsets["X0Y0Z0"]], # e4-e1 = 1-3

            [Nsets["X0Z1"], Nsets["X0Z0"], Nsets["X0Y0Z1"], Nsets["X0Y0Z0"]], # e8-e5 = 7-3
            [Nsets["Y0Z1"], Nsets["Y0Z0"], Nsets["X0Y0Z1"], Nsets["X0Y0Z0"]], # e12-e9 = 7-3

            # Faces
            [Nsets["X1"], Nsets["X0"], Nsets["X1Y0Z0"], Nsets["X0Y0Z0"]], # xFront-xBack = 4-3
            [Nsets["Y1"], Nsets["Y0"], Nsets["X0Y1Z0"], Nsets["X0Y0Z0"]], # yTop-yBottom = 1-3
            [Nsets["Z1"], Nsets["Z0"], Nsets["X0Y0Z1"], Nsets["X0Y0Z0"]], # zLeft-zRight = 7-3
            ]
        else:
            raise ValueError('GridNodes has illegal number of dimensions', self.nodes.dim)
    def to_inp(self, inp_file_obj):
        for node_pair in self.node_pairs:
            eq_type = [SequentialDifferenceEquation, SequentialDifferenceEquation, SequentialDifferenceEquation]
            for i, eqn in enumerate(eq_type):
                dof = i+1 #define for X, Y, (Z)
                # Write Equations
                eqn(node_pair, dof).to_inp(inp_file_obj)

@dataclass
class PronyViscoelasticMaterial(Material):
    shear_modulus_ratios: np.ndarray  # ratio of plateau modulus to instantaneous modulus
    bulk_modulus_ratios: np.ndarray
    relaxation_times: np.ndarray

    def to_inp(self, inp_file_obj):
        super().to_inp(inp_file_obj)
        inp_file_obj.write("*Viscoelastic, frequency=PRONY\n")

        for g, k, t in zip(
            self.shear_modulus_ratios, self.bulk_modulus_ratios, self.relaxation_times
        ):
            inp_file_obj.write(f"{g:.6e}, {k:.6e}, {t:.6e}\n")


@dataclass
class OldPeriodicBoundaryCondition(DisplacementBoundaryCondition):
    nodes: GridNodes

    def __post_init__(self):
        make_set = partial(NodeSet.from_side_name, nodes=self.nodes)
        ndim = len(self.nodes.shape)
        self.driven_nset = make_set("X1")
        self.node_pairs: List[List[NodeSet]] = [
            [make_set("X0"), self.driven_nset],
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

    def to_inp(self, inp_file_obj):
        for node_pair, eq_pair in zip(self.node_pairs, self.eq_pairs):
            node_pair[0].to_inp(inp_file_obj)
            node_pair[1].to_inp(inp_file_obj)
            eq_pair[0].to_inp(inp_file_obj)
            eq_pair[1].to_inp(inp_file_obj)
        super().to_inp(inp_file_obj)


@dataclass
class Static:
    """Data for an ABAQUS STATIC subsection of STEP"""

    long_term: bool = False

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*STATIC{", LONG TERM" if self.long_term else ""}
"""
        )


@dataclass
class Dynamic:
    """Data for an ABAQUS STEADY STATE DYNAMICS subsection of STEP"""

    f_initial: float
    f_final: float
    f_count: int
    bias: int

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*STEADY STATE DYNAMICS, DIRECT
{self.f_initial}, {self.f_final}, {self.f_count}, {self.bias}
"""
        )


@dataclass
class Step:
    subsections: Iterable
    perturbation: bool = False

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*STEP{",PERTURBATION" if self.perturbation else ""}
"""
        )
        for n in self.subsections:
            n.to_inp(inp_file_obj)
        inp_file_obj.write(
            f"""\
*END STEP
"""
        )


@dataclass
class Model:
    nodes: GridNodes
    elements: RectangularElements
    materials: Iterable[Material]
    bcs: Iterable[BoundaryConditions] = ()
    fixed_bnds: Iterable[FixedBoundaryCondition] = ()
    nsets: Iterable[NodeSet] = ()

    def to_inp(self, inp_file_obj):
        self.nodes.to_inp(inp_file_obj)
        for nset in self.nsets:
            nset.to_inp(inp_file_obj)
        self.elements.to_inp(inp_file_obj)
        for m in self.materials:
            m.to_inp(inp_file_obj)
        for bc in self.bcs:
            bc.to_inp(inp_file_obj)
        for fixed_bnd in self.fixed_bnds:
            fixed_bnd.to_inp(inp_file_obj)


@dataclass
class Simulation:
    model: Model
    heading: Optional[Heading] = None
    steps: Iterable[Step] = ()

    def to_inp(self, inp_file_obj: TextIO):
        if self.heading is not None:
            self.heading.to_inp(inp_file_obj)
        self.model.to_inp(inp_file_obj)
        for step in self.steps:
            step.to_inp(inp_file_obj)


####################
# Helper functions #
####################

# High level functions representing important transformations or steps.
# Probably the most important part is the name and docstring, to explain
# WHY a certain procedure is being taken/option being input.


def in_sorted(arr, val):
    """Determine if val is contained in arr, assuming arr is sorted"""
    index = np.searchsorted(arr, val)
    if index < len(arr):
        return val == arr[index]
    else:
        return False


def load_matlab_microstructure(matfile, var_name):
    """Load the microstructure in .mat file into a 2D boolean ndarray.
    @para: matfile --> the file name of the microstructure
           var_name --> the name of the variable in the .mat file
                        that contains the 2D microstructure 0-1 matrix.
    @return: 2D ndarray dtype=bool
    """
    from scipy.io import loadmat

    return loadmat(matfile, matlab_compatible=True)[var_name]


def assign_intph(microstructure: np.ndarray, num_layers_list: List[int]) -> np.ndarray:
    """Generate interphase layers around the particles.

    Microstructure must have at least one zero value.

    :rtype: numpy.ndarray
    :param microstructure: The microstructure array. Particles must be zero,
        matrix must be nonzero.
    :type microstructure: numpy.ndarray

    :param num_layers_list: The list of interphase thickness in pixels. The order of
        the layer values is based on the sorted distances in num_layers_list from
        the particles (near particles -> far from particles)
    :type num_layers_list: List(int)
    """
    from scipy.ndimage import distance_transform_edt

    dists = distance_transform_edt(microstructure)
    intph_img = (dists != 0).view("u1")
    for num_layers in sorted(num_layers_list):
        intph_img += dists > num_layers
    return intph_img


def periodic_assign_intph(
    microstructure: np.ndarray, num_layers_list: List[int]
) -> np.ndarray:
    """Generate interphase layers around the particles with periodic BC.

    Microstructure must have at least one zero value.

    :rtype: numpy.ndarray
    :param microstructure: The microstructure array. Particles must be zero,
        matrix must be nonzero.
    :type microstructure: numpy.ndarray

    :param num_layers_list: The list of interphase thickness in pixels. The order of
        the layer values is based on the sorted distances in num_layers_list from
        the particles (near particles -> far from particles)
    :type num_layers_list: List(int)
    """
    tiled = np.tile(microstructure, (3, 3))
    dimx, dimy = microstructure.shape
    intph_tiled = assign_intph(tiled, num_layers_list)
    # trim tiling
    intph = intph_tiled[dimx : dimx + dimx, dimy : dimy + dimy]
    # free intph's view on intph_tiled's memory
    intph = intph.copy()
    return intph


def load_viscoelasticity(matrl_name):
    """load VE data from a text file according to ABAQUS requirements

    mainly the frequency array needs to be strictly increasing, but also having
    the storage/loss data in complex numbers helps our calculations.
    """
    freq, youngs_real, youngs_imag = np.loadtxt(matrl_name, unpack=True)
    youngs = np.empty_like(youngs_real, dtype=complex)
    youngs.real = youngs_real
    youngs.imag = youngs_imag
    sortind = np.argsort(freq)
    return freq[sortind], youngs[sortind]


@cache
def find_command(command: str) -> Optional[PathLike]:
    x = shutil.which(command)
    if x is None:
        # maybe it's a shell alias?
        if shutil.which("bash") is None:
            return None
        p = subprocess.run(
            ["bash", "-i", "-c", f"alias {command}"],
            capture_output=True,
        )
        if p.returncode:
            return None
        x = p.stdout.split(b"'")[1].decode()
    try:
        return pathlib.Path(x).resolve(strict=True)
    except FileNotFoundError:
        return None


def run_job(job_name, cpus):
    """feed .inp file to ABAQUS and wait for the result"""
    subprocess.run(
        [
            find_command("abaqus"),
            "job=" + job_name,
            "cpus=" + str(cpus),
            "interactive",
        ],
        check=True,
    )


def read_odb(job_name, drive_nset):
    """Extract viscoelastic response from abaqus output ODB

    Uses abaqus python api which is stuck in python 2.7 ancient history,
    so we need to farm it out to a subprocess.
    """
    subprocess.run(
        [
            find_command("abaqus"),
            "python",
            BASE_PATH / "readODB.py",
            job_name,
            drive_nset.name,
        ],
        check=True,
    )
