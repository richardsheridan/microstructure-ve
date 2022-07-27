import pathlib
import subprocess
from functools import partial
from os import PathLike
from typing import Optional, Sequence, List, Union, TextIO, Iterable

import numpy as np
from dataclasses import dataclass

ABAQUS_PATH = pathlib.Path("/var/DassaultSystemes/SIMULIA/Commands/abaqus")
BASE_PATH = pathlib.Path(__file__).parent


###################
# Keyword Classes #
###################

# Each keyword class represents a specific ABAQUS keyword.
# They know what the structure of the keyword section is and what data
# are needed to fill it out. They should do minimize computation outside
# of a to_inp method that actually writes directly to the input file.


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

    @classmethod
    def from_intph_img(cls, intph_img, scale):
        nodes_shape = np.array(intph_img.shape) + 1
        return cls(nodes_shape, scale)

    def __post_init__(self):
        self.node_nums = range(1, 1 + np.prod(self.shape))  # 1-indexing for ABAQUS
        self.virtual_node = self.node_nums[-1] + 1

    def to_inp(self, inp_file_obj):
        y_pos, x_pos = self.scale * np.indices(self.shape)
        inp_file_obj.write("*Node\n")
        for node_num, x, y in zip(self.node_nums, x_pos.ravel(), y_pos.ravel()):
            inp_file_obj.write(f"{node_num:d},\t{x:.6e},\t{y:.6e}\n")
        # noinspection PyUnboundLocalVariable
        # quirk: we abuse the loop variables to put another "virtual" node at the corner
        inp_file_obj.write(f"{self.virtual_node:d},\t{x:.6e},\t{y:.6e}\n")


@dataclass
class CPE4RElements:
    nodes: GridNodes

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
        inp_file_obj.write("*Element, type=CPE4R\n")
        for elem_num, tn, kn, rn, trn in zip(
                self.element_nums, top_nodes, key_nodes, right_nodes, topright_nodes
        ):
            inp_file_obj.write(
                f"{elem_num:d},\t{tn:d},\t{kn:d},\t{rn:d},\t{trn:d},\t\n"
            )


# "top" is image rather than matrix convention
sides = {
    "LeftSurface": np.s_[1:, 0], # include top left
    "RightSurface": np.s_[1:, -1], # include top right
    "BotmSurface": np.s_[0, 1:-1],
    "TopSurface": np.s_[-1, 1:-1],
    "BotmLeft": np.s_[0, 0],
    "TopLeft": np.s_[-1, 0],
    "BotmRight": np.s_[0, -1],
    "TopRight": np.s_[-1, -1],
}


@dataclass(eq=False)
class NodeSet:
    name: str
    node_inds: Union[np.ndarray, List[int]]

    @classmethod
    def from_side_name(cls, name, nodes):
        sl = sides[name]
        row_ind, col_ind = np.indices(nodes.shape)
        node_inds = 1 + np.ravel_multi_index(
            (row_ind[sl].ravel(), col_ind[sl].ravel()),
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
class EqualityEquation:
    # A1*node1 + A2*node2 + ... = 0
    nsets: Sequence[Union[NodeSet, int]] # nodes
    factors: Sequence[int] # A's
    dof: int

    def to_inp(self, inp_file_obj):
        # compute the number of terms in the equation on the fly
        num_terms = len(self.nsets)
        # write the section header
        inp_file_obj.write(
            f"""\
*Equation
{num_terms}
"""
        )
        # assemble nodes and A's
        for i in range(num_terms):
            inp_file_obj.write(
                f"""\
{self.nsets[i]}, {self.dof}, {self.factors[i]}.
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
        # must append .astype(int), pure polymer cases will have float type uniq
        uniq = np.unique(intph_img).astype(int)  # sorted!
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


@dataclass
class DisplacementBoundaryCondition:
    drive_node: Union[NodeSet, int]
    first_dof: int
    last_dof: int
    displacement: Optional[float] = None

    def to_inp(self, inp_file_obj):
        disp = self.displacement if self.displacement is not None else ""
        inp_file_obj.write(
            f"""\
*Nset, nset=drive
{self.drive_node}
*Boundary, type=displacement
{self.drive_node}, {self.first_dof}, {self.last_dof}, {disp}
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
class ViscoelasticMaterial(Material):
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
        return 10 ** freq

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

    def to_inp(self, inp_file_obj):
        super().to_inp(inp_file_obj)
        inp_file_obj.write("*Viscoelastic, frequency=TABULAR\n")

        # special normalized bulk modulus used by abaqus
        # if poisson's ratio is frequency-independent, it drops out
        # and youngs=shear=bulk when normalized
        youngs_inf = self.youngs_cplx[0].real
        real = (self.youngs_cplx.imag / youngs_inf).tolist()
        imag = (1 - self.youngs_cplx.real / youngs_inf).tolist()
        freq = self.apply_shift().tolist()

        for wgr, wgi, wkr, wki, f in zip(real, imag, real, imag, freq):
            inp_file_obj.write(f"{wgr:.6e}, {wgi:.6e}, {wkr:.6e}, {wki:.6e}, {f:.6e}\n")


@dataclass
class PeriodicBoundaryConditions:
    nodes: GridNodes
    disp_bnd: DisplacementBoundaryCondition

    def __post_init__(self):
        make_set = partial(NodeSet.from_side_name, nodes=self.nodes)
        self.node_pairs: List[List[NodeSet]] = [
            [make_set("RightSurface"), make_set("LeftSurface")],
            [make_set("TopSurface"), make_set("BotmSurface")],
        ]
        bl = make_set("BotmLeft")
        self.ref_node_pairs: List[List[NodeSet]] = [
            [make_set("BotmRight"), bl],
            [make_set("TopLeft"), bl],
        ]
        self.factors = [1,-1,-1,1]

    def to_inp(self, inp_file_obj):
        # keep a set to avoid defining reference node sets repeatedly
        seen_ref_node_sets = set()
        # build PBC first
        for s in range(len(self.node_pairs)):
            ref_node_set_0, ref_node_set_1 = self.ref_node_pairs[s]
            # write the node set if not written yet for displacement calculation
            if ref_node_set_0 not in seen_ref_node_sets:
                ref_node_set_0.to_inp(inp_file_obj)
                seen_ref_node_sets.add(ref_node_set_0)
            if ref_node_set_1 not in seen_ref_node_sets:
                ref_node_set_1.to_inp(inp_file_obj)
                seen_ref_node_sets.add(ref_node_set_1)
            # unpack reference nodes (corners)
            ref_node_0 = ref_node_set_0.node_inds[0]
            ref_node_1 = ref_node_set_1.node_inds[0]
            # loop through node pairs to build 4-node equation
            node_set_0, node_set_1 = self.node_pairs[s]
            # write the node set just for record
            node_set_0.to_inp(inp_file_obj)
            node_set_1.to_inp(inp_file_obj)
            # build equation section
            for i in range(len(node_set_0.node_inds)):
                # Displacement at any surface node is equal to the opposing surface
                # node in both degrees of freedom
                eq_type = [EqualityEquation, EqualityEquation]
                node_pair = [node_set_0.node_inds[i],node_set_1.node_inds[i],
                             ref_node_0,ref_node_1]
                eq_type[0](node_pair, self.factors, 1).to_inp(inp_file_obj)
                eq_type[1](node_pair, self.factors, 2).to_inp(inp_file_obj)

        # apply displacement
        # self.disp_bnd must have dof between 1 and 2
        # if first dof include 1, kinematically couple it with the ref nodes in
        # dof 1
        if self.disp_bnd.first_dof == 1:
            ref_node_set_0, ref_node_set_1 = self.ref_node_pairs[
                self.disp_bnd.first_dof-1]
            ref_node_0 = ref_node_set_0.node_inds[0]
            ref_node_1 = ref_node_set_1.node_inds[0]
            node_pair = [ref_node_0, ref_node_1, self.disp_bnd.drive_node]
            eq = EqualityEquation(node_pair, [-1,1,1], 1)
            eq.to_inp(inp_file_obj)
        # if last dof include 2, kinematically couple it with the ref nodes in
        # dof 2
        if self.disp_bnd.last_dof == 2:
            ref_node_set_0, ref_node_set_1 = self.ref_node_pairs[
                self.disp_bnd.last_dof-1]
            ref_node_0 = ref_node_set_0.node_inds[0]
            ref_node_1 = ref_node_set_1.node_inds[0]
            node_pair = [ref_node_0, ref_node_1, self.disp_bnd.drive_node]
            eq = EqualityEquation(node_pair, [-1,1,1], 2)
            eq.to_inp(inp_file_obj)
        

@dataclass
class StepParameters:
    """Data for the ABAQUS STEP keyword"""
    disp_bnd_nodes: Iterable[DisplacementBoundaryCondition]
    f_initial: float
    f_final: float
    f_count: int
    bias: int

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*STEP,NAME=STEP-1,PERTURBATION
*STEADY STATE DYNAMICS, DIRECT
{self.f_initial}, {self.f_final}, {self.f_count}, {self.bias}
"""
        )
        for n in self.disp_bnd_nodes:
            n.to_inp(inp_file_obj)
        inp_file_obj.write(
            f"""\
*RESTART,WRITE,frequency=0
*Output, field, variable=PRESELECT
*Output, field
*Element output, directions=YES
ELEDEN, ELEN, ENER, EVOL
*Output, history
*Energy Output
ALLAE, ALLCD, ALLEE, ALLFD, ALLJD, ALLKE, ALLPD, ALLSD, ALLSE, ALLVD, ALLWK, ETOTAL
*END STEP
"""
        )


####################
# Helper functions #
####################

# High level functions representing important transformations or steps.
# Probably the most important part is the name and docstring, to explain
# WHY a certain procedure is being taken/option being input.


def write_abaqus_input(
        *,
        nodes: GridNodes,
        elements: CPE4RElements,
        materials: Iterable[Material],
        bcs: PeriodicBoundaryConditions,
        step_parm: StepParameters,
        heading: Optional[Heading] = None,
        extra_nsets: Iterable[NodeSet] = (),
        path: Union[None, str, PathLike[str]] = None,
        inp_file_obj: Optional[TextIO] = None
):
    if inp_file_obj is None:
        if path is None:
            raise ValueError("Supply either path or inp_file_obj")
        with open(path, mode="w", encoding="ascii") as f:
            return write_abaqus_input(heading=heading, nodes=nodes, elements=elements,
                                      materials=materials, bcs=bcs, step_parm=step_parm,
                                      extra_nsets=extra_nsets, inp_file_obj=f)

    if heading is not None:
        heading.to_inp(inp_file_obj)
    nodes.to_inp(inp_file_obj)
    for nset in extra_nsets:
        nset.to_inp(inp_file_obj)
    elements.to_inp(inp_file_obj)
    for m in materials:
        m.to_inp(inp_file_obj)
    bcs.to_inp(inp_file_obj)
    step_parm.to_inp(inp_file_obj)


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


def periodic_assign_intph(microstructure: np.ndarray,
                          num_layers_list: List[int]) -> np.ndarray:
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
    intph = intph_tiled[dimx:dimx + dimx, dimy:dimy + dimy]
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


def run_job(job_name, cpus):
    """feed .inp file to ABAQUS and wait for the result"""
    subprocess.run(
        [ABAQUS_PATH, "job=" + job_name, "cpus=" + str(cpus), "interactive"],
        check=True,
    )


def read_odb(job_name, displacement):
    """Extract viscoelastic response from abaqus output ODB

    Uses abaqus python api which is stuck in python 2.7 ancient history,
    so we need to farm it out to a subprocess.
    """
    subprocess.run(
        [ABAQUS_PATH, "python", BASE_PATH / "readODB.py", job_name, str(displacement)],
        check=True,
    )
