import pathlib
import subprocess
from os import PathLike
from typing import Optional, Sequence, List, Union, TextIO

import numpy as np
from dataclasses import dataclass

ABAQUS_PATH = pathlib.Path("/var/DassaultSystemes/SIMULIA/Commands/abaqus")
BASE_PATH = pathlib.Path(__file__).parent

# discard corners
# "top" is image rather than matrix convention
sides = {
    "LeftSurface": np.s_[1:-1, 0],
    "RightSurface": np.s_[1:-1, -1],
    "BotmSurface": np.s_[0, 1:-1],
    "TopSurface": np.s_[-1, 1:-1],
}

corners = {
    "BotmLeft": np.s_[0, 0],
    "TopLeft": np.s_[-1, 0],
    "BotmRight": np.s_[0, -1],
    "TopRight": np.s_[-1, -1],
}


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
        inp_file_obj.write("*Heading\n")
        inp_file_obj.write(self.text)


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

    def to_inp(self, inp_file_obj):
        y_pos, x_pos = self.scale * np.indices(self.shape)
        node_nums = range(1, 1 + x_pos.size)  # 1-indexing for ABAQUS
        inp_file_obj.write("*Node\n")
        for node_num, x, y in zip(node_nums, x_pos.ravel(), y_pos.ravel()):
            inp_file_obj.write(f"{node_num:d},\t{x:.6e},\t{y:.6e}\n")


@dataclass
class CPE4RElements:
    nodes: GridNodes

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
        element_nums = range(1, 1 + key_nodes.size)
        inp_file_obj.write("*Element, type=CPE4R\n")
        for elem_num, tn, kn, rn, trn in zip(
                element_nums, top_nodes, key_nodes, right_nodes, topright_nodes
        ):
            inp_file_obj.write(
                f"{elem_num:d},\t{tn:d},\t{kn:d},\t{rn:d},\t{trn:d},\t\n"
            )


@dataclass
class NodeSet:
    name: str
    node_inds: Union[np.ndarray, List[int]]

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(f"*Nset, nset={self.name}\n")
        for i in self.node_inds:
            inp_file_obj.write(f"{i:d}\n")


@dataclass
class EqualityEquation:
    nsets: Sequence[NodeSet]
    dof: int

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*Equation
2
{self.nsets[0].name:s}, {self.dof:d}, 1.
{self.nsets[1].name:s}, {self.dof:d}, -1.
"""
        )


# @dataclass
# class BoundaryEquation:
#     nsets: Sequence[NodeSet]
#     dof: int
#     boundary_name: str
#
#     def to_inp(self, inp_file_obj):
#         inp_file_obj.write(
#             f"""\
# *Equation
# 3
# {self.nsets[0].name:s}, {self.dof:d}, 1.
# {self.nsets[1].name:s}, {self.dof:d}, -1.
# {self.boundary_name:s}, {self.dof:d}, 1.
# """
#         )


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

    def to_inp(self, inp_file_obj):
        row_ind, col_ind = np.indices(self.nodes.shape)

        def node_inds(sl):
            """Efficiently convert numpy slices of the index arrays into abaqus inds"""
            return 1 + np.ravel_multi_index(
                (row_ind[sl].ravel(), col_ind[sl].ravel()),
                dims=self.nodes.shape,
            )

        # Displacement at any surface node is equal to the opposing surface node
        inds_l = node_inds(sides["LeftSurface"])
        inds_r = node_inds(sides["RightSurface"])
        for ind_l, ind_r in zip(inds_l, inds_r):
            node_l = NodeSet(f"LeftSurface{ind_l:d}", [ind_l])
            node_r = NodeSet(f"RightSurface{ind_r:d}", [ind_r])
            eq1 = EqualityEquation([node_l, node_r], 1)
            eq2 = EqualityEquation([node_l, node_r], 2)
            node_l.to_inp(inp_file_obj)
            node_r.to_inp(inp_file_obj)
            eq1.to_inp(inp_file_obj)
            eq2.to_inp(inp_file_obj)

        inds_b = node_inds(sides["BotmSurface"])
        inds_t = node_inds(sides["TopSurface"])
        for ind_b, ind_t in zip(inds_b, inds_t):
            node_b = NodeSet(f"BotmSurface{ind_b:d}", [ind_b])
            node_t = NodeSet(f"TopSurface{ind_t:d}", [ind_t])
            eq1 = EqualityEquation([node_b, node_t], 1)
            eq2 = EqualityEquation([node_b, node_t], 2)
            node_b.to_inp(inp_file_obj)
            node_t.to_inp(inp_file_obj)
            eq1.to_inp(inp_file_obj)
            eq2.to_inp(inp_file_obj)

        # All corner nodes displacement should be identical as they
        # represent the same conceptual point in periodic space (north pole?).
        ind_bl = node_inds(corners["BotmLeft"])
        ind_tl = node_inds(corners["TopLeft"])
        ind_br = node_inds(corners["BotmRight"])
        ind_tr = node_inds(corners["TopRight"])
        node_bl = NodeSet("BotmLeft", ind_bl)
        node_tl = NodeSet("TopLeft", ind_tl)
        node_br = NodeSet("BotmRight", ind_br)
        node_tr = NodeSet("TopRight", ind_tr)
        node_bl.to_inp(inp_file_obj)
        node_tl.to_inp(inp_file_obj)
        node_br.to_inp(inp_file_obj)
        node_tr.to_inp(inp_file_obj)

        # We can only equate pairs of nodes in this sense so chain the eqns
        for node_a, node_b in zip([node_bl, node_tl, node_br],
                                  [node_tl, node_br, node_tr]):
            EqualityEquation([node_b, node_a], 1).to_inp(inp_file_obj)
            EqualityEquation([node_b, node_a], 2).to_inp(inp_file_obj)


@dataclass
class StepParameters:
    """Data for the ABAQUS STEP keyword"""

    nodes: GridNodes
    f_initial: float = 1e-7  # min frequency
    f_final: float = 1e5  # max frequency
    f_count: int = 30  # number of interval picked
    bias: int = 1  # bias parameter
    displacement: float = 0.005

    def to_inp(self, inp_file_obj):
        # select surface to drive
        side = "RightSurface"
        # and which way to push
        dof = "1"
        sl = sides[side]
        row_ind, col_ind = np.indices(self.nodes.shape)
        inds = 1 + np.ravel_multi_index(
            (row_ind[sl].ravel(), col_ind[sl].ravel()),
            dims=self.nodes.shape,
        )
        nset = NodeSet(side, inds)
        nset.to_inp(inp_file_obj)
        inp_file_obj.write(
            f"""\
*STEP,NAME=STEP-1,PERTURBATION
*STEADY STATE DYNAMICS, DIRECT
{self.f_initial}, {self.f_final}, {self.f_count}, {self.bias}
*BOUNDARY, TYPE=DISPLACEMENT
** strain in x direction
{side}, {dof}, {dof}, {self.displacement}
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
        heading: Heading,
        nodes: GridNodes,
        elements: CPE4RElements,
        materials: List[Material],
        bcs: PeriodicBoundaryConditions,
        step_parm: StepParameters,
        *,
        path: Union[None, str, PathLike[str]] = None,
        inp_file_obj: Optional[TextIO] = None
):
    if inp_file_obj is None:
        with open(path, mode="w", encoding="ascii") as f:
            return write_abaqus_input(heading, nodes, elements, materials, bcs,
                                      step_parm, inp_file_obj=f)

    heading.to_inp(inp_file_obj)
    nodes.to_inp(inp_file_obj)
    elements.to_inp(inp_file_obj)
    for m in materials:
        m.to_inp(inp_file_obj)
    bcs.to_inp(inp_file_obj)
    step_parm.to_inp(inp_file_obj)


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
