import pathlib
import subprocess
from typing import Optional, Sequence, List

import numpy as np
from dataclasses import dataclass


ABAQUS_PATH = pathlib.Path("/var/DassaultSystemes/SIMULIA/Commands/abaqus")
BASE_PATH = pathlib.Path(__file__).parent


def load_viscoelasticity(matrl_name):
    freq, youngs_real, youngs_imag = np.loadtxt(matrl_name, unpack=True)
    youngs = np.empty_like(youngs_real, dtype=complex)
    youngs.real = youngs_real
    youngs.imag = youngs_imag
    sortind = np.argsort(freq)
    return freq[sortind], youngs[sortind]


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

    def to_inp(self, inp_file_obj):
        y_pos, x_pos = self.scale * np.indices(self.shape)
        node_nums = range(1, 1 + x_pos.size)  # 1-indexing for ABAQUS
        inp_file_obj.write("*Node\n")
        # noinspection PyDataclass
        for node_num, x, y in zip(node_nums, x_pos.ravel(), y_pos.ravel()):
            inp_file_obj.write(f"{node_num:d},\t{x:.6e},\t{y:.6e}\n")


@dataclass
class BoundaryNodes:
    offset: int = 10000000
    lr_nset: str = "SET-LR"
    tb_nset: str = "SET-TB"

    def to_inp(self, inp_file_obj):
        # create two dummy nodes for PBC in x and y directions
        inp_file_obj.write(
            f"""\
{self.offset:d}, 1.0, 0.0
{self.offset+1:d}, 0.0, 1.0
*Nset, nset={self.lr_nset:s}
{self.offset:d}
*Nset, nset={self.tb_nset:s}
{self.offset+1:d}
"""
        )


def node_index_helper(row_ind, col_ind, dims):
    return 1 + np.ravel_multi_index((row_ind.ravel(), col_ind.ravel()), dims=dims)


@dataclass
class CPE4RElements:
    node_shape: np.ndarray

    def to_inp(self, inp_file_obj):
        # strategy: generate one array representing all nodes, then make slices of it
        # that represent offsets to the right, top, and topright nodes to iterate
        all_nodes = 1 + np.ravel_multi_index(
            np.indices(self.node_shape), self.node_shape
        )
        # elements are defined counterclockwise
        right_nodes = all_nodes[:-1, 1:].ravel()
        key_nodes = all_nodes[:-1, :-1].ravel()
        top_nodes = all_nodes[1:, :-1].ravel()
        topright_nodes = all_nodes[1:, 1:].ravel()
        element_nums = range(1, 1 + key_nodes.size)
        inp_file_obj.write("*Element, type=CPE4R\n")
        # noinspection PyDataclass
        for elem_num, tn, kn, rn, trn in zip(
            element_nums, top_nodes, key_nodes, right_nodes, topright_nodes
        ):
            inp_file_obj.write(
                f"{elem_num:d},\t{tn:d},\t{kn:d},\t{rn:d},\t{trn:d},\t\n"
            )


@dataclass
class NodeSet:
    name: str
    nodes: np.ndarray

    @classmethod
    def from_image_and_slicedict(cls, intph_img, slicedict):
        nodes_shape = np.array(intph_img.shape) + 1
        row_ind, col_ind = np.indices(nodes_shape)
        # noinspection PyArgumentList
        return [
            cls(name, node_index_helper(row_ind[sl], col_ind[sl], nodes_shape))
            for name, sl in slicedict.items()
        ]

    def __iter__(self):
        for node in self.nodes:
            # noinspection PyArgumentList
            yield type(self)(f"{self.name:s}{node:d}", [node])

    def to_inp(self, inp_file_obj):
        for node in self.nodes:
            inp_file_obj.write(f"*Nset, nset={self.name:s}{node:d}\n{node:d}\n")


@dataclass
class BigNodeSet(NodeSet):
    """Like a NodeSet but also grouping all nodes into an additional unified set"""

    def to_inp(self, inp_file_obj):
        super().to_inp(inp_file_obj)
        inp_file_obj.write(f"*Nset, nset={self.name}\n")
        for node in self.nodes:
            inp_file_obj.write(f"{node:d}\n")


@dataclass
class EqualityEquation:
    nsets: Sequence[NodeSet]
    dof: int
    boundary_name: Optional[str] = None

    def to_inp(self, inp_file_obj):
        bnode = self.boundary_name is not None
        inp_file_obj.write(
            f"""\
*Equation
{2 + bnode:d}
{self.nsets[0].name:s}, {self.dof:d}, 1.
{self.nsets[1].name:s}, {self.dof:d}, -1.
"""
        )
        if bnode:
            inp_file_obj.write(f"{self.boundary_name:s}, {self.dof:d}, 1.\n")


@dataclass
class ElementSet:
    matl_code: int
    elements: list

    @classmethod
    def from_intph_image(cls, intph_img):
        intph_img = intph_img.ravel()
        uniq = np.unique(intph_img)  # sorted!
        indices = np.arange(1, 1 + intph_img.size)

        # noinspection PyArgumentList
        return [cls(matl_code, indices[intph_img == matl_code]) for matl_code in uniq]

    def to_inp(self, inp_file_obj):
        mc = self.matl_code
        inp_file_obj.write(f"*Elset, elset=SET-{mc:d}\n")
        for element in self.elements:
            inp_file_obj.write(f"{element:d}\n")
        inp_file_obj.write(
            f"""\
*Solid Section, elset=SET-{mc:d}, material=MAT-{mc:d}
1.
"""
        )


@dataclass
class Material:
    elset: ElementSet
    density: float  # kg/micron^3
    poisson: float
    youngs: float  # MPa, long term, low freq modulus

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*Material, name=MAT-{self.elset.matl_code:d}
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
    left_broadening: float = 0.0
    right_broadening: float = 0.0

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
        # Assume frequency-independent poisson's ratio
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

        wgstar, wkstar = self.normalize_modulus()
        freq = self.apply_shift()

        for wgr, wgi, wkr, wki, f in zip(
            wgstar.real, wgstar.imag, wkstar.real, wkstar.imag, freq
        ):
            inp_file_obj.write(f"{wgr:.6e}, {wgi:.6e}, {wkr:.6e}, {wki:.6e}, {f:.6e}\n")


@dataclass
class StepParameters:
    """Data for the ABAQUS STEP keyword"""

    bnodes: BoundaryNodes
    f_initial: float = 1e-7  # min frequency
    f_final: float = 1e5  # max frequency
    NoF: int = 30  # number of interval picked
    Bias: int = 1  # bias parameter
    displacement: float = 0.005  # displacement

    def to_inp(self, inp_file_obj):
        inp_file_obj.write(
            f"""\
*STEP,NAME=STEP-1,PERTURBATION
*STEADY STATE DYNAMICS, DIRECT
{self.f_initial}, {self.f_final}, {self.NoF}, {self.Bias}
*BOUNDARY
** strain in x direction
{self.bnodes.lr_nset}, 1, 1, {self.displacement}
*BOUNDARY
{self.bnodes.lr_nset}, 2, 2, 0
*BOUNDARY
{self.bnodes.tb_nset}, 1, 2, 0
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
    :type num_layer_list: List(int)
    """
    from scipy.ndimage import distance_transform_edt

    dists = distance_transform_edt(microstructure)
    intph_img = (dists != 0).view("u1")
    for num_layers in sorted(num_layers_list):
        intph_img += dists > num_layers
    return intph_img


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
