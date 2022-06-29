# wrapper around microstructure_ve
import numpy as np
import os

from microstructure_ve.microstructure_ve_periodic import (
    Heading,
    GridNodes,
    BoundaryNodes,
    CPE4RElements,
    NodeSet,
    BigNodeSet,
    EqualityEquation,
    ElementSet,
    ViscoelasticMaterial,
    Material,
    StepParameters,
    assign_intph,
    periodic_assign_intph,
    load_matlab_microstructure,
    load_viscoelasticity
)

class msve_wrapper(object):
    def __init__(self):
        return

    def load_param(self, **kwargs):
        '''
        Load parameters to generate the .inp file.

        kwargs can include:

        -------matrix properties-------
        :param mtx_density: matrix density in kg/micron**3
        :type mtx_density: float

        :param mtx_youngs: matrix instantaneous Young's modulus in MPa
        :type mtx_youngs: float

        :param master_curve: filename of the master curve for the matrix, three column format: freq, E', E''
        :type master_curve: str

        :param mtx_poisson: matrix instantaneous poisson ratio
        :type mtx_poisson: float

        -------filler properties-------
        :param fil_density: filler density in kg/micron**3
        :type fil_density: float

        :param fil_youngs: filler Young's modulus in MPa
        :type fil_youngs: float

        :param fil_poisson: filler poisson ratio
        :type fil_poisson: float

        -------interphase properties-------
        :param layers: a list for number of interphase layers
        :type layers: List(int)

        :param periodic_intph: assign periodic interphase on True
        :type periodic_intph: boolean

        :param intph_density: interphase density in kg/micron**3
        :type intph_density: float

        :param intph_youngs: interphase instantaneous Young's modulus in MPa
        :type intph_youngs: float

        :param intph_poisson: interphase instantaneous poisson ratio
        :type intph_poisson: float

        :param intph_shift: shifting factor for interphase
        :type intph_shift: float

        :param intph_l_brd: left broadening factor for interphase
        :type intph_l_brd: float

        :param intph_r_brd: right broadening factor for interphase
        :type intph_r_brd: float

        -------boundary condition-------
        :param displacement: step parameter displacement
        :type displacement: float

        -------global properties-------
        :param scale: scale in nm per pixel
        :type scale: float

        :param fmin: lower limit of frequency range for a steady-state dynamics analysis in Abaqus
        :type fmin: float

        :param fmax: upper limit of frequency range for a steady-state dynamics analysis in Abaqus
        :type fmax: float

        :param num_freq: number of frequency interval
        :type num_freq: int

        -------global properties-------
        :param ms_filename: filename of the microstructure data, .mat or .npy
        :type ms_filename: str

        :param ms_mat_var: variable name of the microstructure matrix in .mat
        :type ms_mat_var: str

        :param reverse: set to True if matrix is 0, filler is 1 in the microstructure
        :type reverse: bool
        '''
        self.params = kwargs
        # load parameters
        if 'mtx_youngs' not in kwargs:
            self.mtx_youngs = 0
        else:
            self.mtx_youngs = kwargs['mtx_youngs']
        self.mtx_density = kwargs['mtx_density']
        self.mtx_poisson = kwargs['mtx_poisson']
        if 'master_curve' not in kwargs:
            raise Exception("master_curve parameter undefined in the input!")
        self.master_curve = kwargs['master_curve']
        if 'mtx_shift' not in kwargs:
            self.mtx_shift = 0
        else:
            self.mtx_shift = kwargs['mtx_shift']
        self.fil_density = kwargs['fil_density']
        self.fil_youngs = kwargs['fil_youngs']
        self.fil_poisson = kwargs['fil_poisson']
        self.layers = kwargs['layers']
        if 'periodic_intph' not in kwargs:
            periodic_intph = kwargs['periodic_intph']
        else:
            periodic_intph = False
        self.intph_density = kwargs['intph_density']
        self.intph_youngs = kwargs['intph_youngs']
        self.intph_poisson = kwargs['intph_poisson']
        self.intph_shift = kwargs['intph_shift']
        self.intph_l_brd = kwargs['intph_l_brd']
        self.intph_r_brd = kwargs['intph_r_brd']
        self.displacement = kwargs['displacement']
        self.scale = kwargs['scale']
        self.ms_filename = kwargs['ms_filename']
        self.fmin = kwargs['fmin']
        self.fmax = kwargs['fmax']
        if 'num_freq' not in kwargs:
            self.num_freq = 30
        else:
            self.num_freq = kwargs['num_freq']
        # load microstructure
        if 'ms_mat_var' not in kwargs:
            self.ms_img = self.load_microstructure(self.ms_filename)
        else:
            self.ms_img = self.load_microstructure(self.ms_filename, kwargs['ms_mat_var'])
        # swap matrix and filler if reverse set to be True
        if 'reverse' in kwargs and kwargs['reverse']:
            self.ms_img = self.ms_img.max() - self.ms_img
        # load master curve
        # excitation freq in Hz and complex modulus
        self.mtx_freq, self.mtx_youngs_cplx = load_viscoelasticity(self.master_curve)
        # assign interphase if needed
        if self.layers[0] > 0:
            if periodic_intph:
                self.intph_img = periodic_assign_intph(self.ms_img, self.layers)
            else:
                self.intph_img = assign_intph(self.ms_img, self.layers)
        else:
            self.intph_img = self.ms_img
        # build inp
        self.sections = self.build_inp()

    def load_microstructure(self, ms_filename, ms_varname = ''):
        # check extension
        ext = os.path.splitext(ms_filename)[1]
        # .npy
        if ext == ".npy":
            return np.load(ms_filename)
        # .mat
        if ext == ".mat":
            return load_matlab_microstructure(ms_filename,ms_varname).astype('uint8')
        raise Exception(f"File extension {ext} is not a valid microstructure file format.")
    
    def build_inp(self):
        sides_lr = {
            "LeftSurface": np.s_[:, 0],
            "RightSurface": np.s_[:, -1],
        }

        sides_tb = {
            # discard results redundant to an entry in left or right
            # also note "top" is image rather than matrix convention
            "BotmSurface": np.s_[0, 1:-1],
            "TopSurface": np.s_[-1, 1:-1],
        }

        corners = {
            "BotmLeft": np.s_[0, 0],
            "TopLeft": np.s_[-1, 0],
            "BotmRight": np.s_[0, -1],
            "TopRight": np.s_[-1, -1],
        }

        sections = []

        heading = Heading()
        nodes = GridNodes(1 + np.array(self.intph_img.shape), self.scale)
        bnodes = BoundaryNodes()
        elements = CPE4RElements(nodes.shape)
        sections.extend((heading, nodes, bnodes, elements))

        nsets_lr = NodeSet.from_image_and_slicedict(self.intph_img, sides_lr)
        nsets_c = NodeSet.from_image_and_slicedict(self.intph_img, corners)
        nsets_tb = BigNodeSet.from_image_and_slicedict(self.intph_img, sides_tb)
        sections.extend((*nsets_lr, *nsets_c, *nsets_tb))

        eqs = [EqualityEquation(nsets, 1, bnodes.lr_nset) for nsets in zip(*nsets_lr)]
        eqs += [EqualityEquation(nsets, 2, bnodes.lr_nset) for nsets in zip(*nsets_lr)][1:-1]
        eqs += [EqualityEquation(nsets, 1, bnodes.tb_nset) for nsets in zip(*nsets_tb)]
        eqs += [EqualityEquation([f, c], 2) for f, (c,) in zip(nsets_tb, nsets_c)]
        sections.extend(eqs)

        if self.layers[0] > 0:
            filler_elset, intph_elset, mat_elset = ElementSet.from_intph_image(self.intph_img)
            sections.extend([filler_elset, intph_elset, mat_elset])

            # filler
            filler_material = Material(filler_elset,
                density=self.fil_density,
                youngs=self.fil_youngs,
                poisson=self.fil_poisson)
            # interphase
            intph_material = ViscoelasticMaterial(
                intph_elset,
                density=self.intph_density,
                poisson=self.intph_poisson,
                youngs=self.intph_youngs,
                freq=self.mtx_freq,
                youngs_cplx=self.mtx_youngs_cplx,
                shift=self.intph_shift,
                left_broadening=self.intph_l_brd,
                right_broadening=self.intph_r_brd
            )
            # matrix
            mat_material = ViscoelasticMaterial(mat_elset,
                density=self.mtx_density,
                poisson=self.mtx_poisson,
                shift=self.mtx_shift,
                left_broadening=1,
                right_broadening=1,
                youngs=self.mtx_youngs,
                freq=self.mtx_freq,
                youngs_cplx=self.mtx_youngs_cplx)
            
            sections.extend([filler_material, intph_material, mat_material])
        else:
            filler_elset, mat_elset = ElementSet.from_intph_image(self.intph_img)
            sections.extend([filler_elset, mat_elset])

            # filler
            filler_material = Material(filler_elset,
                density=self.fil_density,
                youngs=self.fil_youngs,
                poisson=self.fil_poisson)
            # matrix
            mat_material = ViscoelasticMaterial(mat_elset,
                density=self.mtx_density,
                poisson=self.mtx_poisson,
                shift=self.mtx_shift,
                left_broadening=1,
                right_broadening=1,
                youngs=self.mtx_youngs,
                freq=self.mtx_freq,
                youngs_cplx=self.mtx_youngs_cplx)
            
            sections.extend([filler_material, mat_material])

        step_parm = StepParameters(bnodes=bnodes,
            displacement=self.displacement,
            f_initial=self.fmin,
            f_final=self.fmax,
            NoF=self.num_freq)
        sections.append(step_parm)
        return sections

    def to_inp(self, inp_filename):
        with open(inp_filename, "w") as inp_file_obj:
            for section in self.sections:
                section.to_inp(inp_file_obj)