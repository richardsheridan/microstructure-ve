# wrapper around microstructure_ve
import numpy as np
import os

from microstructure_ve.microstructure_ve import (
    Heading,
    GridNodes,
    PeriodicBoundaryConditions,
    CPE4RElements,
    ElementSet,
    ViscoelasticMaterial,
    Material,
    StepParameters,
    assign_intph,
    periodic_assign_intph,
    load_matlab_microstructure,
    load_viscoelasticity,
    write_abaqus_input,
    DisplacementBoundaryCondition,
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

        :param mtx_youngs: matrix long-term Young's modulus in MPa, if it is not
            passed in, the storage modulus of at the lowest freq in the master
            curve will be used as the long-term Young's modulus.
        :type mtx_youngs: float

        :param master_curve: filename of the master curve for the matrix, three
            column format: freq, E', E''. Assumed to be sorted by freq.
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
        :param layers: a list for number of interphase layers, if an int is 
            passed in, it will be automatically wrapped into a list
        :type layers: List(int) or int

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

        :param disp_BC_dof_first: first degree of freedom to apply displacement
        :type disp_BC_dof_first: int

        :param disp_BC_dof_last: last degree of freedom to apply displacement
        :type disp_BC_dof_last: int
    

        -------step parameters-------
        :param fmin: lower limit of frequency range for a steady-state dynamics analysis in Abaqus
        :type fmin: float

        :param fmax: upper limit of frequency range for a steady-state dynamics analysis in Abaqus
        :type fmax: float

        :param num_freq: number of frequency interval
        :type num_freq: int

        -------microstructure properties-------
        :param scale: scale in nm per pixel
        :type scale: float

        :param ms_filename: filename of the microstructure data, .mat or .npy
        :type ms_filename: str

        :param ms_mat_var: variable name of the microstructure matrix in .mat
        :type ms_mat_var: str

        :param reverse: set to True if matrix is 0, filler is 1 in the microstructure
        :type reverse: bool
        '''
        self.params = kwargs
        ## matrix properties
        # load master curve
        if 'master_curve' not in kwargs:
            raise Exception("master_curve parameter undefined in the input!")
        self.master_curve = kwargs['master_curve']
        # excitation freq in Hz and complex modulus
        self.mtx_freq, self.mtx_youngs_cplx = load_viscoelasticity(self.master_curve)
        # load parameters
        if 'mtx_youngs' not in kwargs:
            self.mtx_youngs = self.mtx_youngs_cplx[0].real
        else:
            self.mtx_youngs = kwargs['mtx_youngs']
        self.mtx_density = kwargs['mtx_density']
        self.mtx_poisson = kwargs['mtx_poisson']
        if 'mtx_shift' not in kwargs:
            self.mtx_shift = 0
        else:
            self.mtx_shift = kwargs['mtx_shift']
        ## filler properties
        self.fil_density = kwargs['fil_density']
        self.fil_youngs = kwargs['fil_youngs']
        self.fil_poisson = kwargs['fil_poisson']
        ## interphase properties
        self.layers = kwargs['layers']
        # if layers is passed in as an int, put it in a list
        if type(self.layers) == int:
            self.layers = [self.layers]
        if 'periodic_intph' in kwargs:
            periodic_intph = kwargs['periodic_intph']
        else:
            periodic_intph = False
        self.intph_density = kwargs['intph_density']
        if 'intph_youngs' not in kwargs:
            self.intph_youngs = self.mtx_youngs_cplx[0].real
        else:
            self.intph_youngs = kwargs['intph_youngs']
        self.intph_poisson = kwargs['intph_poisson']
        self.intph_shift = kwargs['intph_shift']
        self.intph_l_brd = kwargs['intph_l_brd']
        self.intph_r_brd = kwargs['intph_r_brd']
        ## boundary conditions
        self.displacement = kwargs['displacement']
        self.disp_BC_dof_first = kwargs['disp_BC_dof_first']
        self.disp_BC_dof_last = kwargs['disp_BC_dof_last']
        ## microstructure properties
        self.scale = kwargs['scale']
        self.ms_filename = kwargs['ms_filename']
        # load microstructure
        if 'ms_mat_var' not in kwargs:
            self.ms_img = self.load_microstructure(self.ms_filename)
        else:
            self.ms_img = self.load_microstructure(self.ms_filename, kwargs['ms_mat_var'])
        # swap matrix and filler if reverse set to be True
        if 'reverse' in kwargs and kwargs['reverse']:
            self.ms_img = self.ms_img.max() - self.ms_img
        ## step parameters
        self.fmin = kwargs['fmin']
        self.fmax = kwargs['fmax']
        if 'num_freq' not in kwargs:
            self.num_freq = 30
        else:
            self.num_freq = kwargs['num_freq']
        # a flag for interphased or not
        self.has_interphase = True if self.layers[0] > 0 else False
        # assign interphase if needed
        if self.has_interphase:
            self.intph_img = periodic_assign_intph(self.ms_img, self.layers
                ) if periodic_intph else assign_intph(self.ms_img, self.layers)
        else:
            self.intph_img = self.ms_img

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
    
    def build_inp(self, inp_filename):
        heading = Heading()
        nodes = GridNodes.from_intph_img(self.intph_img, self.scale)
        elements = CPE4RElements(nodes)
        # MATERIALS
        if self.has_interphase:
            has_matrix = True # flag indicating whether the matrix region still exists
            # since increasing interphase thickness might cover the whole matrix region
            materials = [] # init a list for Material objects
            elsets = ElementSet.from_intph_image(self.intph_img)
            # if len(elsets) <= 1 + len(self.layers) (number of interphase):
            #     case 2: 1 filler set, multiple interphase set
            # else:
            #     case 1: 1 filler set, multiple interphase set, 1 matrix set
            if len(elsets) <= 1 + len(self.layers):
                has_matrix = False
            # filler
            filler_material = Material(
                elsets[0],
                density=self.fil_density,
                youngs=self.fil_youngs,
                poisson=self.fil_poisson
            )
            materials.append(filler_material)
            # interphase
            intph_index_upper = len(elsets) - 1 if has_matrix else len(elsets)
            for i in range(1,intph_index_upper):
                intph_material = ViscoelasticMaterial(
                    elsets[i],
                    density=self.intph_density,
                    poisson=self.intph_poisson,
                    youngs=self.intph_youngs,
                    freq=self.mtx_freq,
                    youngs_cplx=self.mtx_youngs_cplx,
                    shift=self.intph_shift,
                    left_broadening=self.intph_l_brd,
                    right_broadening=self.intph_r_brd
                )
                materials.append(intph_material)
            # matrix
            if has_matrix:
                mat_material = ViscoelasticMaterial(
                    elsets[-1],
                    density=self.mtx_density,
                    poisson=self.mtx_poisson,
                    shift=self.mtx_shift,
                    left_broadening=1,
                    right_broadening=1,
                    youngs=self.mtx_youngs,
                    freq=self.mtx_freq,
                    youngs_cplx=self.mtx_youngs_cplx
                )
                materials.append(mat_material)
        else:
            filler_elset, mat_elset = ElementSet.from_intph_image(self.intph_img)
            # filler
            filler_material = Material(
                filler_elset,
                density=self.fil_density,
                youngs=self.fil_youngs,
                poisson=self.fil_poisson
            )
            # matrix
            mat_material = ViscoelasticMaterial(
                mat_elset,
                density=self.mtx_density,
                poisson=self.mtx_poisson,
                shift=self.mtx_shift,
                left_broadening=1, # no broadening for matrix
                right_broadening=1, # no broadening for matrix
                youngs=self.mtx_youngs,
                freq=self.mtx_freq,
                youngs_cplx=self.mtx_youngs_cplx
            )
            materials = [filler_material, mat_material]
        # BOUNDARY
        disp_bnd = DisplacementBoundaryCondition(
            nodes.virtual_node,
            first_dof=self.disp_BC_dof_first,
            last_dof=self.disp_BC_dof_last,
            displacement=self.displacement,
        )
        pbcs = PeriodicBoundaryConditions(nodes=nodes, disp_bnd=disp_bnd)

        # StepParameters            
        step_parm = StepParameters(
            disp_bnd_nodes=[disp_bnd],
            f_initial=self.fmin,
            f_final=self.fmax,
            f_count=self.num_freq,
            bias=1) # default value
        # write to inp file
        write_abaqus_input(heading=heading, nodes=nodes, elements=elements,
            materials=materials, bcs=pbcs, step_parm=step_parm, path=inp_filename)
