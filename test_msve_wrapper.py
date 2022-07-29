# test msve_wrapper.py with the example provided in the microstructure_ve repo
from msve_wrapper import msve_wrapper
msve = msve_wrapper()
msve.load_param(
    mtx_density=1.18e-15,
    master_curve='PMMA_shifted_R10_data.txt',
    mtx_poisson=0.35,
    mtx_shift=-6.0,
    fil_density=2.65e-15,
    fil_youngs =5e5,
    fil_poisson =0.15,
    layers = 5,
    periodic_intph=True,
    intph_density=1.18e-15,
    intph_poisson=0.35,
    intph_shift=-4.0,
    intph_l_brd=1.8,
    intph_r_brd=1.5,
    displacement=0.005,
    disp_BC_dof_first=1,
    disp_BC_dof_last=1,
    scale = 0.0025, #micrometer/pixel
    fmin=1e-7,
    fmax=1e5,
    num_freq=30,
    ms_filename = 'ms.npy',
    reverse = False # Achtung! Already reversed.
)
msve.build_inp('msve_wrapper_generated.inp')