import pathlib

import numpy as np

from microstructure_ve import (
    Heading,
    GridNodes,
    PeriodicBoundaryConditions,
    CPE4RElements,
    ElementSet,
    ViscoelasticMaterial,
    Material,
    StepParameters,
    periodic_assign_intph,
    load_viscoelasticity,
    write_abaqus_input, NodeSet, DisplacementBoundaryNode
)

scale = 0.0025
displacement = .005
layers = 5

ms_img = np.load("ms.npy")
intph_img = periodic_assign_intph(ms_img, [layers])

base_path = pathlib.Path(__file__).parent
youngs_path = base_path / "PMMA_shifted_R10_data.txt"
freq, youngs_cplx = load_viscoelasticity(youngs_path)
# This is one way to assign a long term modulus, but it is not universal!
# Another strategy is to use 0 for true viscoelastic liquids.
# Pick something physically reasonable for your system.
youngs_plat = youngs_cplx[0].real

heading = Heading()
nodes = GridNodes.from_intph_img(intph_img, scale)
elements = CPE4RElements(nodes)

make_set = lambda name: NodeSet.from_side_name(name, nodes)
right, botmright = driving_nsets = [make_set("RightSurface"), make_set("BotmRight")]
surfaces = [
    [make_set("LeftSurface"), right],
    [make_set("BotmSurface"), make_set("TopSurface")],
    # [make_set("TopLeft"), make_set("TopRight")],
    [botmright, make_set("TopLeft")],
    # [make_set("BotmRight"), make_set("BotmLeft")],
]
disp_bnd_node = DisplacementBoundaryNode(
    nodes.virtual_node,
    first_dof=1,
    last_dof=1,
    displacement=displacement,
)
bcs = PeriodicBoundaryConditions(surfaces, driving_nsets=driving_nsets, disp_bnd_node=disp_bnd_node)
filler_elset, intph_elset, mat_elset = ElementSet.from_intph_image(intph_img)

filler_material = Material(filler_elset, density=2.65e-15, youngs=5e5, poisson=0.15)
intph_material = ViscoelasticMaterial(
    intph_elset,
    density=1.18e-15,
    poisson=0.35,
    shift=-4.0,
    youngs=youngs_plat,
    freq=freq,
    youngs_cplx=youngs_cplx,
    left_broadening=1.8,
    right_broadening=1.5,
)
mat_material = ViscoelasticMaterial(
    mat_elset,
    density=1.18e-15,
    poisson=0.35,
    youngs=youngs_plat,
    freq=freq,
    youngs_cplx=youngs_cplx,
    shift=-6.0,
)
materials = [filler_material, intph_material, mat_material]

step_parm = StepParameters(
    [disp_bnd_node],
    f_initial=1e-7,
    f_final=1e5,
    f_count=30,
    bias=1,
)

write_abaqus_input(heading=heading, nodes=nodes, elements=elements, materials=materials,
                   bcs=bcs, step_parm=step_parm, path="example.inp")

# from microstructure_ve import run_job, read_odb
#
# run_job("abaqus", 4)
# read_odb("abaqus", displacement)
