import pathlib

import numpy as np

from microstructure_ve import (
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
    load_viscoelasticity,
)


scale = 0.0025
layers = 5
displacement = 0.005

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

ms_img = np.load("ms.npy")
intph_img = assign_intph(ms_img, [layers])

base_path = pathlib.Path(__file__).parent
youngs_path = base_path / "PMMA_shifted_R10_data.txt"
freq, youngs_cplx = load_viscoelasticity(youngs_path)
# This is one way to assign a long term modulus, but it is not universal!
# Another strategy is to use 0 for true viscoelastic liquids.
# Pick something physically reasonable for your system.
youngs_plat = youngs_cplx[0].real

sections = []

heading = Heading()
nodes = GridNodes(1 + np.array(intph_img.shape), scale)
bnodes = BoundaryNodes()
elements = CPE4RElements(nodes.shape)
sections.extend((heading, nodes, bnodes, elements))

nsets_lr = NodeSet.from_image_and_slicedict(intph_img, sides_lr)
nsets_c = NodeSet.from_image_and_slicedict(intph_img, corners)
nsets_tb = BigNodeSet.from_image_and_slicedict(intph_img, sides_tb)
sections.extend((*nsets_lr, *nsets_c, *nsets_tb))

eqs = [EqualityEquation(nsets, 1, bnodes.lr_nset) for nsets in zip(*nsets_lr)]
eqs += [EqualityEquation(nsets, 2, bnodes.lr_nset) for nsets in zip(*nsets_lr)][1:-1]
eqs += [EqualityEquation(nsets, 1, bnodes.tb_nset) for nsets in zip(*nsets_tb)]
eqs += [EqualityEquation([f, c], 2) for f, (c,) in zip(nsets_tb, nsets_c)]
sections.extend(eqs)

filler_elset, intph_elset, mat_elset = ElementSet.from_intph_image(intph_img)
sections.extend([filler_elset, intph_elset, mat_elset])

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
sections.extend([filler_material, intph_material, mat_material])

step_parm = StepParameters(bnodes, displacement)
sections.append(step_parm)

with open("example.inp", "w") as inp_file_obj:
    for section in sections:
        section.to_inp(inp_file_obj)

# from microstructure_ve import run_job, read_odb
#
# run_job("abaqus", 4)
# read_odb("abaqus", displacement)
