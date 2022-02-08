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
)


scale = 0.0025
layers = 5
displacement = 0.005
youngs_plat = 100  # MegaPascals

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
intph_img = assign_intph(ms_img, layers)

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

elsets = ElementSet.from_intph_image(intph_img)
sections.extend(elsets)

materials = [
    ViscoelasticMaterial(elset, shift=-4.0, youngs=youngs_plat) for elset in elsets
]
materials[0] = Material(elsets[0], density=2.65e-15, youngs=5e5, poisson=0.15)
materials[-1] = ViscoelasticMaterial(
    elsets[-1],
    youngs=youngs_plat,
    shift=-6.0,
    left_broadening=1.0,
    right_broadening=1.0,
)
sections.extend(materials)

step_parm = StepParameters(bnodes, displacement)
sections.append(step_parm)

with open("abaqus.inp", "w") as inp_file_obj:
    for section in sections:
        section.to_inp(inp_file_obj)

# from microstructure_ve import run_job, read_odb
#
# run_job("abaqus", 4)
# read_odb("abaqus", displacement)
