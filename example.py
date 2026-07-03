import pathlib

import numpy as np

from microstructure_ve.backends.abaqus import write_inp
from microstructure_ve.boundary import (
    BoundaryCondition,
    Fixed,
    PeriodicBoundaryConstraint,
    Prescribed,
)
from microstructure_ve.core import ElementSet, GridElements, GridNodes
from microstructure_ve.constitutive import Elastic, TabularViscoelastic
from microstructure_ve.materials import Material
from microstructure_ve.steps import Dynamic, Heading, Model, Simulation, Step
from microstructure_ve.utils import load_viscoelasticity, periodic_assign_intph

scale = 0.0025
displacement = 0.005
layers = 5

# Resolve inputs relative to this script (not the CWD) so the job can be written
# and run from a dedicated per-simulation working directory.
base_path = pathlib.Path(__file__).parent
work_dir = base_path / "abaqus-work" / "example"
work_dir.mkdir(parents=True, exist_ok=True)

ms_img = np.load(base_path / "ms.npy")
intph_img = periodic_assign_intph(ms_img, [layers])

youngs_path = base_path / "PMMA_shifted_R10_data.txt"
freq, youngs_cplx = load_viscoelasticity(youngs_path)
# This is one way to assign a long term modulus, but it is not universal!
# Another strategy is to use 0 for true viscoelastic liquids.
# Pick something physically reasonable for your system.
youngs_plat = youngs_cplx[0].real

heading = Heading("Example RVE simulation")
nodes = GridNodes.from_matl_img(intph_img, scale)
elements = GridElements(nodes, type="CPE4R")
filler_elset, intph_elset, mat_elset = ElementSet.from_matl_img(intph_img)

filler_material = Material(filler_elset, density=2.65e-15,
                           response=Elastic(youngs=5e5, poisson=0.15))
intph_material = Material(
    intph_elset,
    density=1.18e-15,
    response=TabularViscoelastic(
        poisson=0.35,
        shift=-4.0,
        youngs=youngs_plat,
        freq=freq,
        youngs_cplx=youngs_cplx,
        left_broadening=1.8,
        right_broadening=1.5,
    ),
)
mat_material = Material(
    mat_elset,
    density=1.18e-15,
    response=TabularViscoelastic(
        poisson=0.35,
        youngs=youngs_plat,
        freq=freq,
        youngs_cplx=youngs_cplx,
        shift=-6.0,
    ),
)
# PeriodicBoundaryConstraint ties each face to its opposite through the reference
# corner nodes, which carry the macroscopic deformation. Driving the RVE means
# constraining those corners:
#   X0Y0 - pinned origin (removes rigid-body translation)
#   X1Y0 - driven in x; its dof 2 is held to suppress macroscopic shear
#   X0Y1 - dof 1 held to suppress shear; dof 2 is LEFT FREE so the cell can
#          contract laterally (Poisson). Use Fixed(dofs=[1, 2]) here instead for a
#          laterally-confined (plane-strain-clamped) test.
origin = nodes.nsets["X0Y0"]
x_macro = nodes.nsets["X1Y0"]
y_macro = nodes.nsets["X0Y1"]
model = Model(
    nodes=nodes,
    elements=elements,
    materials=[filler_material, intph_material, mat_material],
    bcs=[
        PeriodicBoundaryConstraint(nodes=nodes),
        BoundaryCondition(origin, Fixed(dofs=[1, 2])),
        BoundaryCondition(x_macro, Fixed(dofs=[2])),
        BoundaryCondition(y_macro, Fixed(dofs=[1])),
        BoundaryCondition(x_macro, Prescribed(dofs=[1], value=0.0)),
    ],
)

disp_bc = BoundaryCondition(x_macro, Prescribed(dofs=[1], value=displacement))
dyn = Dynamic(
    f_initial=1e-7,
    f_final=1e5,
    f_count=30,
    bias=1,
)
step = Step(subsections=[dyn, disp_bc], perturbation=True)

inp_path = work_dir / "example.inp"
write_inp(Simulation(heading=heading, model=model, steps=[step]), inp_path)

# Emit the standalone odb reader next to the job, then run the job from inside the
# per-simulation working directory and extract the reaction forces. The reaction
# work-conjugate to the applied displacement is carried by the X1Y0 corner node, so
# that is the node set the reader reads:
# from microstructure_ve.backends.abaqus import write_odb_reader
# write_odb_reader(work_dir / "read_abaqus_odb.py")
# cd abaqus-work/example
# /path/to/abaqus job=example cpus=4 interactive
# /path/to/abaqus python read_abaqus_odb.py example X1Y0

# import csv
# tsv = csv.reader(open("abaqus-work/example/example-reaction-force.tsv", "r"), dialect=csv.excel_tab)
