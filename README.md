# microstructure-ve
A repository for FEA code developed by members of the Brinson Group at Duke University. Packages specialized for the analysis of polymer nanoparticle composites (PNCs).

Active Maintainers: [Richard Sheridan](richard.sheridan@duke.edu "Contact Richard")†, [Anqi (Claire) Lin](anqi.lin@duke.edu "Contact Claire")†, [Nicholas Finan](nicholas.finan@duke.edu "Contact Nicholas")† 

## Install

```sh
pip install -e .          # numpy + scipy; builds the ABAQUS .inp path and the spec
```

The optional **DOLFINx** finite-element backend additionally needs the FEniCSx stack
(`dolfinx`, `dolfinx_mpc`, `basix`, and a complex-scalar `petsc4py`), which is typically
provided through conda-forge rather than pip — install the package into that environment too.

The package `__init__` is intentionally empty; import the solver-neutral spec dataclasses
from their submodules (or all at once via `from microstructure_ve.api import *`), and each
backend from its own module.

## Quick start

Build a small representative volume element (RVE) and emit an ABAQUS input deck:

```python
import numpy as np
from microstructure_ve.core import GridNodes, GridElements, ElementSet
from microstructure_ve.materials import Material, PlasticMaterial
from microstructure_ve.boundary import (
    PeriodicBoundaryCondition, FixedBoundaryCondition, DisplacementBoundaryCondition,
)
from microstructure_ve.steps import Model, Simulation, Step, Dynamic, Static, Heading
from microstructure_ve.backends.abaqus import write_inp

img = np.ones((4, 4), dtype=int); img[1:3, 1:3] = 0           # 0 = particle, 1 = matrix
nodes = GridNodes.from_matl_img(img, scale=0.0025)
elements = GridElements(nodes, type="CPE4")                    # 2D, full integration
particle_elset, matrix_elset = ElementSet.from_matl_img(img)   # sorted ascending by value
materials = [
    Material(particle_elset, density=2.65e-15, poisson=0.15, youngs=5e5),
    PlasticMaterial(matrix_elset, density=1.18e-15, poisson=0.35, youngs=3e3,
                    yield_stress=[40.0, 25.0], plastic_strain=[0.0, 0.05]),  # softens 40->25 MPa by 5% eps_pl
]

# corner-driven periodic BCs with an x drive
origin, x_drive, y_drive = (nodes.nsets[k] for k in ("X0Y0", "X1Y0", "X0Y1"))
model = Model(nodes=nodes, elements=elements, materials=materials, bcs=[
    PeriodicBoundaryCondition(nodes=nodes),
    FixedBoundaryCondition(origin, dofs=[1, 2]),               # pin origin
    FixedBoundaryCondition(x_drive, dofs=[2]),                 # suppress shear
    FixedBoundaryCondition(y_drive, dofs=[1]),
    DisplacementBoundaryCondition(x_drive, 1, 1, 0.0),         # baseline
])
dyn_step = Step(subsections=[
    Dynamic(f_initial=1e-7, f_final=1e5, f_count=30, bias=1),
    DisplacementBoundaryCondition(x_drive, 1, 1, 0.005),       # harmonic macro drive (perturbation)
], perturbation=True)
# a general (nonlinear) static step that loads the matrix past yield into the softening branch
static_step = Step(subsections=[
    Static(),
    DisplacementBoundaryCondition(x_drive, 1, 1, 4e-4),        # ~4% macro x-strain (Lx = 0.01)
], perturbation=False)
sim = Simulation(heading=Heading("quick start"), model=model, steps=[dyn_step, static_step])

write_inp(sim, "rve.inp")          # -> ABAQUS input deck
```

Solve the same `sim` license-free with the DOLFINx backend (needs the FEniCSx env):

```python
from microstructure_ve.backends.dolfinx import run

result = run(sim, lateral_bc="free")   # ndarray (n_freqs + 1, 1 + 3*dim)
# columns: [frame_value, RF_Real_1..d, RF_Imag_1..d, U_1..d]
# homogenized modulus along x:  E*_x(f) = (RF_Real_1 + 1j*RF_Imag_1) / (cross_area * exx)
# the trailing row is the static step (frame_value 1.0); the DOLFINx backend solves it
# linear-elastic -- the *Plastic table is honored only by the ABAQUS deck above.
```

`lateral_bc="free"` lets the cell contract laterally (apparent uniaxial modulus);
`"confined"` holds the lateral macro strains at zero. Pass `workers=N` to fan the
independent per-frequency solves across processes (guard the call under
`if __name__ == "__main__":`).

## Documentation

## Attributions
†Duke University, Brinson Group
