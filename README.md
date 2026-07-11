# microstructure-ve
A repository for FEA code developed by members of the Brinson Group at Duke University. Packages specialized for the analysis of polymer nanoparticle composites (PNCs).

## Install

The base package (ABAQUS `.inp` emitter + solver-neutral spec) needs only numpy and scipy;
the optional **DOLFINx** finite-element backend needs the FEniCSx stack, which is provided
through conda-forge (not pip). The package imports dolfinx lazily, so the base install stays
importable without it. Two conda env files are provided at the repo root.

**Base only (numpy + scipy) — Linux, macOS, Windows.**

```sh
pip install -e .                              # into any existing Python ≥ 3.10
# or an isolated conda env:
conda env create -f environment-numpy.yml     # creates env "msve"
conda activate msve
```

**DOLFINx FE backend — Linux & macOS (incl. Apple Silicon).** conda-forge ships `linux-64`,
`osx-64` and `osx-arm64` builds:

```sh
conda env create -f environment.yml           # creates env "fenicsx" (dolfinx + complex PETSc)
conda activate fenicsx
```

**DOLFINx FE backend — Windows.** conda-forge has **no native Windows DOLFINx build**.
Run the Linux instructions inside **WSL2** (Ubuntu).

The package `__init__` is intentionally empty; import the solver-neutral spec dataclasses
from their submodules (or all at once via `from microstructure_ve.api import *`), and each
backend from its own module.

## Quick start

Build a small representative volume element (RVE) and emit an ABAQUS input deck:

```python
import numpy as np
from microstructure_ve.core import GridNodes, GridElements, ElementSet
from microstructure_ve.materials import Material
from microstructure_ve.constitutive import Elastic, Plastic
from microstructure_ve.boundary import (
    PeriodicBoundaryConstraint, BoundaryCondition, Fixed, Prescribed,
)
from microstructure_ve.steps import Model, Simulation, Step, Dynamic, Static, Heading
from microstructure_ve.backends.abaqus import write_inp

img = np.ones((4, 4), dtype=int); img[1:3, 1:3] = 0           # 0 = particle, 1 = matrix
nodes = GridNodes.from_matl_img(img, scale=0.0025)
elements = GridElements(nodes, type="CPE4")                    # 2D, full integration
particle_elset, matrix_elset = ElementSet.from_matl_img(img)   # sorted ascending by value
materials = [
    Material(particle_elset, density=2.65e-15, response=Elastic(poisson=0.15, youngs=5e5)),
    Material(matrix_elset, density=1.18e-15, response=Plastic(
        poisson=0.35, youngs=3e3,
        yield_stress=[40.0, 25.0], plastic_strain=[0.0, 0.05])),  # softens 40->25 MPa by 5% eps_pl
]

# corner-driven periodic BCs with an x drive
origin, x_drive, y_drive = (nodes.nsets[k] for k in ("X0Y0", "X1Y0", "X0Y1"))
model = Model(nodes=nodes, elements=elements, materials=materials, bcs=[
    PeriodicBoundaryConstraint(nodes=nodes),
    BoundaryCondition(origin, Fixed(dofs=[1, 2])),             # pin origin
    BoundaryCondition(x_drive, Fixed(dofs=[2])),               # suppress shear
    BoundaryCondition(y_drive, Fixed(dofs=[1])),
    BoundaryCondition(x_drive, Prescribed(dofs=[1], value=0.0)),   # baseline
])
dyn_step = Step(subsections=[
    Dynamic(f_initial=1e-7, f_final=1e5, f_count=30, bias=1),
    BoundaryCondition(
        x_drive,
        Prescribed(dofs=[1], value=0.005),
    ),
], perturbation=True)  # harmonic macro drive (perturbation)
# a general (nonlinear) static step that loads the matrix past yield into the softening branch
static_step = Step(subsections=[
    Static(),
    BoundaryCondition(x_drive, Prescribed(dofs=[1], value=4e-4)),   # ~4% macro x-strain (Lx = 0.01)
], perturbation=False)
sim = Simulation(heading=Heading("quick start"), model=model, steps=[dyn_step, static_step])

write_inp(sim, "rve.inp")          # -> ABAQUS input deck
```

Solve the same `sim` with the DOLFINx backend (needs the FEniCSx env):

```python
from microstructure_ve.backends.dolfinx import run

result = run(sim)                      # ndarray (n_freqs + 1, 1 + 3*dim)
# columns: [frame_value, RF_Real_1..d, RF_Imag_1..d, U_1..d]
# homogenized modulus along x:  E*_x(f) = (RF_Real_1 + 1j*RF_Imag_1) / (cross_area * exx)
# the trailing row is the static step (frame_value 1.0); the DOLFINx backend solves it
# linear-elastic -- the *Plastic table is honored only by the ABAQUS deck above.
```

Pass `workers=N` to fan the independent per-frequency solves
across processes (guard the call under `if __name__ == "__main__":`).

## Documentation
