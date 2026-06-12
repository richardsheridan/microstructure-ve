"""3D DOLFINx example: homogenize a synthetic 3D RVE with FEniCSx.

The repo ships a 2D microstructure (ms.npy); this builds a small synthetic 3D RVE (a stiff
cubic inclusion in a softer matrix) to exercise the backend's 3D path. Demonstrates both
confined and free-lateral loading and writes example_dolfinx_3d-reaction-force.tsv.

Run in the fenicsx conda env:
    /home/rjs80/miniconda3/envs/fenicsx/bin/python example_dolfinx_3d.py

The backend is dimension-general; the only differences from 2D are hexahedral elements,
three macro axes, and B-bar with κ = λ + 2μ/3. Validated against an ABAQUS C3D8 run to ~1e-8.
"""
import numpy as np

from microstructure_ve import (
    GridNodes, GridElements, ElementSet, Material,
    PeriodicBoundaryCondition, FixedBoundaryCondition, DisplacementBoundaryCondition,
    Dynamic, Step, Model, Simulation, Heading,
)
import dolfinx_backend as db

scale = 0.0025
displacement = 0.005

img = np.zeros((8, 8, 8), dtype=int)
img[2:6, 2:6, 2:6] = 1  # central stiff cube inclusion

nodes = GridNodes.from_matl_img(img, scale)
elements = GridElements(nodes, type="C3D8")  # full integration (B-bar), matches the backend
matrix_elset, filler_elset = ElementSet.from_matl_img(img)
materials = [
    Material(matrix_elset, density=1.18e-15, poisson=0.3, youngs=1.0e3),
    Material(filler_elset, density=1.18e-15, poisson=0.3, youngs=5.0e3),
]
ns = nodes.nsets
model = Model(
    nodes=nodes, elements=elements, materials=materials,
    bcs=[
        PeriodicBoundaryCondition(nodes=nodes),
        FixedBoundaryCondition(ns["X0Y0Z0"], dofs=[1, 2, 3]),
        FixedBoundaryCondition(ns["X1Y0Z0"], dofs=[2, 3]),
        DisplacementBoundaryCondition(ns["X1Y0Z0"], 1, 1, 0.0),
    ],
)
step = Step(
    subsections=[Dynamic(f_initial=1.0, f_final=1.0, f_count=1, bias=1),
                 DisplacementBoundaryCondition(ns["X1Y0Z0"], 1, 1, displacement)],
    perturbation=True,
)
sim = Simulation(heading=Heading("3D RVE"), model=model, steps=[step])

freqs = db.default_frequencies(sim)
conf = db.run(sim, freqs=freqs, lateral="confined", bbar=True)
free = db.run(sim, freqs=freqs, lateral="free", bbar=True,
              output_path="example_dolfinx_3d-reaction-force.tsv")

# Effective x-modulus E*_xx = (σ̄_xx) / ε̄_xx = RF1 / (cross_area) / (delta / Lx)
Lx = (nodes.shape[-1] - 1) * scale
cross = ((nodes.shape[-2] - 1) * scale) * ((nodes.shape[-3] - 1) * scale)
exx = displacement / Lx
print("confined  E*_xx = %.1f" % (conf[0, 1] / cross / exx))
print("free-lat. E*_xx = %.1f  (softer; Poisson relief)" % (free[0, 1] / cross / exx))
print("wrote example_dolfinx_3d-reaction-force.tsv")
