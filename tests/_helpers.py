"""Synthetic fixtures shared across the test suite.

Tests build **small synthetic microstructures** here; they never load the 50x50
``ms.npy`` (that file is reserved for running ``example.py``). The real
``PMMA_shifted_R10_data.txt`` master curve *is* used where a viscoelastic material is
needed, so the emitted tables match what the package actually produces in practice.
"""
import pathlib

import numpy as np

from microstructure_ve.boundary import (
    DisplacementBoundaryCondition,
    FixedBoundaryCondition,
    PeriodicBoundaryCondition,
)
from microstructure_ve.core import ElementSet, GridElements, GridNodes
from microstructure_ve.materials import Material, TabularViscoelasticMaterial
from microstructure_ve.steps import Dynamic, Heading, Model, Simulation, Step
from microstructure_ve.utils import load_viscoelasticity, periodic_assign_intph

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PMMA_DATA = REPO_ROOT / "PMMA_shifted_R10_data.txt"

SCALE = 0.0025
DISPLACEMENT = 0.005


def synthetic_microstructure():
    """A tiny deterministic 6x6 phase image with a central 2x2 particle.

    ``periodic_assign_intph([1])`` then yields all three phases (0 = filler particle,
    1 = interphase, 2 = matrix), exercising the multi-material emission paths without
    any external data file.
    """
    img = np.ones((6, 6), dtype=int)
    img[2:4, 2:4] = 0  # particle (must contain at least one zero)
    return periodic_assign_intph(img, [1])


# The real master curve is dense (~300 R10 points). The golden .inp only needs to
# exercise the *Viscoelastic table emission, not its resolution, so the synthetic
# fixture subsamples it to keep the committed golden small. Material-physics tests
# (complex_modulus round-trips) load the full curve directly instead.
_TABLE_STRIDE = 8


def synthetic_materials(intph_img):
    """Filler (elastic) + interphase/matrix (viscoelastic off the PMMA data) elsets.

    Mirrors ``example.py``'s material choices so the synthetic ``.inp`` covers the
    elastic and tabular-viscoelastic keyword blocks.
    """
    freq, youngs_cplx = load_viscoelasticity(PMMA_DATA)
    # subsample but always keep the endpoints (youngs_plat uses index 0)
    keep = np.unique(np.r_[np.arange(0, len(freq), _TABLE_STRIDE), len(freq) - 1])
    freq, youngs_cplx = freq[keep], youngs_cplx[keep]
    youngs_plat = youngs_cplx[0].real
    filler_elset, intph_elset, mat_elset = ElementSet.from_matl_img(intph_img)
    return [
        Material(filler_elset, density=2.65e-15, youngs=5e5, poisson=0.15),
        TabularViscoelasticMaterial(
            intph_elset,
            density=1.18e-15,
            poisson=0.35,
            shift=-4.0,
            youngs=youngs_plat,
            freq=freq,
            youngs_cplx=youngs_cplx,
            left_broadening=1.8,
            right_broadening=1.5,
        ),
        TabularViscoelasticMaterial(
            mat_elset,
            density=1.18e-15,
            poisson=0.35,
            youngs=youngs_plat,
            freq=freq,
            youngs_cplx=youngs_cplx,
            shift=-6.0,
        ),
    ]


def synthetic_simulation():
    """A complete small ``Simulation`` mirroring ``example.py`` on the synthetic RVE.

    Corner-driven periodic BCs with a free lateral edge (Poisson contraction) and a
    30-frequency steady-state dynamic step. Deterministic, so it backs a golden ``.inp``.
    """
    intph_img = synthetic_microstructure()
    nodes = GridNodes.from_matl_img(intph_img, SCALE)
    elements = GridElements(nodes, type="CPE4R")
    materials = synthetic_materials(intph_img)

    origin = nodes.nsets["X0Y0"]
    x_macro = nodes.nsets["X1Y0"]
    y_macro = nodes.nsets["X0Y1"]
    model = Model(
        nodes=nodes,
        elements=elements,
        materials=materials,
        bcs=[
            PeriodicBoundaryCondition(nodes=nodes),
            FixedBoundaryCondition(origin, dofs=[1, 2]),
            FixedBoundaryCondition(x_macro, dofs=[2]),
            FixedBoundaryCondition(y_macro, dofs=[1]),
            DisplacementBoundaryCondition(x_macro, first_dof=1, last_dof=1, displacement=0.0),
        ],
    )
    disp_bc = DisplacementBoundaryCondition(
        x_macro, first_dof=1, last_dof=1, displacement=DISPLACEMENT
    )
    dyn = Dynamic(f_initial=1e-7, f_final=1e5, f_count=30, bias=1)
    step = Step(subsections=[dyn, disp_bc], perturbation=True)
    return Simulation(
        heading=Heading("Synthetic test RVE"), model=model, steps=[step]
    )


def homogeneous_simulation(n=4, dim=2, E=3000.0, nu=0.3, scale=SCALE,
                           displacement=DISPLACEMENT, etype=None):
    """A single-material (homogeneous) RVE for analytic FE checks.

    One elastic material fills an ``n**dim`` grid; periodic BCs + an x drive. Lets the
    FE backend's homogenized stress be compared to closed-form uniaxial results.
    """
    img = np.zeros((n,) * dim, dtype=int)  # one material everywhere
    nodes = GridNodes.from_matl_img(img, scale)
    if etype is None:
        etype = "CPE4" if dim == 2 else "C3D8"
    elements = GridElements(nodes, type=etype)
    (elset,) = ElementSet.from_matl_img(img)
    materials = [Material(elset, density=1.0, poisson=nu, youngs=E)]

    model = Model(nodes=nodes, elements=elements, materials=materials,
                  bcs=[PeriodicBoundaryCondition(nodes=nodes)])
    drive = nodes.nsets["X1Y0"]  # only its displacement value is read by the FE backend
    disp_bc = DisplacementBoundaryCondition(drive, 1, 1, displacement)
    dyn = Dynamic(f_initial=1.0, f_final=1.0, f_count=1, bias=1)
    step = Step(subsections=[dyn, disp_bc], perturbation=True)
    return Simulation(heading=Heading("Homogeneous RVE"), model=model, steps=[step])
