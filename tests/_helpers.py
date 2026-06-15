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


def synthetic_materials(intph_img, stride=_TABLE_STRIDE):
    """Filler (elastic) + interphase/matrix (viscoelastic off the PMMA data) elsets.

    Mirrors ``example.py``'s material choices so the synthetic ``.inp`` covers the
    elastic and tabular-viscoelastic keyword blocks. ``stride`` subsamples the master
    curve (default keeps the golden ``.inp`` small; ``stride=1`` uses the full,
    denser table so an ABAQUS-vs-FE interpolation comparison is tight).
    """
    freq, youngs_cplx = load_viscoelasticity(PMMA_DATA)
    # subsample but always keep the endpoints (youngs_plat uses index 0)
    keep = np.unique(np.r_[np.arange(0, len(freq), stride), len(freq) - 1])
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


def _macro_corner_bcs(nodes, lateral_bc):
    """Corner-driven macro BCs for x-uniaxial loading (the ABAQUS reference scheme).

    Pins the origin, drives the x reference corner, and holds the lateral reference
    corner(s) for confined loading (or frees the lateral normal for 2D ``free``).
    Returns ``(bcs, drive_nset)``.
    """
    dim = nodes.dim
    if dim == 2:
        origin, xm, ym = (nodes.nsets[k] for k in ("X0Y0", "X1Y0", "X0Y1"))
        bcs = [FixedBoundaryCondition(origin, [1, 2]), FixedBoundaryCondition(xm, [2])]
        bcs.append(FixedBoundaryCondition(ym, [1, 2] if lateral_bc == "confined" else [1]))
        return bcs, xm
    if dim == 3:
        origin, xm, ym, zm = (
            nodes.nsets[k] for k in ("X0Y0Z0", "X1Y0Z0", "X0Y1Z0", "X0Y0Z1")
        )
        bcs = [FixedBoundaryCondition(origin, [1, 2, 3]), FixedBoundaryCondition(xm, [2, 3])]
        if lateral_bc == "confined":
            # hold both lateral normals: Eyy = Ezz = 0
            bcs += [FixedBoundaryCondition(ym, [1, 2, 3]), FixedBoundaryCondition(zm, [1, 2, 3])]
        else:
            # free: leave each lateral normal floating (ym dof 2, zm dof 3), pin only their
            # tangential dofs so the lateral macro normal *stresses* vanish (uniaxial stress)
            bcs += [FixedBoundaryCondition(ym, [1, 3]), FixedBoundaryCondition(zm, [1, 2])]
        return bcs, xm
    raise ValueError("unsupported dim")


def oracle_simulation_2d(lateral_bc="confined"):
    """Small 2D CPE4 viscoelastic RVE, corner-driven -- the ABAQUS parity oracle.

    The same Simulation is solved by ABAQUS (corner-driven) and by the FE backend
    (which ignores the corner BCs and applies ``lateral_bc`` via its own pure-periodic
    constraints), so their homogenized x-response must agree.
    """
    img = synthetic_microstructure()
    nodes = GridNodes.from_matl_img(img, SCALE)
    elements = GridElements(nodes, type="CPE4")  # full integration to match FE Q1
    # full-resolution master curve so FE's log-f interp matches ABAQUS's linear-f interp
    materials = synthetic_materials(img, stride=1)
    bcs, drive = _macro_corner_bcs(nodes, lateral_bc)
    bcs = (
        [PeriodicBoundaryCondition(nodes=nodes)]
        + bcs
        + [DisplacementBoundaryCondition(drive, 1, 1, 0.0)]  # zero baseline
    )
    model = Model(nodes=nodes, elements=elements, materials=materials, bcs=bcs)
    disp = DisplacementBoundaryCondition(drive, 1, 1, DISPLACEMENT)
    dyn = Dynamic(f_initial=1e-7, f_final=1e5, f_count=30, bias=1)
    step = Step(subsections=[dyn, disp], perturbation=True)
    return Simulation(heading=Heading("oracle 2d " + lateral_bc), model=model, steps=[step])


def oracle_simulation_3d():
    """Small 3D C3D8 heterogeneous *elastic* RVE, corner-driven (confined).

    Elastic (no tabular interpolation) so FE-vs-ABAQUS agreement is near machine
    precision, isolating the FE/homogenization.
    """
    img = (np.indices((3, 3, 3)).sum(0) % 2)  # two-material checkerboard
    nodes = GridNodes.from_matl_img(img, SCALE)
    elements = GridElements(nodes, type="C3D8")
    sets = ElementSet.from_matl_img(img)
    # realistic small densities (kg/micron^3) so ABAQUS's inertia term -w^2 M stays
    # negligible vs the quasi-static FE backend even at the top sweep frequency
    materials = [
        Material(sets[0], density=2.65e-15, poisson=0.3, youngs=1000.0),
        Material(sets[1], density=2.65e-15, poisson=0.3, youngs=5000.0),
    ]
    bcs, drive = _macro_corner_bcs(nodes, "confined")
    bcs = (
        [PeriodicBoundaryCondition(nodes=nodes)]
        + bcs
        + [DisplacementBoundaryCondition(drive, 1, 1, 0.0)]
    )
    model = Model(nodes=nodes, elements=elements, materials=materials, bcs=bcs)
    disp = DisplacementBoundaryCondition(drive, 1, 1, DISPLACEMENT)
    dyn = Dynamic(f_initial=1e-7, f_final=1e5, f_count=2, bias=1)
    step = Step(subsections=[dyn, disp], perturbation=True)
    return Simulation(heading=Heading("oracle 3d elastic"), model=model, steps=[step])


def homogeneous_simulation(n=4, dim=2, E=3000.0, nu=0.3, scale=SCALE,
                           displacement=DISPLACEMENT, etype=None,
                           lateral_bc="confined", f_count=1):
    """A single-material (homogeneous) RVE for analytic FE checks.

    One elastic material fills an ``n**dim`` grid; corner-driven periodic BCs + an x
    drive. The ``lateral_bc`` ("confined"/"free") is encoded in the corner BCs so the FE
    backend can parse it from the Simulation (see ``_loading.macro_loading``); the
    homogenized stress can then be compared to closed-form uniaxial results. ``f_count``
    sets the number of frequency points (a single point by default).
    """
    img = np.zeros((n,) * dim, dtype=int)  # one material everywhere
    nodes = GridNodes.from_matl_img(img, scale)
    if etype is None:
        etype = "CPE4" if dim == 2 else "C3D8"
    elements = GridElements(nodes, type=etype)
    (elset,) = ElementSet.from_matl_img(img)
    materials = [Material(elset, density=2.65e-15, poisson=nu, youngs=E)]

    corner_bcs, drive = _macro_corner_bcs(nodes, lateral_bc)
    bcs = (
        [PeriodicBoundaryCondition(nodes=nodes)]
        + corner_bcs
        + [DisplacementBoundaryCondition(drive, 1, 1, 0.0)]  # zero baseline
    )
    model = Model(nodes=nodes, elements=elements, materials=materials, bcs=bcs)
    disp_bc = DisplacementBoundaryCondition(drive, 1, 1, displacement)
    dyn = Dynamic(f_initial=1.0, f_final=1.0, f_count=f_count, bias=1)
    step = Step(subsections=[dyn, disp_bc], perturbation=True)
    return Simulation(heading=Heading("Homogeneous RVE"), model=model, steps=[step])
