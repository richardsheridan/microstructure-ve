"""Shared Mechanical-MNIST problem builder (pure numpy; no dolfinx import).

Reproduces the uniaxial-extension benchmark of github.com/elejeune11/Mechanical-MNIST
(MIT license): a 28x28-unit plane-strain domain whose per-pixel Young's modulus comes from
an MNIST bitmap (E = value/255*99 + 1, nu = 0.3), coupled compressible Neo-Hookean
(``NeoHookean``), bottom edge clamped, top edge driven to u = (0, d) for the 13 successive
displacement values in ``DISP_VALS``, left/right edges free. Their reported output is the
total strain energy at each step minus the d = 0 step.

Their discretization (unstructured P2 triangles, ~140/side, nodal degree-1 E field) is not
reproduced -- this builder uses the package's structured Q1 grid with piecewise-constant
per-pixel E, optionally ``refine``-times finer than the bitmap (``refine_matl_img``, which
keeps the domain 28x28 length units), so agreement is mesh-convergence-limited.

``load_bitmap``/``load_psi`` read the committed data excerpts under
``tests/data/mechanical_mnist/`` (see the README there for provenance).
"""
import pathlib

import numpy as np

from microstructure_ve.boundary import BoundaryCondition, Fixed, Prescribed
from microstructure_ve.constitutive import NeoHookean
from microstructure_ve.core import ElementSet, GridElements, GridNodes, NodeSet
from microstructure_ve.materials import Material
from microstructure_ve.steps import Heading, Model, Simulation, Static, Step
from microstructure_ve.utils import refine_matl_img

# Their loading schedule (absolute top-edge displacement; the domain is 28 units tall, so
# the last value is 50% nominal strain). The d = 0 first entry is their energy baseline.
DISP_VALS = (0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
E_BACK = 1.0     # background (pixel 0) Young's modulus
E_HIGH = 100.0   # digit (pixel 255) Young's modulus
POISSON = 0.3
DENSITY = 2.65e-15  # inertia-free static solve; emitted but immaterial
DATA_DIR = pathlib.Path(__file__).parent / "data" / "mechanical_mnist"


def load_bitmap(name):
    """A committed 28x28 sample bitmap (integer values 0..255) by base name.

    ``name`` is e.g. ``"test_data_0"`` -- the files are verbatim copies of the benchmark's
    ``generate_dataset/input_data/*.txt``.
    """
    return np.loadtxt(DATA_DIR / (name + ".txt"))


def load_psi(which):
    """The committed (10, 13) strain-energy excerpt for ``which`` in {"train", "test"}.

    Rows are MNIST images 0-9 of that set; columns follow ``DISP_VALS`` (already baseline-
    subtracted by the benchmark, so column 0 is 0).
    """
    return np.loadtxt(DATA_DIR / f"summary_psi_{which}_first10.txt")


def pixel_youngs(value):
    """Their bitmap-to-modulus map: E = value/255*(E_HIGH - E_BACK) + E_BACK."""
    return float(value) / 255.0 * (E_HIGH - E_BACK) + E_BACK


def mm_simulation(bitmap, disp_vals=DISP_VALS, refine=1, flip=True):
    """The Mechanical-MNIST uniaxial-extension ``Simulation`` for one bitmap.

    ``refine`` subdivides each pixel into refine x refine Q1 elements (the domain stays
    28x28 length units -- the displacement schedule is absolute). ``flip`` flips the bitmap
    vertically so image row 0 (the top of the digit) lands at y = 28, matching the image
    orientation on the grid (pixel row i of the grid occupies y in [i, i+1]); the reported
    energy is invariant under this flip (mirror + rigid-shift symmetry of the clamped
    top/bottom BCs), so it only matters for displacement-field comparisons.
    """
    img = np.asarray(bitmap)
    if img.ndim != 2:
        raise ValueError(f"bitmap must be 2D; got shape {img.shape}")
    img = np.rint(img).astype(int)
    if flip:
        img = np.flipud(img)
    img, scale = refine_matl_img(img, 1.0, refine)

    nodes = GridNodes.from_matl_img(img, scale)
    elements = GridElements(nodes, type="CPE4")
    materials = [
        Material(elset, density=DENSITY,
                 response=NeoHookean(poisson=POISSON, youngs=pixel_youngs(elset.matl_code)))
        for elset in ElementSet.from_matl_img(img)
    ]

    # full-edge node sets INCLUDING corners (the stock "Y0"/"Y1" nsets are interior-only)
    y0 = NodeSet.from_slice("Y0ALL", (0, slice(None)), nodes)
    y1 = NodeSet.from_slice("Y1ALL", (-1, slice(None)), nodes)
    bcs = [
        BoundaryCondition(y0, Fixed([1, 2])),          # bottom: u = (0, 0)
        BoundaryCondition(y1, Fixed([1])),             # top: ux = 0 ...
        BoundaryCondition(y1, Prescribed([2], 0.0)),   # ... uy driven per step (baseline)
    ]
    model = Model(nodes=nodes, elements=elements, materials=materials,
                  bcs=bcs, nsets=(y0, y1))

    steps = [
        Step(subsections=[Static(), BoundaryCondition(y1, Prescribed([2], float(d)))],
             perturbation=False, nlgeom=True)
        for d in disp_vals
    ]
    heading = Heading("Mechanical-MNIST uniaxial extension")
    return Simulation(heading=heading, model=model, steps=steps)
