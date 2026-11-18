"""Grid scaling of a periodic free-lateral ``Static`` run on the FEniCS_test report's RVE.

The case is the grid-scaling report at https://github.com/AnlanChenA/FEniCS_test (commit 3115e2f):
a 400x400 pixel image of particles in a PMMA matrix with an interphase, bilinear J2 hardening in
the matrix and interphase, periodic boundaries, uniaxial x drive with the lateral strain free,
taken through the 13 strain points of the report's ABAQUS curve (one Static step each). The
report's own script predates the current spec API; this rebuilds the same model through it.
``--coarsen C`` downsamples the image by ``C`` as the report's ``COARSEN`` does.
``--material neohookean`` swaps every phase for a coupled neo-Hookean of the same small-strain
moduli under NLGEOM, which times the hyperelastic free-lateral path on the same microstructure.

Run from the fenicsx env against a checkout of the report, which is read as data only::

    git clone https://github.com/AnlanChenA/FEniCS_test /path/to/FEniCS_test
    /home/rjs80/miniconda3/envs/fenicsx/bin/python tools/bench_grid_scaling.py \
        /path/to/FEniCS_test --coarsen 4 [--material plastic|neohookean] \
        [--points abaqus|first:K|e1,e2,...] [--n-incr N] [--profile N] [--json OUT]

Prints the wall time of ``run``, the PETSc factorization and back-substitution counts and times,
and the stress at each point (against the ABAQUS curve for the plastic case). ``--profile N``
adds the N functions with the most own time; ``--json`` writes the numbers to a file. One thread
per process.

Results
-------
Plastic case, ABAQUS points, on hyperelastic at e99437f (native PETSc LU, nested free-lateral
root-find, finite-difference J2 tangent), one core of the shared 2x Xeon Gold 6148 host:

    grid      ndof      wall       factorizations   LU time    back-solves
    50x50     5,202     11.6 s     67               7.5 s      268
    100x100   20,402    92.9 s     74               74.1 s     280
    200x200   80,802    890 s      87               791 s      328
    400x400   321,602   7,610 s    100              7,096 s    364

The factorization count drifts up with resolution; the cost is the per-factorization time, which
grows about as ndof^1.5 (71 s at 400x400, peak memory 2.9 GB). The report's estimate of hours to
days came from 65d63c0, which ran 20 increments per free-lateral step.

Prototypes, applied as scratch patches that this script does not carry: MUMPS in place of native
LU; the free-lateral strains solved inside one bordered Newton (Schur complement on the same
factorization) in place of the nested root-find; and skipping a tangent refresh that changed by
less than 1e-10 relative, which stands in for an exact tangent. 100x100 rows ran concurrently, so
their times are indicative; the 400x400 row of all three ran alone.

    variant                         100x100 wall   factorizations   400x400 wall   factorizations
    baseline                        92.9 s         74               7,610 s        100
    MUMPS                           29.1 s         74               921 s          100
    nested + skip                   84.5 s         63
    bordered                        31.8 s         22
    bordered + skip                 26.9 s         17
    bordered + skip + MUMPS         11.7 s         17               338 s          27

At 400x400 the MUMPS stresses match the baseline to 7.7e-14 and the bordered ones to 1.7e-9, and
the stresses lie within +0.05% to +0.15% of ABAQUS at all 13 points (50x50: up to +1.16%). A
factorization takes 4.6 s under MUMPS against 71 s native at 400x400.
"""
from __future__ import annotations

# Pin BLAS/OpenMP to one thread before importing numpy/dolfinx (mirrors tools/bench_solvers.py).
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import cProfile
import json
import pathlib
import pstats
import time

import numpy as np
import scipy.io as sio

from microstructure_ve.api import (
    BoundaryCondition, Elastic, ElementSet, Fixed, GridElements, GridNodes, Heading, Material,
    Model, NeoHookean, PeriodicBoundaryConstraint, Plastic, Prescribed, Simulation, Static, Step,
)

SCALE = 0.0025                       # micron per pixel at full resolution
PARTICLE = dict(density=2.65e-15, youngs=70000.0, poisson=0.3)
MATRIX = dict(density=1.18e-15, youngs=3500.0, poisson=0.3)
TANGENT_MODULUS = 500.0              # MPa, the bilinear hardening slope
T_BASE = 293.0                       # K, the temperature the yield stress is read at
STRAIN_CAP = 0.15                    # the strain the bilinear table extends to
_EVENTS = ("MatLUFactorSym", "MatLUFactorNum", "MatSolve")


def bilinear_table(sigma_y0, youngs):
    """``(yield_stress, plastic_strain)`` of the report's bilinear hardening up to STRAIN_CAP."""
    sigma2 = sigma_y0 + TANGENT_MODULUS * (STRAIN_CAP - sigma_y0 / youngs)
    return [sigma_y0, sigma2], [0.0, STRAIN_CAP - sigma2 / youngs]


def load_report(root):
    """The report's interphase image, interphase levels, yield curve and ABAQUS curve."""
    root = pathlib.Path(root)
    img = sio.loadmat(root / "data" / "b=0.03_e=0.929_z=0_UniIP_IP1_20pix_intph.mat")
    yield_curve = sio.loadmat(root / "data" / "Temp_Yield_PMMA.mat")["Temp_sigmaY"]
    uni = np.loadtxt(root / "data" / "uni_level.txt")
    se = np.loadtxt(root / "S-Eabaqus.txt")
    n = len(se) // 2
    return (img["interphase_image"].astype(int), uni, yield_curve,
            {"stress": se[:n], "strain": se[n:]})


def build(img, uni, yield_curve, coarsen, strains, material):
    """The periodic free-lateral simulation through ``strains``; returns ``(sim, Lx)``."""
    scale = SCALE * coarsen
    img = img[::coarsen, ::coarsen]
    int_youngs = MATRIX["youngs"] * float(uni[0])

    def sigma_y(temperature):
        return float(np.interp(temperature, yield_curve[:, 0], yield_curve[:, 1]))

    nodes = GridNodes.from_matl_img(img, scale=scale)
    elements = GridElements(nodes, type="CPE4")
    matrix_el, interphase_el, particle_el = ElementSet.from_matl_img(img)
    phases = [(particle_el, PARTICLE["density"], PARTICLE["youngs"], None),
              (matrix_el, MATRIX["density"], MATRIX["youngs"], sigma_y(T_BASE)),
              (interphase_el, MATRIX["density"], int_youngs, sigma_y(T_BASE - float(uni[1])))]
    materials = []
    for elset, density, youngs, sy0 in phases:
        if material == "neohookean":
            response = NeoHookean(poisson=0.3, youngs=youngs)
        elif sy0 is None:
            response = Elastic(poisson=0.3, youngs=youngs)
        else:
            ys, ps = bilinear_table(sy0, youngs)
            response = Plastic(poisson=0.3, youngs=youngs, yield_stress=ys, plastic_strain=ps)
        materials.append(Material(elset, density=density, response=response))

    origin, x_corner, y_corner = (nodes.nsets[k] for k in ("X0Y0", "X1Y0", "X0Y1"))
    bcs = [
        PeriodicBoundaryConstraint(nodes=nodes),
        BoundaryCondition(origin, Fixed(dofs=[1, 2])),
        BoundaryCondition(x_corner, Fixed(dofs=[2])),
        BoundaryCondition(y_corner, Fixed(dofs=[1])),     # lateral normal left free
        BoundaryCondition(x_corner, Prescribed(dofs=[1], value=0.0)),
    ]
    model = Model(nodes=nodes, elements=elements, materials=materials, bcs=bcs)
    Lx = img.shape[1] * scale
    steps = [Step(subsections=[Static(), BoundaryCondition(
        x_corner, Prescribed(dofs=[1], value=float(e * Lx)))], nlgeom=material == "neohookean")
        for e in strains]
    return Simulation(heading=Heading("grid scaling"), model=model, steps=steps), Lx


def petsc_events():
    """``{event: [seconds, count]}`` of the factorization and back-substitution events."""
    from petsc4py import PETSc

    out = {}
    for name in _EVENTS:
        info = PETSc.Log.Event(name).getPerfInfo()
        out[name] = [info["time"], info["count"]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", help="checkout of github.com/AnlanChenA/FEniCS_test")
    ap.add_argument("--coarsen", type=int, default=8)
    ap.add_argument("--material", choices=("plastic", "neohookean"), default="plastic")
    ap.add_argument("--points", default="abaqus")
    ap.add_argument("--n-incr", type=int, default=None)
    ap.add_argument("--profile", type=int, default=0, metavar="N")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    img, uni, yield_curve, abaqus = load_report(a.report)
    if a.points == "abaqus":
        strains = abaqus["strain"]
    elif a.points.startswith("first:"):
        strains = abaqus["strain"][:int(a.points.split(":")[1])]
    else:
        strains = np.array([float(s) for s in a.points.split(",")])
    sim, Lx = build(img, uni, yield_curve, a.coarsen, strains, a.material)
    shape = img[::a.coarsen, ::a.coarsen].shape
    ndof = 2 * (shape[0] + 1) * (shape[1] + 1)

    from petsc4py import PETSc
    from microstructure_ve.backends.dolfinx import run

    PETSc.Log.begin()
    prof = cProfile.Profile() if a.profile else None
    t0 = time.perf_counter()
    if prof:
        prof.enable()
    rows = run(sim, n_incr=a.n_incr)
    if prof:
        prof.disable()
    wall = time.perf_counter() - t0

    stress, strain = rows[:, 1] / Lx, rows[:, 5] / Lx
    ev = petsc_events()
    res = {"grid": list(shape), "ndof": ndof, "material": a.material, "points": a.points,
           "n_incr": a.n_incr, "wall_s": wall, "petsc": ev, "strain": strain.tolist(),
           "stress": stress.tolist()}
    print(f"grid {shape[0]}x{shape[1]} ndof {ndof} {a.material} points {a.points}: "
          f"wall {wall:.1f} s")
    for name, (sec, count) in ev.items():
        print(f"  {name:16s} {count:6d} calls {sec:10.2f} s")
    if a.material == "plastic":
        ref = np.interp(strain, abaqus["strain"], abaqus["stress"])
        res["abaqus"] = ref.tolist()
        for e, s, r in zip(strain, stress, ref):
            print(f"  strain {e:.5f}  stress {s:9.3f}  ABAQUS {r:9.3f}  {100 * (s - r) / r:+6.2f}%")
    else:
        for e, s in zip(strain, stress):
            print(f"  strain {e:.5f}  nominal stress {s:9.3f}")
    if prof:
        pstats.Stats(prof).sort_stats("tottime").print_stats(a.profile)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
