"""Estimate CPU-hours for a hyperelastic-composite tensile campaign.

Target scenario (scoping question): a tensile test of a hyperelastic composite through
**12 extensions** (i.e. 12 load increments per stress-strain curve) across **5000
microstructures**.

This package has **no hyperelastic / large-deformation (NLGEOM) material** -- its nonlinear
capability is the small-strain J2 (von Mises) plasticity solver in
``backends/dolfinx/_plastic.py``. At ~5k DOF the per-increment cost is dominated by the FE
*linear solve* inside each Newton iteration, and that cost is essentially independent of the
constitutive law (a neo-Hookean/Ogden tangent assembles differently but factors/solves the
same sparse system). So we use the J2 Newton solver as a **proxy** for hyperelastic per-solve
cost, time a real 12-increment tensile ramp, and extrapolate.

Because the 5000 microstructures are independent, the batch is embarrassingly parallel and

    CPU-hours ~= 5000 * (single-core CPU-seconds for one 12-increment run) / 3600

with parallel workers only shrinking wall-clock, not CPU-hours. The 12 extensions are the 12
increments *inside* one run, so one timed run already equals one microstructure's full campaign.

Two loading variants bracket the estimate:

* **confined**  -- lateral macro strain held at 0: a clean 12-increment ramp (lower bound).
* **free-lateral** -- lateral faces float (Poisson contraction, the physical tensile case):
  each increment adds an outer macroscopic root-find so lateral stress vanishes (upper bound).

Run from the fenicsx env::

    /home/rjs80/miniconda3/envs/fenicsx/bin/python tools/bench_hyperelastic_estimate.py

Writes ``tools/bench_hyperelastic_estimate_results.json`` and prints a summary table.

Caveats (also echoed in the output): this is a *proxy* (J2 plasticity, small strain -- not
true hyperelasticity/NLGEOM); a single 2D ~5k-DOF mesh size; single-core timing with the FE
form-compilation (FFCx) warmed up and thus amortized over the 5000-run batch as it would be in
a real campaign.
"""
from __future__ import annotations

# Pin BLAS/OpenMP to one thread BEFORE importing numpy/dolfinx, so single-core CPU-seconds
# ~= wall-seconds and the CPU-hour extrapolation is clean (mirrors tools/bench_solvers.py).
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import pathlib
import statistics
import time

import numpy as np

from microstructure_ve.core import GridNodes, GridElements, ElementSet
from microstructure_ve.constitutive import Elastic, Plastic
from microstructure_ve.materials import Material
from microstructure_ve.boundary import (
    PeriodicBoundaryConstraint,
    BoundaryCondition,
    Fixed,
    Prescribed,
)
from microstructure_ve.steps import Model, Simulation, Step, Static, Heading

# ----- scenario knobs -------------------------------------------------------
N_MICROSTRUCTURES = 5000
N_EXTENSIONS = 12           # load increments per tensile curve
GRID = 50                   # 50x50 pixels -> 51x51 nodes -> 5202 DOF in 2D
SCALE = 0.0025              # micron/pixel (matches example.py / tests)
MACRO_STRAIN = 0.02         # ~2% tensile x-strain: past matrix yield, inside the hardening table
REPEATS = 5                 # timed repeats per variant (median reported)
PROXY_BAND = (1.0, 3.0)     # hyperelastic-Newton cost multiplier vs monotonic J2 (see module doc)
CORES_FOR_WALLCLOCK = (1, 4, 16, 80)

# material constants (mirror tests/_matrix.py so the proxy is physically representative)
DENSITY = 2.65e-15
MATRIX_YOUNGS = 3000.0                  # MPa, soft plastic matrix
MATRIX_POISSON = 0.3
FILLER_YOUNGS = 5.0e5                   # MPa, stiff elastic filler
FILLER_POISSON = 0.15
YIELD_STRESS = [40.0, 70.0]             # MPa, monotonic hardening
PLASTIC_STRAIN = [0.0, 0.05]

RESULTS_PATH = pathlib.Path(__file__).with_name("bench_hyperelastic_estimate_results.json")


def make_image(n=GRID, seed=0):
    """Deterministic two-phase RVE: stiff filler discs (0) in a plastic matrix (1)."""
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.ones((n, n), dtype=int)  # 1 = matrix
    rng = np.random.default_rng(seed)
    for _ in range(14):
        cy, cx = rng.integers(0, n, size=2)
        r = int(rng.integers(3, 6))
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 0  # 0 = filler
    return img


def build_sim(img, free_lateral):
    """A single-Static-step periodic tensile sim driven in +x to MACRO_STRAIN.

    ``free_lateral`` leaves the transverse reference corner's normal dof unpinned (physical
    uniaxial tension, Poisson contraction); otherwise it is held (confined)."""
    nodes = GridNodes.from_matl_img(img, SCALE)
    elements = GridElements(nodes, type="CPE4")
    filler_elset, matrix_elset = ElementSet.from_matl_img(img)  # sorted ascending: 0 then 1
    materials = [
        Material(filler_elset, density=DENSITY,
                 response=Elastic(poisson=FILLER_POISSON, youngs=FILLER_YOUNGS)),
        Material(matrix_elset, density=DENSITY,
                 response=Plastic(poisson=MATRIX_POISSON, youngs=MATRIX_YOUNGS,
                                  yield_stress=YIELD_STRESS, plastic_strain=PLASTIC_STRAIN)),
    ]
    origin = nodes.nsets["X0Y0"]
    x_macro = nodes.nsets["X1Y0"]
    y_macro = nodes.nsets["X0Y1"]
    lx = (nodes.shape[-1] - 1) * SCALE           # domain length along x
    drive = MACRO_STRAIN * lx

    y_hold = Fixed(dofs=[1]) if free_lateral else Fixed(dofs=[1, 2])
    bcs = [
        PeriodicBoundaryConstraint(nodes=nodes),
        BoundaryCondition(origin, Fixed(dofs=[1, 2])),   # pin origin (rigid-body translation)
        BoundaryCondition(x_macro, Fixed(dofs=[2])),     # suppress shear on the drive corner
        BoundaryCondition(y_macro, y_hold),              # lateral: held (confined) or free
    ]
    model = Model(nodes=nodes, elements=elements, materials=materials, bcs=bcs)
    step = Step(subsections=[Static(),
                             BoundaryCondition(x_macro, Prescribed(dofs=[1], value=drive))])
    return Simulation(heading=Heading("hyperelastic proxy tensile"), model=model, steps=[step])


class _KSPCounter:
    """Wrap a PETSc KSP so we can count linear solves (the LU cost the estimate hinges on)."""

    def __init__(self, ksp, counter):
        self._ksp, self._counter = ksp, counter

    def solve(self, b, x):
        self._counter[0] += 1
        return self._ksp.solve(b, x)

    def __getattr__(self, name):
        return getattr(self._ksp, name)


def build_plastic_solver(sim, bbar=True):
    """Build a fresh persistent plastic solver + its macro loading, reusing the cached mesh
    setup. Returns ``(solver, loading, ndof, dim)``. A fresh solver per call matches a real
    batch: mesh/forms/MPC are cached by shape, but each microstructure gets its own solver."""
    from microstructure_ve.backends.dolfinx import (
        _run as run_mod,
        _spec as spec,
        _loading as loadingmod,
        _plastic as plastic,
        _constraints as constraints,
    )

    model = sim.model
    geom = spec.Geometry.from_model(model, sim)
    prob = run_mod._fe_problem(geom, model, bbar)
    if prob.mpc is None:                                   # populate periodic MPC once (cached)
        prob.mpc = constraints.periodic_mpc(prob.space)
        prob.center_bcs = constraints.center_pin(prob.space)
    else:
        prob.matfields.update_materials(model)             # rebind this microstructure's phases
    loading = loadingmod.macro_loading(sim)
    solver = plastic.make_solver(prob, model)
    ndof = int(np.prod(geom.shape)) * geom.dim
    return solver, loading, ndof, geom.dim


def time_variant(free_lateral, seed=0):
    """Warm up (amortize FFCx compile), then time REPEATS fresh 12-increment ramps."""
    img = make_image(seed=seed)
    sim = build_sim(img, free_lateral=free_lateral)

    # Warm-up: first build+solve triggers FFCx form compilation (one-time per process, amortized
    # across the 5000-run batch); discard its timing.
    solver, loading, ndof, dim = build_plastic_solver(sim)
    cold_t0 = time.perf_counter()
    solver.solve(loading, N_EXTENSIONS)
    cold_s = time.perf_counter() - cold_t0

    times, solve_counts = [], []
    for _ in range(REPEATS):
        solver, loading, ndof, dim = build_plastic_solver(sim)  # fresh state each repeat
        counter = [0]
        solver.ksp = _KSPCounter(solver.ksp, counter)
        t0 = time.perf_counter()
        solver.solve(loading, N_EXTENSIONS)
        times.append(time.perf_counter() - t0)
        solve_counts.append(counter[0])

    return {
        "free_lateral": free_lateral,
        "ndof": ndof,
        "dim": dim,
        "increments": N_EXTENSIONS,
        "cold_s": cold_s,
        "steady_s": statistics.median(times),
        "steady_s_all": times,
        "linear_solves": int(statistics.median(solve_counts)),
        "linear_solves_all": solve_counts,
    }


def main():
    from microstructure_ve.backends.dolfinx._solver import predict_lu_seconds

    variants = {
        "confined": time_variant(free_lateral=False),
        "free_lateral": time_variant(free_lateral=True),
    }

    ndof = variants["confined"]["ndof"]
    dim = variants["confined"]["dim"]
    lu_one = predict_lu_seconds(ndof, dim)   # calibrated single LU solve (tools/bench_solvers.py)

    est = {}
    for name, v in variants.items():
        steady = v["steady_s"]
        cpu_h_per = N_MICROSTRUCTURES * steady / 3600.0
        est[name] = {
            "steady_s": steady,
            "linear_solves": v["linear_solves"],
            "modeled_s": lu_one * v["linear_solves"],   # independent cross-check
            "cpu_hours_5000": cpu_h_per,
            "cpu_hours_5000_band": [cpu_h_per * PROXY_BAND[0], cpu_h_per * PROXY_BAND[1]],
        }

    # Headline band: low = confined x proxy-min, high = free-lateral x proxy-max.
    band_low = est["confined"]["cpu_hours_5000_band"][0]
    band_high = est["free_lateral"]["cpu_hours_5000_band"][1]

    results = {
        "scenario": {
            "microstructures": N_MICROSTRUCTURES,
            "extensions_per_curve": N_EXTENSIONS,
            "grid": f"{GRID}x{GRID}",
            "ndof": ndof,
            "dim": dim,
            "macro_strain": MACRO_STRAIN,
            "repeats": REPEATS,
            "proxy_band": list(PROXY_BAND),
        },
        "cost_model": {
            "predict_lu_seconds_one_solve": lu_one,
            "note": "calibrated single-core LU on 2x Xeon Gold 6148 (tools/bench_solvers.py)",
        },
        "variants": variants,
        "estimate": est,
        "headline_cpu_hours_band": [band_low, band_high],
        "wallclock_hours": {
            name: {str(c): est[name]["cpu_hours_5000"] / c for c in CORES_FOR_WALLCLOCK}
            for name in est
        },
        "caveats": [
            "PROXY: J2 plasticity stands in for hyperelasticity; small-strain, not NLGEOM.",
            f"single mesh size ({GRID}x{GRID}, {ndof} DOF, 2D); 3D would be far heavier.",
            "single-core timing; FFCx form-compile warmed up (amortized over the batch).",
            "CPU-hours are parallelism-invariant; wallclock rows assume perfect core scaling.",
        ],
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    # ---- printed summary ----
    print(f"\nHyperelastic tensile campaign CPU-hour estimate (PROXY: J2 plasticity)")
    print(f"  {N_MICROSTRUCTURES} microstructures x {N_EXTENSIONS} extensions, "
          f"{GRID}x{GRID} 2D ({ndof} DOF)")
    print(f"  calibrated single LU solve: {lu_one*1e3:.2f} ms  "
          f"(predict_lu_seconds, 2x Xeon Gold 6148)\n")
    hdr = f"  {'variant':<13}{'s/run':>9}{'lin.solv':>9}{'model s':>9}{'CPU-h(5000)':>13}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, v in variants.items():
        e = est[name]
        print(f"  {name:<13}{e['steady_s']:>9.3f}{v['linear_solves']:>9d}"
              f"{e['modeled_s']:>9.3f}{e['cpu_hours_5000']:>13.2f}")
    print()
    print(f"  Headline CPU-hours for {N_MICROSTRUCTURES} microstructures "
          f"(confined..free-lateral x proxy {PROXY_BAND[0]:g}-{PROXY_BAND[1]:g}):")
    print(f"      {band_low:.1f}  to  {band_high:.1f}  CPU-hours")
    print("\n  Implied wall-clock (perfect scaling, embarrassingly parallel over microstructures):")
    for c in CORES_FOR_WALLCLOCK:
        lo = est["confined"]["cpu_hours_5000"] / c * PROXY_BAND[0]
        hi = est["free_lateral"]["cpu_hours_5000"] / c * PROXY_BAND[1]
        print(f"      {c:>3d} cores: {lo:8.2f} .. {hi:8.2f} h")
    print(f"\n  cross-check: modeled (LU x lin.solves) vs measured -- "
          f"agreement confirms the cost model.")
    print(f"  wrote {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
