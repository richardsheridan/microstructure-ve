"""Benchmark LU vs GMRES+ILU per-frequency solve time vs mesh size (periodic homogenization).

This is the reproducer behind the solver-kind crossover: the ``_solver._COST`` LU cost model
and the 2D-diverges / 3D-iterative-wins boundaries recorded in the design doc (§10) are fit to
its output. Run from anywhere with the fenicsx interpreter; it rewrites
``tools/bench_solvers_results.json`` (the committed calibration) beside this script.

Pinned to one BLAS thread (the serial cost-model regime). For each mesh size it times the
LU and the iterative solver BACK TO BACK, repeated over rounds, so contention hits both
equally; the paired ratio is robust even as absolute times drift with load. The timed unit
is one ``solve_one(f)`` = reassemble (refactor / rebuild preconditioner) + solve all active
modes -- the cost a frequency sweep pays repeatedly. LU is skipped above a predicted-time cap
so no single factorization dominates the run (that regime is exactly where iterative is for).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests._matrix + the package live under the repo root

from tests._matrix import matrix_simulation  # noqa: E402
from microstructure_ve.backends.dolfinx import _run as run  # noqa: E402
from microstructure_ve.backends.dolfinx._solver import predict_lu_seconds  # noqa: E402

RESULTS_JSON = HERE / "bench_solvers_results.json"
SIZES = [(2, n) for n in (48, 96, 144, 192, 240)] + [(3, n) for n in (8, 12, 16, 20)]
ROUNDS = 3
LU_CAP_S = 75.0   # skip LU whose predicted factor time exceeds this (keeps the run bounded)
FREQ = 1.0
CELL = dict(mode="uniaxial_x", traction="free", bc="periodic", dim=None)


def time_solve(sim, kind):
    """Build fresh (untimed) then time one solve_one(FREQ). Returns (seconds, status)."""
    run.clear_cache()
    try:
        solve_one, _ = run.build_solver(sim, solver=kind)
    except Exception as e:  # noqa: BLE001
        return None, f"build-fail:{type(e).__name__}"
    try:
        t0 = time.perf_counter()
        solve_one(FREQ)
        return time.perf_counter() - t0, "ok"
    except Exception as e:  # noqa: BLE001 (iterative may raise non-convergence)
        return time.perf_counter() - t0, f"fail:{type(e).__name__}"


def main():
    print(f"cores={os.cpu_count()} threads=1 rounds={ROUNDS} lu_cap={LU_CAP_S}s", flush=True)
    print(f"{'dim':>3} {'n':>4} {'ndof':>8} {'LU(s)':>9} {'iter(s)':>9} {'iter/LU':>8} "
          f"{'iter':>10} {'load1':>6}", flush=True)
    results = []
    for dim, n in SIZES:
        cell = dict(CELL); cell["dim"] = dim
        sim = matrix_simulation(test_type="viscoelastic", homogeneous=True, n=n, **cell)
        ndof = int(np.prod(sim.model.nodes.shape)) * dim
        lu_pred = predict_lu_seconds(ndof, dim)
        rec = {"dim": dim, "n": n, "ndof": ndof, "lu_pred": lu_pred,
               "lu": [], "iter": [], "iter_status": [], "load": []}
        for _ in range(ROUNDS):
            rec["load"].append(os.getloadavg()[0])
            if lu_pred <= LU_CAP_S:
                lt, _ls = time_solve(sim, "lu")
            else:
                lt, _ls = None, "skip:pred"
            it, istatus = time_solve(sim, "iterative")
            rec["lu"].append(lt)
            rec["iter"].append(it)
            rec["iter_status"].append(istatus)
        results.append(rec)
        lu_best = min([t for t in rec["lu"] if t is not None], default=None)
        it_ok = [t for t, s in zip(rec["iter"], rec["iter_status"]) if s == "ok"]
        it_best = min(it_ok, default=None)
        ratio = (it_best / lu_best) if (lu_best and it_best) else None
        istat = rec["iter_status"][-1]
        print(f"{dim:>3} {n:>4} {ndof:>8} "
              f"{(f'{lu_best:.3f}' if lu_best else ('skip' if lu_pred>LU_CAP_S else '-')):>9} "
              f"{(f'{it_best:.3f}' if it_best else 'DNC'):>9} "
              f"{(f'{ratio:.2f}x' if ratio else '-'):>8} {istat:>10} "
              f"{rec['load'][-1]:>6.0f}", flush=True)
    with open(RESULTS_JSON, "w") as fh:
        json.dump({"cores": os.cpu_count(), "rounds": ROUNDS, "lu_cap_s": LU_CAP_S,
                   "results": results}, fh, indent=2)
    print(f"wrote {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
