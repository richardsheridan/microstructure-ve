"""Benchmark LU vs GMRES+ILU per-frequency solve time vs mesh size (periodic homogenization).

This is the reproducer behind the solver-kind crossover: the ``_solver._COST`` LU cost model and the
2D-diverges / 3D-iterative-wins boundaries recorded in the design doc (§10) are fit to its output.
Run from anywhere with the fenicsx interpreter; it prints the table below to stdout and writes no
file.

Pinned to one BLAS thread (the serial cost-model regime). For each mesh size it times the
LU and the iterative solver BACK TO BACK, repeated over rounds, so contention hits both
equally; the paired ratio is robust even as absolute times drift with load. The timed unit
is one ``solve_one(f)`` = reassemble (refactor / rebuild preconditioner) + solve all active
modes -- the cost a frequency sweep pays repeatedly. LU is skipped above a predicted-time cap
so no single factorization dominates the run (that regime is exactly where iterative is for).

Results
-------
Calibration run: 80 logical cores, host load average 27 to 30, 3 rounds, LU cap 75 s, uniaxial_x
free-traction periodic viscoelastic homogeneous cell. Best of the rounds; the rounds agreed within a
few percent apart from the first 2D iterative rows. ``pred`` is ``predict_lu_seconds`` at the time
of the run (the fit to these LU times).

    dim  n    ndof     LU s    pred s   iter s   iter/LU   iterative status
    2    48     4802    0.151    0.132    0.247    1.6x     ok
    2    96    18818    1.297    1.027    3.172    2.4x     ok
    2    144   42050    4.380    3.432   20.28     4.6x     ok
    2    192   74498   10.63     8.093   38.09     3.6x     ok
    2    240  116162   20.03    15.76    44.14     -        fail:RuntimeError
    3    8      2187    0.729    0.918    0.280    0.38x    ok
    3    12     6591    9.70     8.341    1.106    0.11x    ok
    3    16    14739   50.02    41.71     3.078    0.06x    ok
    3    20    27783   skipped  148.2     6.878    -        ok (LU predicted over the cap)

Reading: in 2D the LU wins at every size and the iterative solver fails to converge at the largest;
in 3D the iterative solver wins from the smallest size. The ``iter/LU`` column is the ratio of the
best iterative round to the best LU round.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests._matrix + the package live under the repo root

from tests._matrix import matrix_simulation  # noqa: E402
from microstructure_ve.backends.dolfinx import _run as run  # noqa: E402
from microstructure_ve.backends.dolfinx._solver import predict_lu_seconds  # noqa: E402

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


if __name__ == "__main__":
    main()
