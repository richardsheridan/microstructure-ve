"""Benchmark the dolfinx backend's frequency-parallel strategies.

Compares three ways to parallelize the independent per-frequency solves:
  - ProcessPool   : dolfinx_backend.run(workers=N) (spawn); true parallelism.
  - ThreadPool    : threads each building their own solver; GIL + PETSc thread-unsafety.
  - *_NUM_THREADS : serial sweep with intra-solve BLAS threading (native LU barely threads).

Methodology (per design): the ProcessPool path is decomposed with time.perf_counter into
  spawn   - interpreter process creation + re-import of this/backend modules (measured once,
            parent-side, with a no-op child; assumed stable across workers),
  init    - the worker initializer: import dolfinx + build the solver (_build_solver),
  first   - the first solve_one(f) (carries FFCx/PETSc warmup),
  steady  - the mean of subsequent solves -- THE BENCHMARK TARGET.
Process startup (spawn+init+first) is a fixed per-worker cost; when comparing against the
thread approach (treated as instant-startup) it is AMORTIZED over the sweep length N:
  processpool per-freq(N, W) ~= (spawn + init + first) / N + steady / W
  threadpool  per-freq       ~= steady_thread / W_eff   (W_eff ~ 1 if GIL-bound)
The crossover sweep length N* (where processes beat threads) follows directly.

Each measurement runs in its OWN subprocess so *_NUM_THREADS can be set before import and a
PETSc/ThreadPool segfault is contained. Host is shared; run off-peak. Use --quick to smoke-test.

Usage:
    python bench_dolfinx.py --quick                # fast smoke (one small size)
    python bench_dolfinx.py                        # full sweep (run OFF-PEAK)
    python bench_dolfinx.py --measure ...          # internal: one isolated measurement
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

SIZES_2D = [64, 128, 256]
SIZES_3D = [16, 24, 32]
N_STEADY = 5          # solves averaged for the steady-state estimate
N_SWEEP = 30          # nominal sweep length for amortized comparison
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


# --------------------------------------------------------------------------- builders
def mk(n, dim):
    """Synthetic two-material RVE (stiff cube inclusion) of n^dim cells."""
    from microstructure_ve import (
        GridNodes, GridElements, ElementSet, Material,
        PeriodicBoundaryCondition, FixedBoundaryCondition, DisplacementBoundaryCondition,
        Dynamic, Step, Model, Simulation, Heading)
    img = np.zeros((n,) * dim, int)
    img[tuple(slice(n // 4, 3 * n // 4) for _ in range(dim))] = 1
    nodes = GridNodes.from_matl_img(img, 0.0025)
    es = ElementSet.from_matl_img(img)
    mats = [Material(es[0], density=1.18e-15, poisson=0.3, youngs=1e3),
            Material(es[1], density=1.18e-15, poisson=0.3, youngs=5e3)]
    ns = nodes.nsets
    et = "CPE4" if dim == 2 else "C3D8"
    o = {2: "X0Y0", 3: "X0Y0Z0"}[dim]
    xm = {2: "X1Y0", 3: "X1Y0Z0"}[dim]
    model = Model(nodes=nodes, elements=GridElements(nodes, type=et), materials=mats,
                  bcs=[PeriodicBoundaryCondition(nodes=nodes),
                       FixedBoundaryCondition(ns[o], dofs=list(range(1, dim + 1))),
                       DisplacementBoundaryCondition(ns[xm], 1, 1, 0.0)])
    step = Step(subsections=[Dynamic(f_initial=1, f_final=1, f_count=1, bias=1),
                             DisplacementBoundaryCondition(ns[xm], 1, 1, 0.005)], perturbation=True)
    return Simulation(heading=Heading("bench"), model=model, steps=[step])


def _ndof(n, dim):
    return (n + 1) ** dim * dim


# --------------------------------------------------------------------------- measurements
# (each runs in its own subprocess; prints a single JSON line on stdout)

def measure_phases(dim, n, lateral="free"):
    """init / first / steady, measured in-process with perf_counter."""
    import dolfinx_backend as db
    sim = mk(n, dim)
    fs = np.logspace(-2, 2, N_STEADY + 1)
    t0 = time.perf_counter()
    solve_one, _ = db._build_solver(sim, lateral, True)
    t_init = time.perf_counter() - t0
    t0 = time.perf_counter()
    solve_one(fs[0])
    t_first = time.perf_counter() - t0
    steady = []
    for f in fs[1:]:
        t0 = time.perf_counter()
        solve_one(f)
        steady.append(time.perf_counter() - t0)
    return {"phase": "phases", "dim": dim, "n": n, "ndof": _ndof(n, dim),
            "init": t_init, "first": t_first,
            "steady": float(np.mean(steady)), "steady_std": float(np.std(steady))}


def measure_spawn(dim, n):
    """Pure spawn cost: time to start a child that imports this+backend and returns."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    reps, ts = 3, []
    for _ in range(reps):
        t0 = time.perf_counter()
        p = ctx.Process(target=_noop)
        p.start()
        p.join()
        ts.append(time.perf_counter() - t0)
    return {"phase": "spawn", "dim": dim, "n": n, "spawn": float(np.median(ts))}


def measure_thread(dim, n, workers, nfreq, lateral="free"):
    """ThreadPool steady throughput: `workers` threads, each builds its own solver and
    solves a chunk. Captures GIL/PETSc behaviour (may not scale, may crash -> isolated)."""
    import dolfinx_backend as db
    from concurrent.futures import ThreadPoolExecutor
    sim = mk(n, dim)
    fs = np.logspace(-2, 2, nfreq)
    chunks = [fs[i::workers] for i in range(workers)]

    def work(chunk):
        solve_one, _ = db._build_solver(sim, lateral, True)
        for f in chunk:
            solve_one(f)
        return len(chunk)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, chunks))
    wall = time.perf_counter() - t0
    return {"phase": "thread", "dim": dim, "n": n, "workers": workers, "nfreq": nfreq,
            "wall": wall, "per_freq": wall / nfreq}


def measure_process(dim, n, workers, nfreq, lateral="free"):
    """Actual ProcessPool wall for a real sweep (validates the amortized model)."""
    import dolfinx_backend as db
    sim = mk(n, dim)
    fs = np.logspace(-2, 2, nfreq)
    t0 = time.perf_counter()
    db.run(sim, freqs=fs, lateral=lateral, workers=workers)
    wall = time.perf_counter() - t0
    return {"phase": "process", "dim": dim, "n": n, "workers": workers, "nfreq": nfreq,
            "wall": wall, "per_freq": wall / nfreq}


def _noop():
    return None


# --------------------------------------------------------------------------- orchestration
def _run_isolated(args_list, env_extra=None):
    """Run one measurement in a fresh subprocess; return its parsed JSON (or an error dict)."""
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    cmd = [PY, os.path.join(HERE, "bench_dolfinx.py"), "--measure"] + [str(a) for a in args_list]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "args": args_list}
    for line in reversed(out.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"error": "no-json", "args": args_list, "stderr": out.stderr[-400:]}


def amortized(phase, spawn, W, N):
    """ProcessPool per-frequency cost with startup amortized over N frequencies, W workers."""
    startup = spawn + phase["init"] + phase["first"]
    return startup / N + phase["steady"] / W


def crossover_N(phase, spawn, W, steady_thread):
    """Smallest sweep length N where ProcessPool(W) beats the thread approach."""
    startup = spawn + phase["init"] + phase["first"]
    gain = steady_thread - phase["steady"] / W  # per-freq advantage of processes once warm
    if gain <= 0:
        return float("inf")
    return startup / gain


def full_sweep(quick=False):
    # be a good neighbour on the shared host
    try:
        os.nice(19)
    except OSError:
        pass
    cores = sorted(os.sched_getaffinity(0))
    pin = cores[: (4 if quick else min(16, len(cores)))]
    try:
        os.sched_setaffinity(0, set(pin))
    except OSError:
        pass

    plan = [(2, SIZES_2D[0])] if quick else [(2, n) for n in SIZES_2D] + [(3, n) for n in SIZES_3D]
    worker_sweep = [1, 2, 4] if quick else [1, 2, 4, 8, 16]
    load0 = os.getloadavg()
    print(f"# host: {len(cores)} cores affinity, pinned to {len(pin)} | load {load0} | nice 19")
    print(f"# {'dim/n':>8} {'ndof':>8} {'spawn':>7} {'init':>7} {'first':>7} {'steady':>8} "
          f"{'thr/f':>8} {'P@4/f':>8} {'P@8/f':>8} {'N*vs4':>7}")

    results = []
    for dim, n in plan:
        ph = _run_isolated(["phases", dim, n])
        sp = _run_isolated(["spawn", dim, n])
        th = _run_isolated(["thread", dim, n, 4, 8])
        if "error" in ph or "error" in sp:
            print(f"  {dim}D/{n}: ERROR {ph.get('error') or sp.get('error')}")
            continue
        steady_thread = th.get("per_freq", float("nan"))
        row = {**ph, "spawn": sp["spawn"], "thread_per_freq": steady_thread,
               "amort": {W: amortized(ph, sp["spawn"], W, N_SWEEP) for W in worker_sweep},
               "Nstar_vs4": crossover_N(ph, sp["spawn"], 4, steady_thread)}
        # optional: actually run a real ProcessPool sweep to validate the model
        if not quick:
            row["process_measured"] = {W: _run_isolated(["process", dim, n, W, 8]).get("per_freq")
                                       for W in (2, 4)}
        results.append(row)
        print(f"  {dim}D/{n:<5d} {ph['ndof']:>8d} {sp['spawn']:>7.2f} {ph['init']:>7.2f} "
              f"{ph['first']:>7.3f} {ph['steady']:>8.3f} {steady_thread:>8.3f} "
              f"{row['amort'].get(4, float('nan')):>8.3f} {row['amort'].get(8, float('nan')):>8.3f} "
              f"{row['Nstar_vs4']:>7.1f}")

    out = {"host_cores": len(cores), "pinned": len(pin), "load_before": load0,
           "load_after": os.getloadavg(), "n_sweep": N_SWEEP, "results": results}
    with open(os.path.join(HERE, "bench_dolfinx_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"# wrote bench_dolfinx_results.json | load after {out['load_after']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="fast smoke (one small size)")
    p.add_argument("--measure", nargs="+", help="internal: <phase> <dim> <n> [workers nfreq]")
    a = p.parse_args()
    if a.measure:
        phase = a.measure[0]
        rest = [int(x) for x in a.measure[1:]]
        fn = {"phases": measure_phases, "spawn": measure_spawn,
              "thread": measure_thread, "process": measure_process}[phase]
        print(json.dumps(fn(*rest)))
    else:
        full_sweep(quick=a.quick)


if __name__ == "__main__":
    main()
