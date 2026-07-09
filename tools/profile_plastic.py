"""Profiling harness for the DOLFINx J2-plasticity solver (_PlasticSolver).

Layers:
  1. Section timers (monkeypatch): return_map, eps_eval, assemble_vector,
     assemble_matrix, ksp_solve -- accumulated wall-clock per section.
  2. Convergence log: per-_newton-call ksp-solve counts + residual norms
     (b.norm() captured inside the ksp proxy before each linear solve).
  3. Optional cProfile (--cprofile).

CLI::

    profile_plastic.py [--quick] [--variant confined|free_lateral|both]
                       [--n-incr N] [--grid N] [--cprofile]

Default (full mode): grid=50, n_incr=12 (confined) / 20 (free_lateral).
--quick: n_incr=4 both variants, grid=50.

Writes tools/profile_plastic_results.json.
"""
from __future__ import annotations

# Pin BLAS/OpenMP to one thread BEFORE importing numpy/dolfinx (mirrors bench_hyperelastic_estimate.py).
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import cProfile
import json
import math
import pathlib
import pstats
import sys
import time
import types

# Ensure tools/ is on sys.path so we can import bench_hyperelastic_estimate.
_TOOLS_DIR = pathlib.Path(__file__).parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# Also ensure the src/ package is importable (editable install may already cover it).
_SRC_DIR = _TOOLS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# bench_hyperelastic_estimate.py is guarded by `if __name__ == "__main__"` so importing
# it will NOT trigger its main().
from bench_hyperelastic_estimate import make_image, build_sim, build_plastic_solver, _KSPCounter  # noqa: E402

import numpy as np  # noqa: E402  (imported after thread-env pins)

RESULTS_PATH = pathlib.Path(__file__).with_name("profile_plastic_results.json")

# Default n_incr per variant (mirrors _run.py hard-codes)
_DEFAULT_N_INCR = {"confined": 12, "free_lateral": 20}
_QUICK_N_INCR = 4
_DEFAULT_GRID = 50


# ---------------------------------------------------------------------------
# Layer 1 + 2 helpers — timing/counting proxies
# ---------------------------------------------------------------------------

class _TimedKSP:
    """KSP proxy that times each solve, counts calls, and records b.norm() (residual)
    before delegating — giving Layer 1 timing AND Layer 2 residual sequences.

    newton_calls: list of lists; each inner list is residuals for one _newton call.
    _current_newton_residuals: the inner list being filled for the current _newton call.
    """

    def __init__(self, ksp, section_timers, newton_calls):
        self._ksp = ksp
        self._timers = section_timers   # shared dict: name -> [total_s, call_count]
        self._newton_calls = newton_calls
        self._current_newton_residuals = None   # set by the _newton wrapper

    def solve(self, b, x):
        # Record residual norm BEFORE the linear solve (= Newton residual at this iteration)
        rnorm = b.norm()
        if self._current_newton_residuals is not None:
            self._current_newton_residuals.append(float(rnorm))

        t0 = time.perf_counter()
        result = self._ksp.solve(b, x)
        elapsed = time.perf_counter() - t0

        entry = self._timers.setdefault("ksp_solve", [0.0, 0])
        entry[0] += elapsed
        entry[1] += 1
        return result

    def __getattr__(self, name):
        return getattr(self._ksp, name)


def _make_newton_wrapper(solver, ksp_proxy, newton_calls_list, newton_meta_list):
    """Return a wrapper for solver._newton that:
      - marks call boundaries on the ksp proxy (so residuals are grouped per _newton call)
      - records call count, duration, and ksp-solve count delta per call
    """
    original_newton = solver._newton  # bound method

    def _wrapped_newton(E_voigt):
        # Start a new per-call residual list
        this_call_residuals = []
        ksp_proxy._current_newton_residuals = this_call_residuals

        ksp_before = solver._timers["ksp_solve"][1] if "ksp_solve" in solver._timers else 0
        t0 = time.perf_counter()
        result = original_newton(E_voigt)
        elapsed = time.perf_counter() - t0
        ksp_after = solver._timers["ksp_solve"][1] if "ksp_solve" in solver._timers else 0

        ksp_proxy._current_newton_residuals = None
        newton_calls_list.append(this_call_residuals)
        newton_meta_list.append({
            "duration_s": elapsed,
            "ksp_solves": ksp_after - ksp_before,
            "residuals": this_call_residuals,
        })
        return result

    return _wrapped_newton


# ---------------------------------------------------------------------------
# Core profiling routine
# ---------------------------------------------------------------------------

def profile_variant(variant_name, n_incr, grid, use_cprofile):
    """Profile one variant. Returns the results dict for that variant."""
    free_lateral = (variant_name == "free_lateral")
    img = make_image(n=grid)
    sim = build_sim(img, free_lateral=free_lateral)

    # -- Warm-up: trigger FFCx form compilation (amortized over a real batch)
    print(f"  [{variant_name}] warming up FFCx compilation (n_incr=1)...")
    warmup_solver, warmup_loading, ndof, dim = build_plastic_solver(sim)
    warmup_solver.solve(warmup_loading, 1)
    print(f"  [{variant_name}] warm-up done. Building instrumented solver...")

    # -- Build a fresh instrumented solver
    solver, loading, ndof, dim = build_plastic_solver(sim)

    # Attach a shared timers dict to the solver so the _newton wrapper can read ksp counts.
    solver._timers = {}
    timers = solver._timers  # alias for brevity

    # Newton-call lists
    newton_calls_list = []   # list of per-call residual lists
    newton_meta_list = []    # list of per-call metadata dicts

    # -- Install Layer 2+1 ksp proxy
    ksp_proxy = _TimedKSP(solver.ksp, timers, newton_calls_list)
    solver.ksp = ksp_proxy

    # -- Wrap _newton (Layer 2 bookkeeping)
    wrapped_newton = _make_newton_wrapper(solver, ksp_proxy, newton_calls_list, newton_meta_list)
    solver._newton = wrapped_newton

    # -- Layer 1: monkeypatch module-level functions on dolfinx_mpc and _plastic

    # Import the _plastic module to patch its references
    import microstructure_ve.backends.dolfinx._plastic as _plastic_mod
    import dolfinx_mpc as _mpc_mod

    # Save originals
    _orig_return_map = _plastic_mod._return_map
    _orig_assemble_vector = _mpc_mod.assemble_vector
    _orig_assemble_matrix = _mpc_mod.assemble_matrix
    _orig_apply_lifting = _mpc_mod.apply_lifting

    def _make_timer_wrapper(fn, section_name):
        def _timed(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            entry = timers.setdefault(section_name, [0.0, 0])
            entry[0] += elapsed
            entry[1] += 1
            return result
        return _timed

    # Wrap eps_expr.eval (instance attribute) — stored on the object itself
    _orig_eps_eval = solver.eps_expr.eval

    def _timed_eps_eval(*args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_eps_eval(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        entry = timers.setdefault("eps_eval", [0.0, 0])
        entry[0] += elapsed
        entry[1] += 1
        return result

    solver.eps_expr.eval = _timed_eps_eval

    try:
        # Patch module attributes (affects _plastic._newton which reads them at call time)
        _plastic_mod._return_map = _make_timer_wrapper(_orig_return_map, "return_map")
        _mpc_mod.assemble_vector = _make_timer_wrapper(_orig_assemble_vector, "assemble_vector")
        _mpc_mod.assemble_matrix = _make_timer_wrapper(_orig_assemble_matrix, "assemble_matrix")
        _mpc_mod.apply_lifting = _make_timer_wrapper(_orig_apply_lifting, "apply_lifting")

        # -- Run the timed solve
        print(f"  [{variant_name}] running solve (n_incr={n_incr})...")
        wall_t0 = time.perf_counter()

        if use_cprofile:
            pr = cProfile.Profile()
            pr.enable()
            solver.solve(loading, n_incr)
            pr.disable()
        else:
            solver.solve(loading, n_incr)

        wall_s = time.perf_counter() - wall_t0
        print(f"  [{variant_name}] done. wall={wall_s:.2f}s")

    finally:
        # Restore all patched attributes
        _plastic_mod._return_map = _orig_return_map
        _mpc_mod.assemble_vector = _orig_assemble_vector
        _mpc_mod.assemble_matrix = _orig_assemble_matrix
        _mpc_mod.apply_lifting = _orig_apply_lifting
        solver.eps_expr.eval = _orig_eps_eval

    # -- cProfile output
    cprofile_rows = None
    if use_cprofile:
        ps = pstats.Stats(pr)
        ps.sort_stats("cumulative")
        print(f"\n  [{variant_name}] cProfile top 40 by cumulative time:")
        ps.print_stats(40)

        # Extract top 100 rows for JSON
        stats = ps.stats  # {(file,line,name): (cc, nc, tt, ct, callers)}
        rows = []
        for (f, lineno, name), (cc, nc, tt, ct, _callers) in stats.items():
            rows.append({"func": f"{f}:{lineno}({name})", "ncalls": nc,
                         "tottime": tt, "cumtime": ct})
        rows.sort(key=lambda r: -r["cumtime"])
        cprofile_rows = rows[:100]

    # -- Build section summary
    section_names = ["return_map", "eps_eval", "assemble_vector", "assemble_matrix",
                     "apply_lifting", "ksp_solve"]
    sections = {}
    for name in section_names:
        if name in timers:
            total_s, count = timers[name]
            sections[name] = {"total_s": total_s, "call_count": count,
                               "pct_wall": 100.0 * total_s / wall_s if wall_s > 0 else 0.0}
        else:
            sections[name] = {"total_s": 0.0, "call_count": 0, "pct_wall": 0.0}

    section_sum = sum(v["total_s"] for v in sections.values())
    other_s = wall_s - section_sum

    ksp_count = timers.get("ksp_solve", [0, 0])[1]
    newton_count = len(newton_meta_list)
    mean_iters = (sum(m["ksp_solves"] for m in newton_meta_list) / newton_count
                  if newton_count > 0 else 0.0)

    # -- Residual decay summary (first 3 + last _newton calls)
    def _resid_summary(residuals):
        if not residuals:
            return {"residuals": [], "geom_mean_ratio": None}
        ratios = [residuals[i + 1] / residuals[i]
                  for i in range(len(residuals) - 1)
                  if residuals[i] > 0]
        if ratios:
            log_ratios = [math.log(r) for r in ratios if r > 0]
            gm = math.exp(sum(log_ratios) / len(log_ratios)) if log_ratios else None
        else:
            gm = None
        return {"residuals": residuals, "geom_mean_ratio": gm}

    highlight_indices = list(range(min(3, newton_count)))
    if newton_count > 3:
        highlight_indices.append(newton_count - 1)
    resid_highlights = {
        str(i): _resid_summary(newton_meta_list[i]["residuals"])
        for i in highlight_indices
    }

    result = {
        "variant": variant_name,
        "n_incr": n_incr,
        "grid": grid,
        "ndof": ndof,
        "dim": dim,
        "wall_s": wall_s,
        "sections": sections,
        "section_sum_s": section_sum,
        "other_s": other_s,
        "other_pct_wall": 100.0 * other_s / wall_s if wall_s > 0 else 0.0,
        "ksp_solve_count": ksp_count,
        "newton_call_count": newton_count,
        "mean_newton_iters": mean_iters,
        "newton_meta": newton_meta_list,
        "resid_highlights": resid_highlights,
    }
    if cprofile_rows is not None:
        result["cprofile_top100"] = cprofile_rows
    return result


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(variant_name, r):
    print()
    print(f"=== {variant_name.upper()} (grid={r['grid']}, n_incr={r['n_incr']}, "
          f"ndof={r['ndof']}) ===")
    print(f"  wall time : {r['wall_s']:.3f} s")
    print()
    print(f"  {'section':<20} {'total_s':>9} {'%wall':>7} {'calls':>8}")
    print("  " + "-" * 46)
    for name, sv in r["sections"].items():
        print(f"  {name:<20} {sv['total_s']:>9.3f} {sv['pct_wall']:>6.1f}% {sv['call_count']:>8d}")
    print(f"  {'[sum]':<20} {r['section_sum_s']:>9.3f} "
          f"{100*r['section_sum_s']/r['wall_s']:>6.1f}%")
    print(f"  {'other':<20} {r['other_s']:>9.3f} {r['other_pct_wall']:>6.1f}%")
    print()
    print(f"  ksp.solve calls    : {r['ksp_solve_count']}")
    print(f"  _newton calls      : {r['newton_call_count']}")
    print(f"  mean Newton iters  : {r['mean_newton_iters']:.2f}")
    print()
    print("  Residual decay (first 3 + last _newton calls):")
    for idx_str, rd in r["resid_highlights"].items():
        resids = rd["residuals"]
        gm = rd["geom_mean_ratio"]
        resid_str = ", ".join(f"{v:.3e}" for v in resids) if resids else "(none)"
        gm_str = f"{gm:.4f}" if gm is not None else "N/A"
        print(f"    _newton[{idx_str}]: {resid_str}  (geom-mean ratio={gm_str})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile the DOLFINx J2-plasticity solver")
    parser.add_argument("--quick", action="store_true",
                        help=f"quick mode: n_incr={_QUICK_N_INCR} both variants, grid={_DEFAULT_GRID}")
    parser.add_argument("--variant", choices=["confined", "free_lateral", "both"],
                        default="both", help="which variant(s) to run (default: both)")
    parser.add_argument("--n-incr", type=int, default=None,
                        help="override n_incr for all variants")
    parser.add_argument("--grid", type=int, default=_DEFAULT_GRID,
                        help=f"grid size NxN (default: {_DEFAULT_GRID})")
    parser.add_argument("--cprofile", action="store_true",
                        help="run under cProfile and print top 40 by cumulative time")
    args = parser.parse_args()

    # Determine which variants to run
    if args.variant == "both":
        variants_to_run = ["confined", "free_lateral"]
    else:
        variants_to_run = [args.variant]

    # Determine n_incr per variant
    def get_n_incr(variant_name):
        if args.n_incr is not None:
            return args.n_incr
        if args.quick:
            return _QUICK_N_INCR
        return _DEFAULT_N_INCR[variant_name]

    grid = args.grid
    meta = {
        "argv": sys.argv,
        "grid": grid,
        "quick": args.quick,
        "cprofile": args.cprofile,
        "run_start": time.time(),
    }

    print(f"profile_plastic.py  grid={grid}  quick={args.quick}  "
          f"variants={variants_to_run}  cprofile={args.cprofile}")

    variant_results = {}
    for vname in variants_to_run:
        n_incr = get_n_incr(vname)
        print(f"\n--- profiling variant: {vname} (n_incr={n_incr}) ---")
        r = profile_variant(vname, n_incr, grid, args.cprofile)
        variant_results[vname] = r
        print_table(vname, r)

    meta["run_end"] = time.time()

    # Write JSON results
    # newton_meta lists residuals which may be large; keep them but convert to plain lists.
    def _serialise(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    results = {"meta": meta, "variants": variant_results}
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=_serialise))
    print(f"\nwrote {RESULTS_PATH}")

    # -- Quick sanity assertions (non-fatal warnings)
    for vname, r in variant_results.items():
        wall = r["wall_s"]
        s_sum = r["section_sum_s"]
        ksp = r["ksp_solve_count"]
        newton = r["newton_call_count"]
        pct = 100 * s_sum / wall if wall > 0 else 0.0
        issues = []
        if s_sum > wall:
            issues.append(f"section_sum ({s_sum:.3f}s) > wall ({wall:.3f}s) — clock overlap?")
        if pct < 60:
            issues.append(f"sections account for only {pct:.1f}% of wall (expected >60%)")
        if ksp == 0:
            issues.append("ksp.solve was never called (proxy may not be wired)")
        if newton == 0:
            issues.append("_newton was never called (wrapper may not be wired)")
        # Check residuals non-empty and decreasing for at least some calls
        all_resids = [m["residuals"] for m in r["newton_meta"] if m["residuals"]]
        if not all_resids:
            issues.append("no residuals captured (ksp proxy may not be recording b.norm())")
        else:
            # Check at least one sequence is (weakly) decreasing
            has_dec = any(
                all(seq[i] >= seq[i+1] for i in range(len(seq)-1))
                for seq in all_resids if len(seq) > 1
            )
            if not has_dec:
                issues.append("no residual sequence is non-increasing (unexpected)")
        if issues:
            print(f"\n[WARNING] {vname}:")
            for iss in issues:
                print(f"  - {iss}")
        else:
            print(f"\n[OK] {vname}: all sanity checks passed "
                  f"(sections={pct:.1f}% wall, ksp={ksp}, newton={newton}, "
                  f"residuals captured)")


if __name__ == "__main__":
    main()
