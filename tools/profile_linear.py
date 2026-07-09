"""Profile the DOLFINx linear (viscoelastic) solve paths: periodic homogenize and standard.

For a frequency sweep of F frequencies, times each named section and reports per-section
cost, call counts, and % of wall time for both paths.

Sections instrumented:
  - set_moduli         : material field refill (both paths)
  - mpc_assemble_matrix: dolfinx_mpc.assemble_matrix (periodic path)
  - std_assemble_A_bc  : fempetsc.assemble_matrix for A_bc (standard path)
  - mpc_assemble_vector: dolfinx_mpc.assemble_vector (periodic path, per mode)
  - ksp_solve          : ksp.solve (both paths -- LU back-sub for periodic, LU factor+solve for standard)
  - sbar_assemble      : fem.assemble_scalar stress-averaging forms (periodic path)
  - react_assemble     : action-form reaction vector assembly (standard path)

Usage::

    /home/rjs80/miniconda3/envs/fenicsx/bin/python tools/profile_linear.py [--n N] [--freqs F]

Default: n=50 (50x50 grid), F=20 frequencies, 2D two-phase viscoelastic, uniaxial_x free lateral.
Writes tools/profile_linear_results.json.
"""
from __future__ import annotations

# Pin BLAS/OpenMP to one thread BEFORE importing numpy/dolfinx.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pathlib
import sys
import time

# Ensure repo root is on sys.path so tests._matrix and the package are importable.
_TOOLS_DIR = pathlib.Path(__file__).parent
_REPO_ROOT = _TOOLS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np  # noqa: E402

RESULTS_PATH = pathlib.Path(__file__).with_name("profile_linear_results.json")

_DEFAULT_N = 50
_DEFAULT_FREQS = 20


# ---------------------------------------------------------------------------
# Timer helpers
# ---------------------------------------------------------------------------

def _make_timer(timers, section_name):
    """Return a wrapper factory that accumulates (total_s, call_count) in timers[section_name]."""
    def _wrap(fn):
        def _timed(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            entry = timers.setdefault(section_name, [0.0, 0])
            entry[0] += elapsed
            entry[1] += 1
            return result
        return _timed
    return _wrap


# ---------------------------------------------------------------------------
# Periodic / homogenize path
# ---------------------------------------------------------------------------

def profile_periodic(n, n_freqs):
    """Profile the periodic homogenize path on an n×n two-phase viscoelastic grid.

    Returns dict with timing results.
    """
    from tests._matrix import matrix_simulation, MATRIX_FREQS
    from microstructure_ve.backends.dolfinx import _run as run
    import microstructure_ve.backends.dolfinx._assembly as _assembly_mod
    import microstructure_ve.backends.dolfinx._homogenize as _homogenize_mod
    import dolfinx_mpc as _mpc_mod
    from dolfinx import fem

    sim = matrix_simulation(
        test_type="viscoelastic",
        mode="uniaxial_x",
        traction="free",
        bc="periodic",
        dim=2,
        n=n,
        homogeneous=False,
    )

    # Override frequency sweep to exactly n_freqs log-spaced frequencies
    freqs = np.logspace(-2.0, 2.0, n_freqs)

    # -- Warm-up: trigger FFCx compilation (untimed)
    print(f"  [periodic] warming up FFCx compilation (n={n})...", flush=True)
    run.clear_cache()
    solve_one_warm, _ = run.build_solver(sim, solver="lu")
    solve_one_warm(float(freqs[0]))
    print(f"  [periodic] warm-up done. Building instrumented solver...", flush=True)

    # -- Build a fresh solver with a clean cache
    run.clear_cache()

    timers = {}

    # -- Patch MaterialFields.set_moduli on the class (affects the instance used inside)
    import microstructure_ve.backends.dolfinx._assembly as _asmbl
    _orig_set_moduli = _asmbl.MaterialFields.set_moduli

    def _timed_set_moduli(self, f):
        t0 = time.perf_counter()
        result = _orig_set_moduli(self, f)
        elapsed = time.perf_counter() - t0
        entry = timers.setdefault("set_moduli", [0.0, 0])
        entry[0] += elapsed
        entry[1] += 1
        return result

    # -- Patch dolfinx_mpc.assemble_matrix
    _orig_mpc_asm_matrix = _mpc_mod.assemble_matrix

    def _timed_mpc_asm_matrix(*args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_mpc_asm_matrix(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        entry = timers.setdefault("mpc_assemble_matrix", [0.0, 0])
        entry[0] += elapsed
        entry[1] += 1
        return result

    # -- Patch dolfinx_mpc.assemble_vector
    _orig_mpc_asm_vector = _mpc_mod.assemble_vector

    def _timed_mpc_asm_vector(*args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_mpc_asm_vector(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        entry = timers.setdefault("mpc_assemble_vector", [0.0, 0])
        entry[0] += elapsed
        entry[1] += 1
        return result

    # -- Patch fem.assemble_scalar (for sbar stress averaging)
    _orig_fem_asm_scalar = fem.assemble_scalar

    def _timed_fem_asm_scalar(*args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_fem_asm_scalar(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        entry = timers.setdefault("sbar_assemble", [0.0, 0])
        entry[0] += elapsed
        entry[1] += 1
        return result

    # Apply patches
    _asmbl.MaterialFields.set_moduli = _timed_set_moduli
    _mpc_mod.assemble_matrix = _timed_mpc_asm_matrix
    _mpc_mod.assemble_vector = _timed_mpc_asm_vector

    # fem.assemble_scalar is imported in _homogenize; patch it there
    import microstructure_ve.backends.dolfinx._homogenize as _hom_mod
    _orig_hom_fem_asm_scalar = _hom_mod.fem.assemble_scalar

    # fem is the dolfinx.fem module; we need to patch the name as seen by _homogenize
    # _homogenize does: from dolfinx import fem; fem.assemble_scalar(...)
    # We patch the module attribute on the fem object inside _homogenize's namespace
    _hom_mod.fem.assemble_scalar = _timed_fem_asm_scalar

    # Import _solver_mod before the try so the finally block can always restore it.
    import microstructure_ve.backends.dolfinx._solver as _solver_mod
    _orig_lu_solve = _solver_mod.LuSolver.solve

    try:
        solve_one, _ = run.build_solver(sim, solver="lu")

        # -- Patch LuSolver.solve (the per-mode solve: assembles RHS + calls ksp.solve).
        # PETSc KSP.solve is a C-extension slot and is read-only, so we wrap the Python
        # method on the LuSolver class instead (affects all instances, including the
        # LuSolver built during build_solver).
        prob_key = next(iter(run._FE_CACHE))
        prob = run._FE_CACHE[prob_key]

        def _timed_lu_solve(self, j):
            t0 = time.perf_counter()
            result = _orig_lu_solve(self, j)
            elapsed = time.perf_counter() - t0
            entry = timers.setdefault("ksp_solve", [0.0, 0])
            entry[0] += elapsed
            entry[1] += 1
            return result

        _solver_mod.LuSolver.solve = _timed_lu_solve

        print(f"  [periodic] running sweep ({n_freqs} freqs)...", flush=True)
        wall_t0 = time.perf_counter()
        for f in freqs:
            solve_one(float(f))
        wall_s = time.perf_counter() - wall_t0
        print(f"  [periodic] done. wall={wall_s:.2f}s", flush=True)

    finally:
        _asmbl.MaterialFields.set_moduli = _orig_set_moduli
        _mpc_mod.assemble_matrix = _orig_mpc_asm_matrix
        _mpc_mod.assemble_vector = _orig_mpc_asm_vector
        _hom_mod.fem.assemble_scalar = _orig_hom_fem_asm_scalar
        _solver_mod.LuSolver.solve = _orig_lu_solve

    # Compute ndof
    ndof = int(np.prod(sim.model.nodes.shape)) * 2

    section_names = [
        "set_moduli",
        "mpc_assemble_matrix",
        "mpc_assemble_vector",
        "ksp_solve",
        "sbar_assemble",
    ]
    sections = {}
    for name in section_names:
        if name in timers:
            total_s, count = timers[name]
            sections[name] = {
                "total_s": total_s,
                "call_count": count,
                "per_call_ms": 1000.0 * total_s / count if count > 0 else 0.0,
                "pct_wall": 100.0 * total_s / wall_s if wall_s > 0 else 0.0,
            }
        else:
            sections[name] = {"total_s": 0.0, "call_count": 0,
                               "per_call_ms": 0.0, "pct_wall": 0.0}

    section_sum = sum(v["total_s"] for v in sections.values())
    return {
        "path": "periodic",
        "n": n,
        "n_freqs": n_freqs,
        "ndof": ndof,
        "wall_s": wall_s,
        "sections": sections,
        "section_sum_s": section_sum,
        "other_s": wall_s - section_sum,
        "other_pct_wall": 100.0 * (wall_s - section_sum) / wall_s if wall_s > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Standard path
# ---------------------------------------------------------------------------

def profile_standard(n, n_freqs):
    """Profile the standard (non-periodic) direct-Dirichlet path on an n×n grid.

    Returns dict with timing results.
    """
    from tests._matrix import matrix_simulation
    from microstructure_ve.backends.dolfinx import _run as run
    import microstructure_ve.backends.dolfinx._assembly as _asmbl

    sim = matrix_simulation(
        test_type="viscoelastic",
        mode="uniaxial_x",
        traction="confined_slip",
        bc="standard",
        dim=2,
        n=n,
        homogeneous=False,
    )

    freqs = np.logspace(-2.0, 2.0, n_freqs)

    # -- Warm-up
    print(f"  [standard] warming up FFCx compilation (n={n})...", flush=True)
    run.clear_cache()
    solve_one_warm, _ = run.build_solver(sim, solver="lu")
    solve_one_warm(float(freqs[0]))
    print(f"  [standard] warm-up done. Building instrumented solver...", flush=True)

    # -- Fresh solver
    run.clear_cache()

    timers = {}

    # -- Patch set_moduli
    _orig_set_moduli = _asmbl.MaterialFields.set_moduli

    def _timed_set_moduli(self, f):
        t0 = time.perf_counter()
        result = _orig_set_moduli(self, f)
        elapsed = time.perf_counter() - t0
        entry = timers.setdefault("set_moduli", [0.0, 0])
        entry[0] += elapsed
        entry[1] += 1
        return result

    _asmbl.MaterialFields.set_moduli = _timed_set_moduli

    # Capture the original fempetsc.assemble_matrix for use in timed_solve_one.
    # We do NOT apply a module-level patch here -- instead, timed_solve_one calls the
    # original directly with explicit timers, bypassing any need to intercept the module.
    import dolfinx.fem.petsc as _fempetsc_mod
    _orig_fempetsc_asm = _fempetsc_mod.assemble_matrix

    prob = None
    try:
        solve_one, _ = run.build_solver(sim, solver="lu")

        # Retrieve the cached prob and its std_mats to build a timed solve_one from scratch.
        # The closure returned by build_solver captures local PETSc object references whose
        # C-extension methods (ksp.solve, mat.mult) are read-only slots and cannot be
        # monkeypatched. Instead, we rebuild the same logic here with explicit timers.
        prob_key = next(iter(run._FE_CACHE))
        prob = run._FE_CACHE[prob_key]
        A_bc, b, x, u_sol, react_form, f_int, ksp = prob.std_mats

        from petsc4py import PETSc as _PETSc
        import dolfinx.fem.petsc as _fempetsc_local
        from microstructure_ve.backends.dolfinx._standard import _parse_bcs

        space, matfields, forms = prob.space, prob.matfields, prob.forms
        Vc_spaces, inv_maps = prob.vc_spaces, prob.inv_maps
        dirichlet_bcs, drive_nodes = _parse_bcs(sim, space, Vc_spaces, inv_maps)
        V = space.V
        bs = V.dofmap.index_map_bs
        blocks_drive = space.block_of_node[drive_nodes]
        dim = space.dim

        def timed_solve_one(f):
            # set_moduli already timed via class-level patch
            matfields.set_moduli(f)

            # A_bc reassemble: call the original directly (not the module-level shim)
            A_bc.zeroEntries()
            t0 = time.perf_counter()
            _orig_fempetsc_asm(A_bc, forms.a_form, bcs=dirichlet_bcs)
            A_bc.assemble()
            elapsed = time.perf_counter() - t0
            entry = timers.setdefault("std_assemble_A_bc", [0.0, 0])
            entry[0] += elapsed
            entry[1] += 1

            # RHS build (apply_lifting + set_bc)
            b.set(0.0)
            t0 = time.perf_counter()
            _fempetsc_local.apply_lifting(b, [forms.a_form], [dirichlet_bcs])
            b.ghostUpdate(addv=_PETSc.InsertMode.ADD, mode=_PETSc.ScatterMode.REVERSE)
            _fempetsc_local.set_bc(b, dirichlet_bcs)
            b.ghostUpdate(addv=_PETSc.InsertMode.INSERT, mode=_PETSc.ScatterMode.FORWARD)
            elapsed = time.perf_counter() - t0
            entry = timers.setdefault("rhs_build", [0.0, 0])
            entry[0] += elapsed
            entry[1] += 1

            # KSP solve (LU factor + back-sub combined for standard path)
            ksp.setOperators(A_bc)
            t0 = time.perf_counter()
            ksp.solve(b, x)
            elapsed = time.perf_counter() - t0
            entry = timers.setdefault("ksp_solve", [0.0, 0])
            entry[0] += elapsed
            entry[1] += 1

            x.ghostUpdate(addv=_PETSc.InsertMode.INSERT, mode=_PETSc.ScatterMode.FORWARD)
            u_arr = x.getArray()

            # Reaction force: action of the unreduced form on the solution
            t0 = time.perf_counter()
            u_sol.x.array[:] = u_arr
            with f_int.localForm() as fl:
                fl.set(0.0)
            _fempetsc_local.assemble_vector(f_int, react_form)
            f_int.ghostUpdate(addv=_PETSc.InsertMode.ADD, mode=_PETSc.ScatterMode.REVERSE)
            elapsed = time.perf_counter() - t0
            entry = timers.setdefault("react_assemble", [0.0, 0])
            entry[0] += elapsed
            entry[1] += 1

            f_arr = f_int.getArray()
            import numpy as np
            RF = np.zeros(dim, dtype=complex)
            U = np.zeros(dim, dtype=complex)
            for c in range(dim):
                flat_dofs = blocks_drive * bs + c
                RF[c] = f_arr[flat_dofs].sum()
                U[c] = u_arr[flat_dofs].sum()

            return [float(f)] + list(RF.real) + list(RF.imag) + list(U.real)

        print(f"  [standard] running sweep ({n_freqs} freqs)...", flush=True)
        wall_t0 = time.perf_counter()
        for f in freqs:
            timed_solve_one(float(f))
        wall_s = time.perf_counter() - wall_t0
        print(f"  [standard] done. wall={wall_s:.2f}s", flush=True)

    finally:
        _asmbl.MaterialFields.set_moduli = _orig_set_moduli

    ndof = int(np.prod(sim.model.nodes.shape)) * 2

    section_names = [
        "set_moduli",
        "std_assemble_A_bc",
        "rhs_build",
        "ksp_solve",
        "react_assemble",
    ]
    sections = {}
    for name in section_names:
        if name in timers:
            total_s, count = timers[name]
            sections[name] = {
                "total_s": total_s,
                "call_count": count,
                "per_call_ms": 1000.0 * total_s / count if count > 0 else 0.0,
                "pct_wall": 100.0 * total_s / wall_s if wall_s > 0 else 0.0,
            }
        else:
            sections[name] = {"total_s": 0.0, "call_count": 0,
                               "per_call_ms": 0.0, "pct_wall": 0.0}

    section_sum = sum(v["total_s"] for v in sections.values())
    return {
        "path": "standard",
        "n": n,
        "n_freqs": n_freqs,
        "ndof": ndof,
        "wall_s": wall_s,
        "sections": sections,
        "section_sum_s": section_sum,
        "other_s": wall_s - section_sum,
        "other_pct_wall": 100.0 * (wall_s - section_sum) / wall_s if wall_s > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(r):
    path = r["path"]
    print()
    print(f"=== {path.upper()} PATH (n={r['n']}, ndof={r['ndof']}, freqs={r['n_freqs']}) ===")
    print(f"  wall time : {r['wall_s']:.3f} s")
    print(f"  per-freq  : {1000.0 * r['wall_s'] / r['n_freqs']:.1f} ms/freq")
    print()
    print(f"  {'section':<25} {'total_s':>9} {'%wall':>7} {'calls':>7} {'ms/call':>9}")
    print("  " + "-" * 60)
    for name, sv in r["sections"].items():
        print(f"  {name:<25} {sv['total_s']:>9.3f} {sv['pct_wall']:>6.1f}% "
              f"{sv['call_count']:>7d} {sv['per_call_ms']:>9.2f}")
    section_sum = r["section_sum_s"]
    wall = r["wall_s"]
    print(f"  {'[sum]':<25} {section_sum:>9.3f} {100*section_sum/wall:>6.1f}%")
    print(f"  {'other':<25} {r['other_s']:>9.3f} {r['other_pct_wall']:>6.1f}%")


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_check(r):
    issues = []
    wall = r["wall_s"]
    s_sum = r["section_sum_s"]
    pct = 100 * s_sum / wall if wall > 0 else 0.0
    if pct < 60:
        issues.append(f"sections account for only {pct:.1f}% of wall (expected >60%)")
    if s_sum > wall * 1.05:
        issues.append(f"section_sum ({s_sum:.3f}s) > wall ({wall:.3f}s) — clock overlap?")

    n_freqs = r["n_freqs"]
    path = r["path"]
    secs = r["sections"]

    if path == "periodic":
        # set_moduli: 1 per freq
        sm = secs.get("set_moduli", {}).get("call_count", 0)
        if sm != n_freqs:
            issues.append(f"set_moduli calls={sm}, expected {n_freqs}")
        # mpc_assemble_matrix: 1 per freq (refactor)
        ma = secs.get("mpc_assemble_matrix", {}).get("call_count", 0)
        if ma != n_freqs:
            issues.append(f"mpc_assemble_matrix calls={ma}, expected {n_freqs}")
        # ksp.solve: 2 per freq (2 active modes for uniaxial_x free)
        ks = secs.get("ksp_solve", {}).get("call_count", 0)
        if ks != 2 * n_freqs:
            issues.append(f"ksp_solve calls={ks}, expected {2*n_freqs} (2 modes x {n_freqs} freqs)")
        # sbar_assemble: 2 modes x 3 needed_pairs (a0=0 pairs: (0,0),(1,0); free pairs: (0,0),(1,1))
        # Actually: needed_pairs = {(n, a0) for n in range(2)} | set(free)
        # a0=0, free={(1,1)}: needed_pairs={(0,0),(1,0),(1,1)} -> 3 pairs x 2 modes = 6 per freq
        sa = secs.get("sbar_assemble", {}).get("call_count", 0)
        # Could be 3 or 4 per freq per mode depending on free set; just check non-zero
        if sa == 0:
            issues.append("sbar_assemble was never called (fem.assemble_scalar not patched?)")

    elif path == "standard":
        # set_moduli: 1 per freq
        sm = secs.get("set_moduli", {}).get("call_count", 0)
        if sm != n_freqs:
            issues.append(f"set_moduli calls={sm}, expected {n_freqs}")
        # A_bc assembly: exactly 1 per freq (timed inline in timed_solve_one)
        abc = secs.get("std_assemble_A_bc", {}).get("call_count", 0)
        if abc != n_freqs:
            issues.append(f"std_assemble_A_bc calls={abc}, expected {n_freqs}")
        # ksp_solve: 1 per freq (one RHS)
        ks = secs.get("ksp_solve", {}).get("call_count", 0)
        if ks != n_freqs:
            issues.append(f"ksp_solve calls={ks}, expected {n_freqs}")
        # reaction action-form assembly: 1 per freq
        afm = secs.get("react_assemble", {}).get("call_count", 0)
        if afm != n_freqs:
            issues.append(f"react_assemble calls={afm}, expected {n_freqs}")

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile DOLFINx linear solve paths")
    parser.add_argument("--n", type=int, default=_DEFAULT_N,
                        help=f"grid size NxN (default: {_DEFAULT_N})")
    parser.add_argument("--freqs", type=int, default=_DEFAULT_FREQS,
                        help=f"number of frequencies in sweep (default: {_DEFAULT_FREQS})")
    parser.add_argument("--path", choices=["periodic", "standard", "both"],
                        default="both", help="which path(s) to profile (default: both)")
    args = parser.parse_args()

    n = args.n
    n_freqs = args.freqs
    paths = [args.path] if args.path != "both" else ["periodic", "standard"]

    print(f"profile_linear.py  n={n}  freqs={n_freqs}  paths={paths}")

    results = {}
    for path in paths:
        print(f"\n--- profiling: {path} ---")
        if path == "periodic":
            r = profile_periodic(n, n_freqs)
        else:
            r = profile_standard(n, n_freqs)
        results[path] = r
        print_table(r)

        issues = sanity_check(r)
        if issues:
            print(f"\n[WARNING] {path}:")
            for iss in issues:
                print(f"  - {iss}")
        else:
            wall = r["wall_s"]
            s_sum = r["section_sum_s"]
            pct = 100 * s_sum / wall
            print(f"\n[OK] {path}: sections={pct:.1f}% wall — sanity checks passed")

    # Write JSON
    def _serial(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    out = {"n": n, "n_freqs": n_freqs, "results": results}
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=_serial))
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
