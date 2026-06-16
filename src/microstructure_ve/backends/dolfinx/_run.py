"""Orchestration: wire the concern submodules into a solver and sweep the frequencies.

``build_solver`` assembles the spec/assembly/constraints/solver/homogenize pieces into a
``solve_one(f)`` closure (all dolfinx/PETSc state captured in it). ``run`` sweeps the
frequencies and optionally writes a readODB-style tsv, optionally fanning the independent
per-frequency solves across a spawn ProcessPool (``workers``). The macro loading is
uniaxial along x (coordinate axis 0).
"""
from __future__ import annotations

import numpy as np

from . import (
    _assembly as assembly,
    _constraints as constraints,
    _homogenize as homogenize,
    _loading as loadingmod,
    _spec as spec,
    _standard as standard,
)
from ._solver import Cancelled, IterativeSolver, LuSolver, select_solver_kind

from microstructure_ve.boundary import PeriodicBoundaryCondition


class _FEProblem:
    """The mesh-only FE pieces shared across every cell of a given mesh shape.

    ``space``/``matfields`` (DG0 fields)/``forms`` depend only on (shape, scale, bbar), not
    on the materials or loading, so they are built once and reused; the periodic extras
    (``mpc``, ``center_bcs``, ``lu``) and the standard extras (``vc_spaces``, ``inv_maps``)
    are built lazily on first use of each path. Per cell only the material *mapping* is
    refilled (``matfields.update_materials``) and the matrix reassembled.
    """

    def __init__(self, space, matfields, forms):
        self.space, self.matfields, self.forms = space, matfields, forms
        self.mpc = self.center_bcs = self.lu = None          # periodic, lazy
        self.lu_kind = None                                  # "lu" | "iterative" of self.lu
        self.vc_spaces = self.inv_maps = None                # standard dof maps, lazy
        self.std_mats = None                                 # standard matrices/vecs/ksp, lazy


_FE_CACHE = {}  # (shape, scale, dim, bbar) -> _FEProblem (process-level; tiny, a few shapes)


def clear_cache():
    """Drop the cached per-mesh-shape FE problems, releasing their dolfinx/PETSc objects.

    ``run`` transparently caches the mesh-only setup (mesh, dof maps, forms, MPC, factorizable
    matrix) keyed by ``(shape, scale, dim, bbar)``, so repeated calls on simulations that share
    a mesh shape -- e.g. sweeping many microstructure realizations or loadings on one grid --
    reuse it and skip the dominant build cost. The cache lives for the process; call this to
    free memory when you move on to different mesh shapes, or to force a clean rebuild. The
    cache assumes serial use (one ``run`` at a time, the backend's model); the ``workers`` path
    is unaffected since each spawned process has its own cache."""
    _FE_CACHE.clear()


def _fe_problem(geom, model, bbar):
    """Fetch (or build) the cached mesh-only FE problem for this geometry, rebinding it to
    ``model``'s materials. Building the mesh/forms/MPC dominates the suite, so caching by
    shape collapses it from once-per-cell to once-per-shape."""
    key = (tuple(geom.shape), float(geom.scale), int(geom.dim), bool(bbar))
    prob = _FE_CACHE.get(key)
    if prob is None:
        space = assembly.Space.build(geom)
        matfields = assembly.MaterialFields.from_model(space, model)
        forms = assembly.Forms.build(space, matfields, bbar)
        prob = _FEProblem(space, matfields, forms)
        _FE_CACHE[key] = prob
    else:
        prob.matfields.update_materials(model)
    return prob


def build_solver(sim, bbar=True, cancel=None, petsc_options=None, solver="auto"):
    """Build the FE solver for ``sim``; return ``(solve_one, dim)``.

    ``solve_one(f)`` returns the readODB-style row for one frequency. The mesh-only setup
    (mesh, dof map, forms, MPC/BCs, factorizable matrix) is cached by mesh shape and reused
    across cells (see ``_fe_problem``); only the per-cell material mapping and the
    per-frequency matrix values are refilled.

    Dispatches on the model's boundary conditions: *Periodic* (carries a
    ``PeriodicBoundaryCondition``) -> corner-driven homogenization via MPC; *Standard* ->
    direct-Dirichlet solve via ``_standard.build_solver`` (LU only for now).

    ``solver``: ``"auto"`` picks LU below the time budget and the GMRES+ILU
    ``IterativeSolver`` above it (``select_solver_kind``); ``"lu"``/``"iterative"`` force it.
    ``petsc_options`` overrides the iterative KSP/PC. ``cancel`` is polled per KSP iteration
    by the iterative solver (LU is cancelled only between frequencies, by ``run``).
    """
    if solver not in ("auto", "lu", "iterative"):
        raise ValueError(f"solver must be 'auto', 'lu' or 'iterative', got {solver!r}")
    model = sim.model
    has_pbc = any(isinstance(bc, PeriodicBoundaryCondition) for bc in model.bcs)
    geom = spec.Geometry.from_model(model, sim)
    prob = _fe_problem(geom, model, bbar)

    if not has_pbc:
        return standard.build_solver(sim, prob)

    loading = loadingmod.macro_loading(sim)  # raises NotImplementedError if unsupported
    if prob.mpc is None:
        prob.mpc = constraints.periodic_mpc(prob.space)
        prob.center_bcs = constraints.center_pin(prob.space)
    ndof = int(np.prod(geom.shape)) * geom.dim
    kind = solver if solver != "auto" else select_solver_kind(ndof, geom.dim)
    if prob.lu is None or prob.lu_kind != kind:
        if kind == "iterative":
            prob.lu = IterativeSolver(prob.space, prob.forms, prob.matfields, prob.mpc,
                                      prob.center_bcs, cancel=cancel, petsc_options=petsc_options)
        else:
            prob.lu = LuSolver(prob.space, prob.forms, prob.matfields, prob.mpc, prob.center_bcs)
        prob.lu_kind = kind
    if kind == "iterative":
        prob.lu._cancel = cancel  # refresh per-run cancel on a (possibly cached) iterative solver
    solve_one = homogenize.build_solve_one(prob.space, prob.forms, prob.lu, loading)
    return solve_one, prob.space.dim


_THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
)
_WORKER = {}  # per-process solver cache, populated by _init_worker in parallel mode


def _init_worker(sim, bbar, solver, petsc_options):
    """ProcessPool worker initializer: build the solver once and cache it."""
    import os
    for v in _THREAD_VARS:
        os.environ.setdefault(v, "1")
    # cancel is None in workers: the parent polls cancel and SIGTERMs live workers
    _WORKER["solve_one"], _WORKER["dim"] = build_solver(
        sim, bbar, petsc_options=petsc_options, solver=solver)


def _worker_solve(f):
    return _WORKER["solve_one"](float(f))


def _row_header(dim):
    return (
        ["frame_value"]  # frequency (Hz) for Dynamic; step time for Static/transient
        + [f"RF_Real{i + 1}" for i in range(dim)]
        + [f"RF_Imag{i + 1}" for i in range(dim)]
        + [f"U{i + 1}" for i in range(dim)]
    )


def run(sim, output_path=None, bbar=True, workers=1, cancel=None, solver="auto",
        petsc_options=None):
    """Solve ``sim`` over its frequency sweep; one row per frequency, ``(n_freq, 1+3*dim)``.

    Everything about the *problem* is read from ``sim`` -- there are no physics kwargs.
    The frequencies come from ``spec.frequencies``; the macro loading (driven axis/mode and
    the free vs held lateral components) is parsed from the corner BCs and the step drive(s)
    by ``_loading.macro_loading``; the zero-amplitude baseline
    ``DisplacementBoundaryCondition`` in ``model.bcs`` is the ABAQUS path's convention (see
    ``example.py``). The remaining kwargs are execution knobs only.

    Each row is ``[frame_value, RF_Real_1..d, RF_Imag_1..d, U_1..d]`` (same columns as the
    ABAQUS readODB tsv; ``frame_value`` is the frequency for a Dynamic sweep, the step time
    for a Static/transient step), where ``RF`` is the complex reaction on the +x face and ``U`` the
    applied corner displacement. The homogenized complex modulus along x is::

        E*_x(f) = (RF_Real_1 + 1j*RF_Imag_1) / (cross_area * exx)

    with ``cross_area = Ly`` (2D) or ``Ly*Lz`` (3D) and ``exx = U_1 / Lx`` -- i.e. divide
    the x reaction by the cross-section and the applied macro strain (see ``Geometry``).

    bbar:    selective reduced integration on the volumetric term (recommended; matches
             ABAQUS CPE4/C3D8 B-bar and avoids Q1 volumetric locking).
    workers: 1 = serial. >1 spawns a ProcessPoolExecutor and splits the independent
             per-frequency solves across processes -- each worker builds the solver once,
             then solves its share, pinned to one BLAS thread. A driver that calls this
             with workers>1 MUST guard the call under ``if __name__ == "__main__":`` (the
             standard "spawn" requirement) or the workers will re-import and re-run it.
    cancel:  callable() -> bool, or None. Polled before each frequency. Stop the sweep
             either by returning a truthy value (raises ``Cancelled``) or by raising your
             own exception (propagates unchanged) -- the latter lets a caller carry a
             reason/payload out of the sweep. Because an LU solve is one uninterruptible C
             call, cancellation takes effect at the start of the next frequency, so
             worst-case latency is one solve (kept under ~10s by the solver-kind crossover;
             see ``build_solver``). In parallel mode it is polled in this parent process as
             workers finish, and either trigger SIGTERMs the live workers (killing an
             in-flight solve too) before propagating; the predicate runs here, never in a
             worker. With ``solver="iterative"`` ``cancel`` is also polled *inside* each
             solve (per KSP iteration), so latency is sub-second on large meshes.
    solver:  ``"auto"`` (default) keeps LU in 2D (where it scales well and the iterative path
             diverges) and switches to the GMRES+ILU ``IterativeSolver`` only for large 3D
             meshes above the LU time budget (``select_solver_kind``); ``"lu"``/``"iterative"``
             force one. The complex-symmetric system has no AMG (GAMG/hypre are real-only),
             so the iterative path is GMRES+ILU and may stagnate on the hardest meshes (it
             raises on non-convergence). Periodic path only; standard BC uses LU.
    petsc_options: dict of PETSc options overriding the iterative KSP/PC (e.g.
             ``{"pc_type": "bjacobi"}``); ignored for LU.
    """
    dim = sim.model.nodes.dim

    if len(list(sim.steps)) > 1:
        out = _run_multistep(sim, bbar, cancel, solver, petsc_options)
        if output_path is not None:
            np.savetxt(output_path, out, fmt="%.8e", delimiter="\t",
                       header="\t".join(_row_header(dim)), comments="")
        return out

    freqs = np.asarray(spec.frequencies(sim), dtype=float)

    if workers and workers > 1 and len(freqs) > 1:
        import multiprocessing

        # If we are already inside a spawned worker, the pool's children re-imported and
        # re-ran the calling module -- detect that directly (a real parent process exists)
        # rather than trying to infer whether the caller had an __main__ guard.
        if multiprocessing.parent_process() is not None:
            raise RuntimeError(
                "run(workers>1) was reached inside a multiprocessing worker process: the "
                "spawned workers re-imported and re-ran the calling module. Invoke "
                "run(workers>1) from a guarded entry point (under "
                '`if __name__ == "__main__":`) or set workers=1.'
            )
        out = _run_parallel(sim, freqs, bbar, workers, cancel, solver, petsc_options)
    else:
        solve_one, _ = build_solver(sim, bbar, cancel=cancel,
                                    petsc_options=petsc_options, solver=solver)
        rows = []
        for f in freqs:
            if cancel is not None and cancel():
                raise Cancelled("cancelled by callback")
            rows.append(solve_one(f))
        out = np.array(rows)

    if output_path is not None:
        np.savetxt(
            output_path, out, fmt="%.8e", delimiter="\t",
            header="\t".join(_row_header(dim)), comments="",
        )
    return out


def _run_multistep(sim, bbar, cancel=None, solver="auto", petsc_options=None):
    """Sweep a multi-step sim step-by-step, emitting rows in the ABAQUS reader's order (per
    step, then per frame): a ``Static`` step contributes one elastic row (zero loss) at frame
    value 1.0 (real ``*Elastic`` moduli), a ``Dynamic`` step one row per swept frequency
    (ascending). The FE problem (mesh/MPC/forms) is built once and reused across steps. The
    multi-step cells drive the same macro loading each step, so one solver serves all."""
    from microstructure_ve.steps import Dynamic, Static

    solve_one, _ = build_solver(sim, bbar, cancel=cancel,
                                petsc_options=petsc_options, solver=solver)
    rows = []
    for step in sim.steps:
        if cancel is not None and cancel():
            raise Cancelled("cancelled by callback")
        dyn = spec.find(step.subsections, Dynamic)
        if dyn is not None:
            freqs = np.logspace(np.log10(dyn.f_initial), np.log10(dyn.f_final), dyn.f_count)
            rows.extend(solve_one(float(f)) for f in freqs)
        elif spec.find(step.subsections, Static) is not None:
            rows.append(solve_one(1.0, elastic=True))  # frame value 1.0, real *Elastic moduli
    return np.array(rows)


def _kill_workers(ex):
    """SIGTERM every live worker of ``ex``. Uses the private ``_processes`` dict because
    ProcessPoolExecutor exposes no public way to kill in-flight tasks; SIGTERM reaches a
    worker even mid-LU-solve (the C call dies with the process)."""
    import os
    import signal

    for proc in list(getattr(ex, "_processes", {}).values()):
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def _run_parallel(sim, freqs, bbar, workers, cancel=None, solver="auto", petsc_options=None):
    """Fan the per-frequency solves across a spawn ProcessPoolExecutor (order preserved).

    ``cancel`` (parent-process predicate) is polled as futures complete; on True the live
    workers are SIGTERMed and ``Cancelled`` is raised. With ``cancel=None`` this keeps the
    plain ``ex.map`` fast path.
    """
    import os
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # Spawn children inherit os.environ at interpreter start (before they import numpy),
    # so set the BLAS thread caps here to keep workers single-threaded; restore after.
    saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(freqs)), mp_context=ctx,
            initializer=_init_worker, initargs=(sim, bbar, solver, petsc_options),
        ) as ex:
            if cancel is None:
                rows = list(ex.map(_worker_solve, freqs))  # map preserves input order
            else:
                rows = _map_cancellable(ex, freqs, cancel)
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old
    return np.array(rows)


def _map_cancellable(ex, freqs, cancel):
    """Submit all frequencies, collect in order, polling ``cancel`` as each completes.

    ``cancel`` may signal a stop two ways, both handled here: returning truthy raises
    ``Cancelled``; raising its own exception propagates unchanged. Either way the live
    workers are SIGTERMed first, so we never fall through to the executor's ``__exit__``
    which would otherwise block waiting for the running solves to finish.
    """
    from concurrent.futures import as_completed

    index = {ex.submit(_worker_solve, float(f)): i for i, f in enumerate(freqs)}
    rows = [None] * len(freqs)
    for fut in as_completed(index):
        rows[index[fut]] = fut.result()
        try:
            stop = cancel()
        except BaseException:  # the user's own exception: kill workers, then re-raise it
            _kill_workers(ex)
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        if stop:
            _kill_workers(ex)
            ex.shutdown(wait=False, cancel_futures=True)
            raise Cancelled("cancelled by callback")
    return rows
