"""Orchestration: wire the concern submodules into a solver and sweep the frequencies.

``build_solver`` assembles the spec/assembly/constraints/solver/homogenize pieces into a
``solve_one(f)`` closure (all dolfinx/PETSc state captured in it). ``run`` sweeps the
frequencies and optionally writes a readODB-style tsv, optionally fanning the independent
per-frequency solves across a spawn ProcessPool (``workers``). The macro loading is
uniaxial along x (coordinate axis 0).

Thread safety (the guarantee): every entry point of this backend -- ``run``,
``build_solver`` plus the ``solve_one`` closures it returns, and ``clear_cache`` --
is safe to call concurrently from multiple threads. Everything that touches in-process
FE state (serial and multistep ``run``, ``build_solver``/``solve_one``, ``clear_cache``)
serializes on one process-wide re-entrant lock (``_BACKEND_LOCK``): the cached
``_FEProblem``s and the dolfinx/PETSc objects inside them are shared mutable state
(material refill, matrix reassembly, factorization reuse), so those calls execute one
at a time, each seeing a consistent cache -- *safe but not parallel*. The exception is
a ``workers>1`` frequency sweep: its worker processes rebuild everything from the
pickled sim and share nothing with this process's cache, so the fan-out runs *outside*
the lock and parallel sweeps launched from several threads genuinely overlap. That path
touches no process-global state either -- each worker caps its own BLAS/OpenMP threads
in-process (threadpoolctl, in ``_init_worker``), so the parent's ``os.environ`` is never
mutated.
"""
from __future__ import annotations

import os
import threading

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

from microstructure_ve.boundary import PeriodicBoundaryConstraint


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


# (shape, scale, dim, bbar) -> _FEProblem. Process-level and tiny (a few shapes), but the
# _FEProblems it holds are mutated per call (material refill, reassembly, lazy solver swap),
# so every access -- and every use of a cached problem -- happens under _BACKEND_LOCK.
_FE_CACHE = {}

# The backend-wide serialization lock behind the thread-safety guarantee (see the module
# docstring): held for the full duration of every call that touches in-process FE state --
# serial/multistep run(), build_solver(), solve_one(), clear_cache(). The workers>1 fan-out
# runs outside it (nothing in this process is shared with the spawned workers).
# Re-entrant because run() calls build_solver() (and solve_one) while already holding it.
_BACKEND_LOCK = threading.RLock()


def _locked(fn):
    """Wrap ``fn`` so each call runs under ``_BACKEND_LOCK`` (re-entrant, so safe within
    ``run``'s own locked scope)."""
    def locked(*args, **kwargs):
        with _BACKEND_LOCK:
            return fn(*args, **kwargs)
    return locked


def clear_cache():
    """Drop the cached per-mesh-shape FE problems, releasing their dolfinx/PETSc objects.

    ``run`` transparently caches the mesh-only setup (mesh, dof maps, forms, MPC, factorizable
    matrix) keyed by ``(shape, scale, dim, bbar)``, so repeated calls on simulations that share
    a mesh shape -- e.g. sweeping many microstructure realizations or loadings on one grid --
    reuse it and skip the dominant build cost. The cache lives for the process; call this to
    free memory when you move on to different mesh shapes, or to force a clean rebuild.
    Thread-safe: takes the same process-wide lock as ``run``/``build_solver`` (see the module
    docstring); the ``workers`` path is unaffected since each spawned process has its own
    cache."""
    with _BACKEND_LOCK:
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
    ``PeriodicBoundaryConstraint``) -> corner-driven homogenization via MPC; *Standard* ->
    direct-Dirichlet solve via ``_standard.build_solver`` (LU only for now).

    ``solver``: ``"auto"`` picks LU below the time budget and the GMRES+ILU
    ``IterativeSolver`` above it (``select_solver_kind``); ``"lu"``/``"iterative"`` force it.
    ``petsc_options`` overrides the iterative KSP/PC. ``cancel`` is polled per KSP iteration
    by the iterative solver (LU is cancelled only between frequencies, by ``run``).

    Thread safety: both the build and every call of the returned ``solve_one`` serialize
    on the backend lock (they share the cached, mutable ``_FEProblem`` -- see the module
    docstring), so handing the closure to another thread is safe; concurrent calls simply
    run one at a time.
    """
    with _BACKEND_LOCK:
        solve_one, dim = _build_solver(sim, bbar, cancel, petsc_options, solver)
    return _locked(solve_one), dim


def _build_solver(sim, bbar, cancel, petsc_options, solver):
    if solver not in ("auto", "lu", "iterative"):
        raise ValueError(f"solver must be 'auto', 'lu' or 'iterative', got {solver!r}")
    model = sim.model
    has_pbc = any(isinstance(bc, PeriodicBoundaryConstraint) for bc in model.bcs)
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


_WORKER = {}  # per-process solver cache, populated by _init_worker in parallel mode


def _init_worker(sim, bbar, solver, petsc_options):
    """ProcessPool worker initializer: pin BLAS threads, then build the solver once.

    ``threadpool_limits(1)`` caps this worker's OpenBLAS/OpenMP/MKL pools to a single
    thread in-process (regardless of when those libraries loaded), so N workers on one
    box never oversubscribe cores -- and, unlike setting ``os.environ`` in the parent, it
    leaves the parent's environment untouched. The returned controller is kept alive in
    ``_WORKER`` so the cap persists for the worker's lifetime."""
    try:
        import threadpoolctl
    except ImportError as e:  # pragma: no cover - env misconfiguration guard
        raise RuntimeError(
            "run(workers>1) needs 'threadpoolctl' to cap each worker's BLAS threads, but "
            "it is not importable in this environment. Install it into the FEniCSx env "
            "(e.g. `conda install -c conda-forge threadpoolctl`) or use workers=1."
        ) from e
    _WORKER["thread_limits"] = threadpoolctl.threadpool_limits(limits=1)
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
        petsc_options=None, n_incr=None):
    """Solve ``sim`` over its frequency sweep; one row per frequency, ``(n_freq, 1+3*dim)``.

    Everything about the *problem* is read from ``sim`` -- there are no physics kwargs.
    The frequencies come from ``spec.frequencies``; the macro loading (driven axis/mode and
    the free vs held lateral components) is parsed from the corner BCs and the step drive(s)
    by ``_loading.macro_loading``; the zero-amplitude baseline ``Prescribed`` condition
    in ``model.bcs`` is the ABAQUS path's convention (see ``example.py``). The remaining
    kwargs are execution knobs only.

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
    n_incr:  load increments per plastic ``Static`` step (periodic path). Default ``None``
             keeps the built-in choice: 1 when the macro strain is fully prescribed
             (monotonic proportional loading is increment-independent for the radial
             return) and 20 when a free lateral component makes the strain path curved.
             Ignored for non-plastic sims and the standard-BC plastic path (single ramp).

    Thread safety: ``run`` may be called concurrently from multiple threads. Serial
    sweeps and the multistep paths serialize on a process-wide lock held for the whole
    call (see the module docstring), so each returns exactly what a serial call would --
    on those paths threads buy safety, not speed. A ``workers>1`` sweep runs its fan-out
    *outside* that lock (the worker processes share no state with this one), so parallel
    sweeps launched from several threads genuinely overlap. A ``cancel`` callback always
    runs in the calling thread; on the locked (serial/multistep) paths it must not call
    back into this backend from *another* thread and wait on it (that would deadlock) --
    in a ``workers>1`` sweep the lock is not held, so that caveat does not apply.
    """
    dim = sim.model.nodes.dim

    # Dispatch below reads only the caller's sim (pure numpy checks), so it needs no lock.
    if _has_hyperelastic(sim.model):
        from microstructure_ve.steps import Dynamic

        if any(spec.find(step.subsections, Dynamic) is not None for step in sim.steps):
            raise NotImplementedError(
                "steady-state dynamics about a hyperelastic (finite-strain) state is "
                "not supported; hyperelastic sims must be Static-only"
            )
        if _has_plastic_static(sim):
            raise NotImplementedError(
                "mixing hyperelastic and plastic responses in one model is not supported"
            )

    if len(list(sim.steps)) > 1 or _has_plastic_static(sim) or _has_hyperelastic_static(sim):
        with _BACKEND_LOCK:
            out = _run_multistep(sim, bbar, cancel, solver, petsc_options, n_incr)
    else:
        freqs = np.asarray(spec.frequencies(sim), dtype=float)
        if workers and workers > 1 and len(freqs) > 1:
            import multiprocessing

            # If we are already inside a spawned worker, the pool's children re-imported
            # and re-ran the calling module -- detect that directly (a real parent process
            # exists) rather than trying to infer whether the caller had an __main__ guard.
            if multiprocessing.parent_process() is not None:
                raise RuntimeError(
                    "run(workers>1) was reached inside a multiprocessing worker process: "
                    "the spawned workers re-imported and re-ran the calling module. Invoke "
                    "run(workers>1) from a guarded entry point (under "
                    '`if __name__ == "__main__":`) or set workers=1.'
                )
            # No _BACKEND_LOCK here: the fan-out shares nothing with this process's FE
            # cache, so parallel sweeps from several threads may overlap (module docstring).
            out = _run_parallel(sim, freqs, bbar, workers, cancel, solver, petsc_options)
        else:
            with _BACKEND_LOCK:
                out = _run_serial_sweep(sim, freqs, bbar, cancel, solver, petsc_options)

    if output_path is not None:
        np.savetxt(
            output_path, out, fmt="%.8e", delimiter="\t",
            header="\t".join(_row_header(dim)), comments="",
        )
    return out


def _run_serial_sweep(sim, freqs, bbar, cancel, solver, petsc_options):
    solve_one, _ = build_solver(sim, bbar, cancel=cancel,
                                petsc_options=petsc_options, solver=solver)
    rows = []
    for f in freqs:
        if cancel is not None and cancel():
            raise Cancelled("cancelled by callback")
        rows.append(solve_one(f))
    return np.array(rows)


def _has_plastic_static(sim):
    """True iff ``sim`` carries a ``Plastic`` response and a ``Static`` step (numpy-only check).

    Such a Static step needs the nonlinear return-mapping solve (``_plastic``) instead of the
    linear elastic ``solve_one``; a Dynamic step over the same materials stays linear (the
    *Plastic table is irrelevant to a steady-state perturbation)."""
    from microstructure_ve.constitutive import Plastic
    from microstructure_ve.steps import Static

    if not any(isinstance(m.response, Plastic) for m in sim.model.materials):
        return False
    return any(any(isinstance(s, Static) for s in step.subsections) for step in sim.steps)


def _has_hyperelastic(model):
    """True iff ``model`` carries a hyperelastic response (numpy-only check)."""
    from microstructure_ve.constitutive import (
        ArrudaBoyce, NeoHookean, Polynomial, ReducedPolynomial,
    )

    return any(
        isinstance(m.response, (ArrudaBoyce, ReducedPolynomial, Polynomial, NeoHookean))
        for m in model.materials
    )


def _has_hyperelastic_static(sim):
    """True iff ``sim`` carries a hyperelastic response and a ``Static`` step (numpy-only).

    Such a step needs the finite-strain total-Lagrangian solve (``_hyperelastic``) instead
    of the linear elastic ``solve_one``."""
    from microstructure_ve.steps import Static

    if not _has_hyperelastic(sim.model):
        return False
    return any(any(isinstance(s, Static) for s in step.subsections) for step in sim.steps)


def _run_multistep(sim, bbar, cancel=None, solver="auto", petsc_options=None, n_incr=None):
    """Sweep a multi-step sim step-by-step, emitting rows in the ABAQUS reader's order (per
    step, then per frame): a ``Static`` step contributes one row (zero loss) at frame value 1.0
    -- elastic (real ``*Elastic`` moduli) or, if a ``Plastic`` response is present, the nonlinear
    J2 return-mapping solve (``_plastic``) -- and a ``Dynamic`` step one row per swept frequency
    (ascending). The FE problem (mesh/MPC/forms) is built once and reused across steps. The
    multi-step cells drive the same macro loading each step, so one solver serves all."""
    from microstructure_ve.constitutive import Plastic
    from microstructure_ve.steps import Dynamic, Static

    solve_one, _ = build_solver(sim, bbar, cancel=cancel,
                                petsc_options=petsc_options, solver=solver)
    has_plastic = any(isinstance(m.response, Plastic) for m in sim.model.materials)
    has_hyper = _has_hyperelastic(sim.model)
    has_pbc = any(isinstance(bc, PeriodicBoundaryConstraint) for bc in sim.model.bcs)
    plastic_solver = None
    if has_plastic:
        from . import _plastic as _plastic
        geom = spec.Geometry.from_model(sim.model, sim)
        prob = _fe_problem(geom, sim.model, bbar)          # cached, populated by build_solver
        # ONE persistent solver for the whole sim: plastic state (eps_p, p, u~, E_current)
        # carries across consecutive plastic Static steps, so reversal/cyclic patterns
        # accumulate the correct hysteresis. The standard (non-periodic) path drives face
        # Dirichlet directly and is single-step (no macro-strain/free split).
        if has_pbc:
            plastic_solver = _plastic.make_solver(prob, sim.model)
        else:
            plastic_solver = _plastic.make_standard_solver(prob, sim.model, sim)
    hyper_solver = None
    if has_hyper:
        from . import _hyperelastic
        geom = spec.Geometry.from_model(sim.model, sim)
        prob = _fe_problem(geom, sim.model, bbar)          # cached, populated by build_solver
        # ONE persistent finite-strain solver: H_current and the fluctuation warm-start
        # carry across consecutive hyperelastic Static steps (the response itself is
        # path-independent).
        if has_pbc:
            hyper_solver = _hyperelastic.make_solver(prob, sim.model)
        else:
            hyper_solver = _hyperelastic.make_standard_solver(prob, sim.model, sim)

    rows = []
    for step in sim.steps:
        if cancel is not None and cancel():
            raise Cancelled("cancelled by callback")
        dyn = spec.find(step.subsections, Dynamic)
        if dyn is not None:
            freqs = np.logspace(np.log10(dyn.f_initial), np.log10(dyn.f_final), dyn.f_count)
            rows.extend(solve_one(float(f)) for f in freqs)
        elif spec.find(step.subsections, Static) is not None:
            if has_plastic and has_pbc:
                # parse this step's own drive (steps may drive different magnitudes -- e.g. a
                # harmonic step then a plastic load, or a load then a reversal)
                loading = loadingmod.macro_loading(sim, step=step)
                step_incr = n_incr
                if step_incr is None:
                    step_incr = 1 if not list(loading.free) else 20
                rows.append(plastic_solver.solve(loading, step_incr))
            elif has_plastic:
                rows.append(plastic_solver.solve())            # standard: BCs parsed from sim
            elif has_hyper and has_pbc:
                loading = loadingmod.macro_loading(sim, step=step)
                step_incr = n_incr
                # a few increments even when fully prescribed: Newton globalization at
                # ~30-50% strain (the answer is increment-independent -- path-independent
                # energy -- only the warm-start path changes)
                if step_incr is None:
                    step_incr = 5 if not list(loading.free) else 10
                rows.append(hyper_solver.solve(loading, step_incr))
            elif has_hyper:
                rows.append(hyper_solver.solve(n_incr if n_incr is not None else 5))
            else:
                rows.append(solve_one(1.0, elastic=True))  # frame value 1.0, real *Elastic
    return np.array(rows)


def _kill_workers(ex):
    """SIGTERM every live worker of ``ex``. Uses the private ``_processes`` dict because
    ProcessPoolExecutor exposes no public way to kill in-flight tasks; SIGTERM reaches a
    worker even mid-LU-solve (the C call dies with the process)."""
    import signal

    for proc in list(getattr(ex, "_processes", {}).values()):
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def _run_parallel(sim, freqs, bbar, workers, cancel=None, solver="auto", petsc_options=None):
    """Fan the per-frequency solves across a spawn ProcessPoolExecutor (order preserved).

    Each worker caps its own BLAS/OpenMP threads to one in ``_init_worker`` (threadpoolctl),
    so the parent's ``os.environ`` is never touched and workers never oversubscribe cores.

    ``cancel`` (parent-process predicate) is polled as futures complete; on True the live
    workers are SIGTERMed and ``Cancelled`` is raised. With ``cancel=None`` this keeps the
    plain ``ex.map`` fast path.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(workers), len(freqs)), mp_context=ctx,
        initializer=_init_worker, initargs=(sim, bbar, solver, petsc_options),
    ) as ex:
        if cancel is None:
            rows = list(ex.map(_worker_solve, freqs))  # map preserves input order
        else:
            rows = _map_cancellable(ex, freqs, cancel)
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
