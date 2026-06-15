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
)
from ._solver import Cancelled, LuSolver, predict_lu_seconds, select_solver_kind


def build_solver(sim, bbar=True):
    """Build the FE solver for ``sim`` once; return ``(solve_one, dim)``.

    ``solve_one(f)`` returns the readODB-style row for one frequency. The setup (mesh,
    dof map, materials, forms, MPC, factorizable matrix) is paid once and reused across
    frequencies and both unit-strain RHS.

    The solver kind is chosen by ``select_solver_kind`` from the predicted LU time. Above
    the crossover the intended solver is iterative (cancellable mid-solve), but it is not
    implemented yet, so this warns and falls back to LU -- which can only be cancelled
    between whole solves, so a single factorization on such a mesh will block past the
    ~10s budget. ``cancel`` is not threaded in here: the LU path is cancelled by the
    sweep loop in ``run`` between frequencies, not inside ``build_solver``.
    """
    model = sim.model
    spec.require_periodic(model)
    loading = loadingmod.macro_loading(sim)  # raises NotImplementedError if unsupported
    geom = spec.Geometry.from_model(model, sim)
    dim = geom.dim
    ndof = int(np.prod(geom.shape)) * dim
    if select_solver_kind(ndof, dim) == "iterative":
        import warnings

        warnings.warn(
            f"predicted LU solve ~{predict_lu_seconds(ndof, dim):.0f}s for ndof={ndof} "
            f"({dim}D) exceeds the ~10s budget; the iterative solver is not implemented "
            "yet, falling back to LU -- a single solve cannot be cancelled mid-"
            "factorization, so cancellation latency will exceed 10s on this mesh.",
            stacklevel=2,
        )

    space = assembly.Space.build(geom)
    matfields = assembly.MaterialFields.from_model(space, model)
    forms = assembly.Forms.build(space, matfields, bbar)
    mpc = constraints.periodic_mpc(space)
    bcs = constraints.center_pin(space)
    solver = LuSolver(space, forms, matfields, mpc, bcs)
    solve_one = homogenize.build_solve_one(space, forms, solver, loading)
    return solve_one, space.dim


_THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
)
_WORKER = {}  # per-process solver cache, populated by _init_worker in parallel mode


def _init_worker(sim, bbar):
    """ProcessPool worker initializer: build the solver once and cache it."""
    import os
    for v in _THREAD_VARS:
        os.environ.setdefault(v, "1")
    _WORKER["solve_one"], _WORKER["dim"] = build_solver(sim, bbar)


def _worker_solve(f):
    return _WORKER["solve_one"](float(f))


def _row_header(dim):
    return (
        ["frequency"]
        + [f"RF_Real{i + 1}" for i in range(dim)]
        + [f"RF_Imag{i + 1}" for i in range(dim)]
        + [f"U{i + 1}" for i in range(dim)]
    )


def run(sim, output_path=None, bbar=True, workers=1, cancel=None):
    """Solve ``sim`` over its frequency sweep; one row per frequency, ``(n_freq, 1+3*dim)``.

    Everything about the *problem* is read from ``sim`` -- there are no physics kwargs.
    The frequencies come from ``spec.frequencies``; the macro loading (driven axis/mode and
    the free vs held lateral components) is parsed from the corner BCs and the step drive(s)
    by ``_loading.macro_loading``; the zero-amplitude baseline
    ``DisplacementBoundaryCondition`` in ``model.bcs`` is the ABAQUS path's convention (see
    ``example.py``). The remaining kwargs are execution knobs only.

    Each row is ``[frequency, RF_Real_1..d, RF_Imag_1..d, U_1..d]`` (same columns as the
    ABAQUS readODB tsv), where ``RF`` is the complex reaction on the +x face and ``U`` the
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
             worker.
    """
    freqs = np.asarray(spec.frequencies(sim), dtype=float)
    dim = sim.model.nodes.dim

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
        out = _run_parallel(sim, freqs, bbar, workers, cancel)
    else:
        solve_one, _ = build_solver(sim, bbar)
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


def _run_parallel(sim, freqs, bbar, workers, cancel=None):
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
            initializer=_init_worker, initargs=(sim, bbar),
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
