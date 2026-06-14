"""Orchestration: wire the concern submodules into a solver and sweep the frequencies.

``build_solver`` assembles the spec/assembly/constraints/solver/homogenize pieces into a
``solve_one(f)`` closure (all dolfinx/PETSc state captured in it). ``run`` sweeps the
frequencies and optionally writes a readODB-style tsv, optionally fanning the independent
per-frequency solves across a spawn ProcessPool (``workers``). The macro loading is
uniaxial along x (coordinate axis 0).
"""
from __future__ import annotations

import numpy as np

from . import _assembly as assembly, _constraints as constraints, _homogenize as homogenize, _spec as spec
from ._solver import Solver


def build_solver(sim, lateral_bc="confined", bbar=True):
    """Build the FE solver for ``sim`` once; return ``(solve_one, dim)``.

    ``solve_one(f)`` returns the readODB-style row for one frequency. The setup (mesh,
    dof map, materials, forms, MPC, factorizable matrix) is paid once and reused across
    frequencies and both unit-strain RHS.
    """
    model = sim.model
    spec.require_periodic(model)
    geom = spec.Geometry.from_model(model, sim)

    space = assembly.Space.build(geom)
    matfields = assembly.MaterialFields.from_model(space, model)
    forms = assembly.Forms.build(space, matfields, bbar)
    mpc = constraints.periodic_mpc(space)
    bcs = constraints.center_pin(space)
    solver = Solver(space, forms, matfields, mpc, bcs)
    solve_one = homogenize.build_solve_one(space, forms, solver, geom, lateral_bc)
    return solve_one, space.dim


_THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
)
_WORKER = {}  # per-process solver cache, populated by _init_worker in parallel mode


def _init_worker(sim, lateral_bc, bbar):
    """ProcessPool worker initializer: build the solver once and cache it."""
    import os
    for v in _THREAD_VARS:
        os.environ.setdefault(v, "1")
    _WORKER["solve_one"], _WORKER["dim"] = build_solver(sim, lateral_bc, bbar)


def _worker_solve(f):
    return _WORKER["solve_one"](float(f))


def _row_header(dim):
    return (
        ["frequency"]
        + [f"RF_Real{i + 1}" for i in range(dim)]
        + [f"RF_Imag{i + 1}" for i in range(dim)]
        + [f"U{i + 1}" for i in range(dim)]
    )


def run(sim, freqs=None, output_path=None, lateral_bc="confined", bbar=True, workers=1):
    """Solve ``sim`` over ``freqs``; return one row per frequency, ``(len(freqs), 1+3*dim)``.

    The drive is read from the step: put a ``DisplacementBoundaryCondition`` in the
    step's ``subsections`` giving the applied x displacement (and, as the ABAQUS path
    needs, a zero-amplitude baseline ``DisplacementBoundaryCondition`` in ``model.bcs``;
    see ``example.py``). ``freqs`` defaults to the step's ``Dynamic`` sweep.

    Each row is ``[frequency, RF_Real_1..d, RF_Imag_1..d, U_1..d]`` (same columns as the
    ABAQUS readODB tsv), where ``RF`` is the complex reaction on the +x face and ``U`` the
    applied corner displacement. The homogenized complex modulus along x is::

        E*_x(f) = (RF_Real_1 + 1j*RF_Imag_1) / (cross_area * exx)

    with ``cross_area = Ly`` (2D) or ``Ly*Lz`` (3D) and ``exx = U_1 / Lx`` -- i.e. divide
    the x reaction by the cross-section and the applied macro strain (see ``Geometry``).

    lateral_bc: "confined" -> lateral macro normal strains are held at 0 (plane-strain-
             style constraint); "free" -> lateral macro normal *stresses* vanish, the cell
             contracts by Poisson (the right choice for an apparent uniaxial modulus).
    bbar:    selective reduced integration on the volumetric term (recommended; matches
             ABAQUS CPE4/C3D8 B-bar and avoids Q1 volumetric locking).
    workers: 1 = serial. >1 spawns a ProcessPoolExecutor and splits the independent
             per-frequency solves across processes -- each worker builds the solver once,
             then solves its share, pinned to one BLAS thread. A driver that calls this
             with workers>1 MUST guard the call under ``if __name__ == "__main__":`` (the
             standard "spawn" requirement) or the workers will re-import and re-run it.
    """
    if freqs is None:
        freqs = spec.default_frequencies(sim)
    freqs = np.asarray(freqs, dtype=float)
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
        out = _run_parallel(sim, freqs, lateral_bc, bbar, workers)
    else:
        solve_one, _ = build_solver(sim, lateral_bc, bbar)
        out = np.array([solve_one(f) for f in freqs])

    if output_path is not None:
        np.savetxt(
            output_path, out, fmt="%.8e", delimiter="\t",
            header="\t".join(_row_header(dim)), comments="",
        )
    return out


def _run_parallel(sim, freqs, lateral_bc, bbar, workers):
    """Fan the per-frequency solves across a spawn ProcessPoolExecutor (order preserved)."""
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
            initializer=_init_worker, initargs=(sim, lateral_bc, bbar),
        ) as ex:
            rows = list(ex.map(_worker_solve, freqs))  # map preserves input order
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old
    return np.array(rows)
