"""Orchestration: wire the concern submodules into a solver and sweep the frequencies.

``build_solver`` assembles the spec/assembly/constraints/solver/homogenize pieces into a
``solve_one(f)`` closure (all dolfinx/PETSc state captured in it). ``run`` sweeps the
frequencies and optionally writes a readODB-style tsv. The macro loading is uniaxial
along x (coordinate axis 0). Frequency-parallel execution is added in a later step.
"""
from __future__ import annotations

import numpy as np

from . import assembly, constraints, homogenize, spec
from .solver import Solver


def build_solver(sim, lateral="confined", bbar=True):
    """Build the FE solver for ``sim`` once; return ``(solve_one, dim)``.

    ``solve_one(f)`` returns the readODB-style row for one frequency. The setup (mesh,
    dof map, materials, forms, MPC, factorizable matrix) is paid once and reused across
    frequencies and both unit-strain RHS.
    """
    model = sim.model
    spec.require_periodic(model)
    geom = spec.geometry(model, sim)

    space = assembly.build_space(geom)
    matfields = assembly.material_fields(space, model)
    forms = assembly.elasticity_forms(space, matfields, bbar)
    mpc = constraints.periodic_mpc(space)
    bcs = constraints.center_pin(space)
    solver = Solver(space, forms, matfields, mpc, bcs)
    solve_one = homogenize.build_solve_one(space, forms, solver, geom, lateral)
    return solve_one, space.dim


def _row_header(dim):
    return (
        ["frequency"]
        + [f"RF_Real{i + 1}" for i in range(dim)]
        + [f"RF_Imag{i + 1}" for i in range(dim)]
        + [f"U{i + 1}" for i in range(dim)]
    )


def run(sim, freqs=None, output_path=None, lateral="confined", bbar=True, workers=1):
    """Solve ``sim`` over ``freqs``; return a readODB-style array ``(len(freqs), 1+3*dim)``.

    lateral: "confined" (lateral normal strains = 0) or "free" (lateral sigma-bar normals = 0).
    bbar:    selective reduced integration on the volumetric term (recommended).
    workers: serial for now; frequency-parallel execution is added later.
    """
    if freqs is None:
        freqs = spec.default_frequencies(sim)
    freqs = np.asarray(freqs, dtype=float)
    dim = sim.model.nodes.dim

    solve_one, _ = build_solver(sim, lateral, bbar)
    out = np.array([solve_one(f) for f in freqs])

    if output_path is not None:
        np.savetxt(
            output_path, out, fmt="%.8e", delimiter="\t",
            header="\t".join(_row_header(dim)), comments="",
        )
    return out
