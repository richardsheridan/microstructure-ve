"""Per-frequency homogenized response: volume-averaged stress and the readODB row.

Confined loading prescribes the lateral macro normal strains to 0; free-lateral floats
them so the lateral homogenized normal stresses vanish, solved by superposition of the
per-axis unit-strain solves (same stiffness -> the extra solves are back-substitutions).
The emitted row mirrors readODB's columns so ``verify_pbc.compare`` can diff it.
"""
from __future__ import annotations

import numpy as np

import ufl
from dolfinx import fem

from ._assembly import eps


def build_solve_one(space, forms, solver, geom, lateral_bc):
    """Return ``solve_one(f) -> [freq, RF_Real..., RF_Imag..., U...]`` for one frequency."""
    if lateral_bc not in ("confined", "free"):
        raise ValueError("lateral_bc must be 'confined' or 'free'")
    dim = space.dim
    sig, unit_E, uh = forms.sig, forms.unit_E, solver.uh
    exx, Lx, cross_area, area = geom.exx, geom.Lx, geom.cross_area, space.area

    # full sigma-bar tensor per unit strain (averaged over the cell)
    pairs_ij = [(m, n) for m in range(dim) for n in range(dim)]
    sbar_forms = [
        {ij: fem.form(sig(unit_E[j] + eps(uh[j]))[ij[0], ij[1]] * ufl.dx) for ij in pairs_ij}
        for j in range(dim)
    ]

    def solve_one(f):
        solver.reassemble(f)
        nsolve = dim if lateral_bc == "free" else 1
        sbar = []  # sbar[j][(m,n)] = unit-strain-j homogenized stress component
        for j in range(nsolve):
            solver.solve(j)
            sbar.append({ij: fem.assemble_scalar(sbar_forms[j][ij]) / area for ij in pairs_ij})

        e = np.zeros(dim, dtype=complex)
        e[0] = exx
        if lateral_bc == "free" and dim > 1:
            # choose lateral normal strains so sigma-bar_ii = 0 for i = 1..dim-1
            M = np.array([[sbar[k][(i, i)] for k in range(1, dim)] for i in range(1, dim)])
            r = np.array([-exx * sbar[0][(i, i)] for i in range(1, dim)])
            e[1:] = np.linalg.solve(M, r)
        # combined homogenized x-face traction: sigma-bar_{0,n} = sum_j e_j sbar[j][(0,n)]
        sig0 = np.array(
            [sum(e[j] * sbar[j][(0, n)] for j in range(len(sbar))) for n in range(dim)]
        )
        RF = sig0 * cross_area  # x-face reaction (== readODB RF at the x drive node)
        U = np.zeros(dim)
        U[0] = exx * Lx
        return [float(f)] + list(RF.real) + list(RF.imag) + list(U)

    return solve_one
