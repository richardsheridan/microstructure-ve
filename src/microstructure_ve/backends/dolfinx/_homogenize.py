"""Per-frequency homogenized response: volume-averaged stress and the readODB row.

Generic over the imposed macro strain (a ``MacroLoading`` from ``_loading``): the driven
strain components are imposed, any ``free`` components are solved so their conjugate
volume-averaged stress vanishes (the generalized free-lateral condition), and the reported
row is the reaction on the primary-axis face. Only the active modes (driven + free) are
solved; held components contribute nothing. The emitted row mirrors readODB's columns so the
ABAQUS-parity comparison can diff it.
"""
from __future__ import annotations

import numpy as np

import ufl
from dolfinx import fem

from ._assembly import eps


def build_solve_one(space, forms, solver, loading):
    """Return ``solve_one(f) -> [freq, RF_Real..., RF_Imag..., U...]`` for one frequency."""
    dim = space.dim
    area = space.area
    sig, unit_E, modes = forms.sig, forms.unit_E, forms.modes
    uh = solver.uh

    idx_of = {comp: k for k, comp in enumerate(modes)}
    active = list(loading.active)              # symmetric components (i<=j): driven + free
    active_idx = [idx_of[c] for c in active]
    a0 = loading.primary_axis
    imposed, free = loading.imposed, loading.free

    # Only the stress components actually read are assembled: the primary-axis column
    # sigma-bar[:, a0] (the reported reaction) and the free-component diagonals (the
    # zero-stress conditions). This is fewer forms than the full dim x dim tensor.
    needed_pairs = sorted({(n, a0) for n in range(dim)} | set(free))
    sbar_forms = {
        k: {ij: fem.form(sig(unit_E[k] + eps(uh[k]))[ij[0], ij[1]] * ufl.dx)
            for ij in needed_pairs}
        for k in active_idx
    }

    def solve_one(f, elastic=False):
        # elastic=True: a Static step -> real *Elastic moduli (zero loss); else the
        # frequency-domain complex moduli at f.
        if elastic:
            solver.reassemble_elastic()
        else:
            solver.reassemble(f)
        sbar = {}
        for k in active_idx:
            solver.solve(k)
            sbar[k] = {ij: fem.assemble_scalar(sbar_forms[k][ij]) / area for ij in needed_pairs}

        # macro-strain coefficient per active component: driven = imposed; free = unknown
        c = {comp: complex(imposed.get(comp, 0.0)) for comp in active}
        if free:
            # choose the free components so sigma-bar at each vanishes (e.g. lateral normals
            # under apparent-uniaxial loading)
            M = np.array([[sbar[idx_of[fc]][(fb[0], fb[0])] for fc in free] for fb in free])
            r = np.array([
                -sum(c[comp] * sbar[idx_of[comp]][(fb[0], fb[0])] for comp in imposed)
                for fb in free
            ])
            for comp, val in zip(free, np.linalg.solve(M, r)):
                c[comp] = val

        # combined homogenized traction on the primary-axis face: sigma-bar[:, a0]
        sig0 = np.array([
            sum(c[comp] * sbar[idx_of[comp]][(n, a0)] for comp in active) for n in range(dim)
        ])
        RF = sig0 * loading.cross_area
        U = np.zeros(dim)
        U[loading.primary_dof - 1] = loading.drive_value
        return [float(f)] + list(RF.real) + list(RF.imag) + list(U)

    return solve_one
