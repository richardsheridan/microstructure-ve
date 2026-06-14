"""Constraints for the periodic-homogenization solve.

A pure-periodic multi-point constraint on the fluctuation field (slave = each max-face
node, master = its min image, disjoint slaves) plus a single interior Dirichlet pin to
remove rigid-body translation. Imported only by ``run`` (lazily).
"""
from __future__ import annotations

import numpy as np

import dolfinx
from dolfinx import fem
import dolfinx_mpc

from . import _spec as spec


def periodic_mpc(space):
    """Build the pure-periodic ``MultiPointConstraint`` from the spec node pairs."""
    coord = space.coord
    sm = {
        coord(s).tobytes(): {coord(m).tobytes(): 1.0}
        for s, m in spec.periodic_pairs(space.shape)
    }
    mpc = dolfinx_mpc.MultiPointConstraint(space.V)
    for comp in range(space.dim):
        mpc.create_general_constraint(sm, comp, comp)
    mpc.finalize()
    return mpc


def center_pin(space):
    """Dirichlet-zero the interior centre node (all components) for rigid translation."""
    dim, shape, scale, V = space.dim, space.shape, space.scale, space.V
    center = np.array([(shape[dim - 1 - i] - 1) // 2 * scale for i in range(dim)])

    def at_center(x):
        ok = np.ones(x.shape[1], dtype=bool)
        for i in range(dim):
            ok &= np.isclose(x[i], center[i])
        return ok

    bcs = []
    for comp in range(dim):
        Vc, _ = V.sub(comp).collapse()
        dofs = dolfinx.fem.locate_dofs_geometrical((V.sub(comp), Vc), at_center)
        fbc = fem.Function(Vc)
        fbc.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(fbc, dofs, V.sub(comp)))
    return bcs
