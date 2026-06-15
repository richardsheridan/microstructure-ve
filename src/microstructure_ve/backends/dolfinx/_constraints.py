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
    """Build the pure-periodic ``MultiPointConstraint`` from the spec node pairs.

    Each ``(slave_node, master_node)`` pair from ``spec.periodic_pairs`` ties every
    component of the slave to the same component of its min-face image (coeff 1). We feed
    the dof arrays straight to ``add_constraint`` using the structured ``block_of_node`` map
    instead of routing coordinates through ``create_general_constraint`` -- the latter
    re-locates each coordinate geometrically (a per-pair ``sub().collapse()``), which is the
    dominant cost on 3D meshes. Same constraint, ~100x cheaper to assemble. Serial only,
    matching the backend's serial LU path: masters are owned by this (sole) rank.
    """
    V = space.V
    bs = V.dofmap.index_map_bs
    block_of_node = space.block_of_node
    pairs = spec.periodic_pairs(space.shape)
    s_blocks = np.array([block_of_node[s] for s, _ in pairs], dtype=np.int32)
    m_blocks = np.array([block_of_node[m] for _, m in pairs], dtype=np.int32)
    m_global = V.dofmap.index_map.local_to_global(m_blocks)  # int64 (== local in serial)

    comps = np.arange(bs, dtype=np.int64)
    slaves = (s_blocks[:, None].astype(np.int64) * bs + comps).ravel().astype(np.int32)
    masters = (m_global[:, None] * bs + comps).ravel().astype(np.int64)
    n = slaves.size
    coeffs = np.ones(n, dtype=dolfinx.default_scalar_type)
    owners = np.zeros(n, dtype=np.int32)            # serial: master owned by rank 0
    offsets = np.arange(n + 1, dtype=np.int32)      # exactly one master per slave

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.add_constraint(V, slaves, masters, coeffs, owners, offsets)
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
