"""DOLFINx boilerplate: the mesh, node<->dof map, DG0 Lame fields, and UFL forms.

Imports dolfinx/basix/ufl at module top; this module is only ever imported by
``_run`` (lazily), never under the numpy-only env. The mesh/dof state, material fields,
and forms are bundled into dataclasses (``Space``, ``MaterialFields``, ``Forms``) built
through classmethod constructors.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, List

import numpy as np

import basix.ufl
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem

from . import _spec as spec

_ELEMENT = {2: "quadrilateral", 3: "hexahedron"}


def eps(w):
    return ufl.sym(ufl.grad(w))


@dataclass(eq=False)
class Space:
    """The FE mesh, vector P1 space, and the structured node<->dof maps."""

    mesh: Any
    V: Any
    dim: int
    scale: float
    shape: tuple
    dof_x: np.ndarray
    block_of_node: np.ndarray
    oci: np.ndarray
    area: float

    @classmethod
    def build(cls, geom):
        dim, scale, shape = geom.dim, geom.scale, geom.shape

        # tensor-product (basix lexicographic) cell ordering, same in 2D and 3D.
        # [::-1] reverses the [Z, Y, X] index axes into physical (x, y, z); see Sides_3d in core.
        coords = np.column_stack([(scale * c).ravel() for c in np.indices(shape)[::-1]])
        all_nodes = 1 + np.ravel_multi_index(np.indices(shape), shape)
        slices = list(product((np.s_[:-1], np.s_[1:]), repeat=dim))
        cells = np.column_stack([all_nodes[sl].ravel() for sl in slices]) - 1
        c_el = ufl.Mesh(basix.ufl.element("Lagrange", _ELEMENT[dim], 1, shape=(dim,)))
        mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, c_el, coords)  # 0.10: (…, e, x)

        V = fem.functionspace(mesh, ("Lagrange", 1, (dim,)))
        dof_x = V.tabulate_dof_coordinates()[:, :dim]
        grid = np.rint(dof_x / scale).astype(int)  # columns: (i_x, i_y, [i_z])
        ravel_idx = tuple(grid[:, dim - 1 - j] for j in range(dim))  # back to grid-axis order
        ours_of_block = 1 + np.ravel_multi_index(ravel_idx, shape)
        block_of_node = np.empty(int(np.prod(shape)) + 1, dtype=int)
        block_of_node[ours_of_block] = np.arange(len(ours_of_block))

        area = fem.assemble_scalar(
            fem.form(fem.Constant(mesh, PETSc.ScalarType(1.0)) * ufl.dx)
        ).real

        return cls(mesh=mesh, V=V, dim=dim, scale=scale, shape=shape,
                   dof_x=dof_x, block_of_node=block_of_node,
                   oci=mesh.topology.original_cell_index, area=area)

    def coord(self, node):
        """Coordinates of the 1-indexed grid node (exact on the structured grid)."""
        return self.dof_x[self.block_of_node[int(node)]].astype(np.float64)


@dataclass(eq=False)
class MaterialFields:
    """DG0 complex Lame fields (mu, lambda) and the per-frequency refill."""

    mu_fn: Any
    lam_fn: Any
    mat_of_cell: np.ndarray
    nu_cell: np.ndarray
    youngs_cell: np.ndarray
    oci: np.ndarray
    modulus_fns: List[Callable]

    @classmethod
    def from_model(cls, space, model):
        DG0 = fem.functionspace(space.mesh, ("DG", 0))
        self = cls(fem.Function(DG0), fem.Function(DG0),
                   None, None, None, space.oci, None)
        self.update_materials(model)
        return self

    def update_materials(self, model):
        """Re-bind the per-cell material mapping (poisson/youngs/modulus per cell) to a new
        model on the same mesh. The DG0 ``mu_fn``/``lam_fn`` Functions (and the forms built
        on them) are unchanged, so a cached solver can be reused across cells that share the
        mesh -- only the values differ, refilled by ``set_moduli`` per frequency."""
        mat_of_cell, poissons, youngs, modulus_fns = spec.material_cell_maps(model)
        self.mat_of_cell = mat_of_cell
        self.nu_cell = poissons[mat_of_cell]
        self.youngs_cell = youngs[mat_of_cell]
        self.modulus_fns = modulus_fns

    def _fill(self, E_cell):
        mu = E_cell / (2 * (1 + self.nu_cell))
        lam = E_cell * self.nu_cell / ((1 + self.nu_cell) * (1 - 2 * self.nu_cell))
        self.mu_fn.x.array[:] = mu[self.oci]
        self.lam_fn.x.array[:] = lam[self.oci]

    def set_moduli(self, f):
        """Fill mu/lambda from each material's complex_modulus at frequency ``f``.

        Per-cell modulus is assigned through ``original_cell_index`` (our cell order is
        the raveled pixel order, which dolfinx reorders). 3D / plane-strain Lame.
        """
        E_cell = np.empty(len(self.mat_of_cell), dtype=complex)
        for mi, mod in enumerate(self.modulus_fns):
            E_cell[self.mat_of_cell == mi] = complex(np.asarray(mod(np.array([f]))).ravel()[0])
        self._fill(E_cell)

    def set_moduli_elastic(self):
        """Fill mu/lambda from each material's real ``*Elastic`` modulus (``youngs``).

        This is what a ``Static`` step uses: the frequency-domain ``*Viscoelastic`` table
        applies only to steady-state dynamics, so a static step sees the real long-term
        modulus and produces zero loss. Used for the static steps of multi-step sims.
        """
        self._fill(self.youngs_cell.astype(complex))


@dataclass(eq=False)
class Forms:
    """Bilinear form ``a``, the symmetric unit-strain basis, per-mode RHS forms, and ``sig``.

    ``modes`` is the ordered list of symmetric macro-strain components ``(i, j)`` with
    ``i <= j``: the ``dim`` normals first, then the off-diagonal shears -- 3 in 2D
    ((0,0),(1,1),(0,1)), 6 in 3D. ``unit_E[k]`` is the unit tensor for ``modes[k]`` (normal:
    a 1 on the diagonal; shear: 1 in *both* off-diagonal slots, so its coefficient is the
    tensor strain component ``E_ij``), and ``L_forms[k]`` its RHS source. The homogenizer
    selects the active subset per loading; building the full basis here keeps the forms
    mode-agnostic (shear modes are ready without touching this file).
    """

    a_form: Any
    L_forms: List[Any]
    unit_E: List[Any]
    modes: List[tuple]
    sig: Callable
    v: Any

    @classmethod
    def build(cls, space, matfields, bbar):
        """``a`` uses B-bar (selective reduced integration on the volumetric term) when
        ``bbar`` is set, matching ABAQUS's CPE4/C3D8 B-bar formulation. ``sig`` (full
        pointwise stress) is returned for the RHS source and stress recovery.
        """
        V, dim = space.V, space.dim
        mu_fn, lam_fn = matfields.mu_fn, matfields.lam_fn
        eye = ufl.Identity(dim)

        def dev(e):
            return e - ufl.tr(e) / dim * eye

        def sig(e):  # B-bar-consistent (tr(eps) is linear) full pointwise stress
            return 2 * mu_fn * e + lam_fn * ufl.tr(e) * eye

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        if bbar:
            dxf = ufl.dx(metadata={"quadrature_degree": 2})
            dxr = ufl.dx(metadata={"quadrature_degree": 1})  # 1-point (centroid) for quad/hex
            kappa = lam_fn + 2 * mu_fn / dim
            a = (2 * mu_fn * ufl.inner(dev(eps(u)), dev(eps(v)))) * dxf \
                + kappa * ufl.inner(ufl.tr(eps(u)), ufl.tr(eps(v))) * dxr
        else:
            a = ufl.inner(sig(eps(u)), eps(v)) * ufl.dx
        a_form = fem.form(a)

        # full symmetric basis: normals (i,i) then shears (i,j), i<j
        modes = [(i, i) for i in range(dim)]
        modes += [(i, j) for i in range(dim) for j in range(i + 1, dim)]
        unit_E, L_forms = [], []
        for i, j in modes:
            Ei = np.zeros((dim, dim))
            Ei[i, j] = 1.0
            Ei[j, i] = 1.0  # symmetric: shear sets both slots; normal sets the diagonal once
            Em = ufl.as_matrix(Ei.tolist())
            unit_E.append(Em)
            L_forms.append(fem.form(-ufl.inner(sig(Em), eps(v)) * ufl.dx))

        return cls(a_form=a_form, L_forms=L_forms, unit_E=unit_E, modes=modes,
                   sig=sig, v=v)
