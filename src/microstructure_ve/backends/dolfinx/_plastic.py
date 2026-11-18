"""Small-strain J2 (von Mises) isotropic-hardening plasticity for ``Static`` steps.

The linear backend (``_homogenize``) superposes per-unit-macro-strain solves, valid only for a
linear material. A ``Plastic`` response is history/path-dependent, so a Static step carrying one
needs a genuine nonlinear solve. This module provides it for the periodic, corner-driven
homogenization in **2D (plane strain) and 3D**, including multi-step sims where plastic state
persists across consecutive plastic Static steps (load/reverse hysteresis).

Design constraints (docs/dolfinx_backend_design.md, section 12):

* **Complex-scalar PETSc build** -- UFL yield conditionals on complex scalars are ill-defined,
  so the radial return runs in **numpy at the quadrature points** (real arithmetic) and is fed
  back as Quadrature-space coefficients: the stress ``sig_q`` (residual) and algorithmic tangent
  ``C_q`` (Jacobian). Assembled complex matrices/vectors carry real values.
* **Hand-rolled Newton** (``_nonlinear.modified_newton``) over the ``dolfinx_mpc`` assemble +
  complex-LU path; ``dolfinx_mpc.NonlinearProblem``/SNES is skipped in complex builds and would not
  let us refresh the return-map coefficients between iterations.
* **B-bar** (mean-dilatation) on the strain removes Q1/Q1-hex volumetric locking from J2's
  near-isochoric plastic flow, matching the elastic path's selective reduced integration.
* Quadrature degree 2 -> 4 Gauss points on a quad (CPE4) / 8 on a hex (C3D8).

The macro strain ``E`` enters through the *total* strain ``eps(E.x + u~) = E + eps(u~)`` (the
fluctuation formulation the linear path also uses); driven components are imposed and any ``free``
lateral components are found by the outer root-find of ``_nonlinear.FreeLateralRamp`` so their
volume-averaged stress vanishes. Reported row mirrors the readODB columns:
``[1.0, RF_Real..., 0..., U...]`` (zero loss).
"""
from __future__ import annotations

import numpy as np

import basix
import basix.ufl
import ufl
from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc as fempetsc
import dolfinx_mpc

from microstructure_ve.constitutive import Plastic

from ._nonlinear import FreeLateralRamp, face_row, modified_newton, periodic_row
from ._solver import ConvergenceError, lu_ksp

_QUAD_DEGREE = 2          # 2x2(x2) Gauss = ABAQUS CPE4 (4 pts) / C3D8 (8 pts)

# tensor (not engineering) component order used internally: [xx, yy, zz, xy, yz, xz]
_SHEAR_PAIRS = {2: [(0, 1)], 3: [(0, 1), (1, 2), (0, 2)]}


# --------------------------------------------------------------------------- constitutive
def _hardening_table(material):
    """``(plastic_strain, yield_stress)`` arrays for the piecewise-linear ``sigma_y(p)``.

    A non-plastic (elastic) phase gets a single infinite yield so it never returns plastic.
    """
    if isinstance(material.response, Plastic):
        return (np.asarray(material.response.plastic_strain, dtype=float),
                np.asarray(material.response.yield_stress, dtype=float))
    return np.array([0.0]), np.array([np.inf])


def _sigma_y(p, eps_tab, sig_tab):
    """Piecewise-linear yield stress and slope ``H = d sigma_y / d p`` at ``p`` (constant
    extrapolation past the last table point -> ``H = 0``, matching ABAQUS's ``*Plastic``)."""
    sy = np.interp(p, eps_tab, sig_tab)
    if len(eps_tab) == 1:
        return sy, np.zeros_like(p)
    seg_H = np.diff(sig_tab) / np.diff(eps_tab)
    idx = np.clip(np.searchsorted(eps_tab, p, side="right") - 1, 0, len(eps_tab) - 2)
    H = seg_H[idx]
    H[p >= eps_tab[-1]] = 0.0
    return sy, H


def _voigt_to_tensor_strain(eps_v, dim):
    """Engineering Voigt strain -> full (N,6) tensor strain ``[xx,yy,zz,xy,yz,xz]`` (tensor
    shear = engineering/2). 2D is plane strain (``ezz = eyz = exz = 0``)."""
    N = eps_v.shape[0]
    e = np.zeros((N, 6))
    if dim == 2:
        e[:, 0], e[:, 1] = eps_v[:, 0], eps_v[:, 1]
        e[:, 3] = eps_v[:, 2] / 2.0
    else:
        e[:, 0], e[:, 1], e[:, 2] = eps_v[:, 0], eps_v[:, 1], eps_v[:, 2]
        e[:, 3], e[:, 4], e[:, 5] = eps_v[:, 3] / 2.0, eps_v[:, 4] / 2.0, eps_v[:, 5] / 2.0
    return e


def _tensor_stress_to_voigt(s, dim):
    """Full (N,6) tensor stress -> dim Voigt ``[xx,yy,xy]`` (2D) or ``[xx,yy,zz,xy,yz,xz]``."""
    if dim == 2:
        return np.stack([s[:, 0], s[:, 1], s[:, 3]], axis=1)
    return s


def _return_map(eps_v, mu, lam, eps_p_old, p_old, mat_id, materials, dim):
    """Radial-return a vectorized Gauss-point batch (pure).

    Parameters
    ----------
    eps_v : numpy.ndarray
        ``(N, nv)`` total engineering Voigt strain (nv=3 in 2D, 6 in 3D).
    mu, lam : numpy.ndarray
        ``(N,)`` Lame moduli.
    eps_p_old : numpy.ndarray
        ``(N, 6)`` committed tensor plastic strain.
    p_old : numpy.ndarray
        ``(N,)`` committed equivalent plastic strain.
    mat_id : numpy.ndarray
        ``(N,)`` material index of each point into ``materials``.
    materials : list
        The model's materials; a non-``Plastic`` one never yields.
    dim : int
        2 (plane strain) or 3.

    Returns
    -------
    sig_v : numpy.ndarray
        ``(N, nv)`` Voigt stress.
    eps_p_new : numpy.ndarray
        ``(N, 6)`` updated tensor plastic strain.
    p_new : numpy.ndarray
        ``(N,)`` updated equivalent plastic strain.
    """
    return _radial_return(eps_v, mu, lam, eps_p_old, p_old, mat_id, materials, dim)[:3]


def _radial_return(eps_v, mu, lam, eps_p_old, p_old, mat_id, materials, dim):
    """``_return_map``'s products followed by the trial deviator ``d`` (N,6), its von Mises
    norm ``q``, the increment ``dp`` and the hardening slope ``H`` at ``p_old + dp``."""
    e = _voigt_to_tensor_strain(eps_v, dim)
    ee = e - eps_p_old
    tr = ee[:, 0] + ee[:, 1] + ee[:, 2]
    sig = np.empty_like(ee)
    for k in range(3):
        sig[:, k] = lam * tr + 2 * mu * ee[:, k]
    for k in (3, 4, 5):
        sig[:, k] = 2 * mu * ee[:, k]
    mean = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    d = sig.copy()
    d[:, 0] -= mean
    d[:, 1] -= mean
    d[:, 2] -= mean
    q = np.sqrt(np.maximum(
        1.5 * (d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2
               + 2 * (d[:, 3]**2 + d[:, 4]**2 + d[:, 5]**2)), 0.0))

    dp = np.zeros_like(q)
    slope = np.zeros_like(q)
    for i, mat in enumerate(materials):
        m = mat_id == i
        if not np.any(m):
            continue
        eps_tab, sig_tab = _hardening_table(mat)
        sy0, _ = _sigma_y(p_old[m], eps_tab, sig_tab)
        plastic = q[m] > sy0
        if not np.any(plastic):
            continue
        idx = np.where(m)[0][plastic]
        qb, mub, pb = q[idx], mu[idx], p_old[idx]
        dpb = np.zeros_like(qb)
        for _ in range(50):  # local scalar Newton on the consistency condition
            sy, H = _sigma_y(pb + dpb, eps_tab, sig_tab)
            g = qb - 3 * mub * dpb - sy
            dpb -= g / (-3 * mub - H)
            if np.all(np.abs(g) <= 1e-12 * (float(sig_tab[0]) + 1.0)):
                break
        dp[idx] = dpb
        slope[idx] = _sigma_y(pb + dpb, eps_tab, sig_tab)[1]

    yielded = dp > 0
    safe_q = np.where(q > 0, q, 1.0)
    scale = np.where(yielded, 1.0 - 3 * mu * dp / safe_q, 1.0)
    sig_new = d * scale[:, None]
    sig_new[:, 0] += mean
    sig_new[:, 1] += mean
    sig_new[:, 2] += mean
    fac = np.where(yielded, 1.5 * dp / safe_q, 0.0)        # d eps_p = (3/2)(s/q) dp
    eps_p_new = eps_p_old + fac[:, None] * d
    return _tensor_stress_to_voigt(sig_new, dim), eps_p_new, p_old + dp, d, q, dp, slope


# tensor index pair (i, j) of each Voigt component, in the order of ``_tensor_stress_to_voigt``
_VOIGT_PAIRS = {2: [(0, 0), (1, 1), (0, 1)],
                3: [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]}
_TENSOR_INDEX = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (1, 2): 4, (0, 2): 5}


def _tangent(eps_v, mu, lam, eps_p_old, p_old, mat_id, materials, dim):
    """Consistent algorithmic tangent of the radial return with the return-map products.

    Returns
    -------
    C : numpy.ndarray
        ``(N, nv, nv)`` derivative of the Voigt stress with respect to the engineering Voigt
        strain. A point that does not yield gets the elastic stiffness of its ``mu`` and
        ``lam`` alone, so its tangent is identical between evaluations.
    sig_v, eps_p_new, p_new : numpy.ndarray
        ``_return_map``'s products at ``eps_v``.
    """
    sig_v, eps_p_new, p_new, d, q, dp, H = _radial_return(eps_v, mu, lam, eps_p_old, p_old,
                                                          mat_id, materials, dim)
    pairs = _VOIGT_PAIRS[dim]
    nv = len(pairs)
    C = np.zeros((eps_v.shape[0], nv, nv))
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            # lam d_ij d_kl + mu (d_ik d_jl + d_il d_jk); an engineering shear strain carries
            # both e_kl and e_lk, so the column of a shear component is C_ijkl itself
            C[:, a, b] = (lam * (i == j) * (k == l)
                          + mu * ((i == k) * (j == l) + (i == l) * (j == k)))
    yielded = dp > 0
    if np.any(yielded):
        y = np.where(yielded)[0]
        mu_y, q_y, dp_y, H_y = mu[y], q[y], dp[y], H[y]
        theta = 1.0 - 3 * mu_y * dp_y / q_y
        theta_bar = 1.0 / (1.0 + H_y / (3 * mu_y)) - (1.0 - theta)
        n = d[y] * (np.sqrt(1.5) / q_y)[:, None]     # unit deviator: |n| = 1
        for a, (i, j) in enumerate(pairs):
            na = n[:, _TENSOR_INDEX[(i, j)]]
            for b, (k, l) in enumerate(pairs):
                nb = n[:, _TENSOR_INDEX[(k, l)]]
                dev = 0.5 * ((i == k) * (j == l) + (i == l) * (j == k)) - (i == j) * (k == l) / 3.0
                C[y, a, b] -= 2 * mu_y * ((1.0 - theta) * dev + theta_bar * na * nb)
    return C, sig_v, eps_p_new, p_new


# --------------------------------------------------------------------------- FE driver
def _eps_voigt(w, dim):
    """Engineering Voigt strain of a displacement field (dim-aware)."""
    if dim == 2:
        return ufl.as_vector([w[0].dx(0), w[1].dx(1), w[0].dx(1) + w[1].dx(0)])
    return ufl.as_vector([
        w[0].dx(0), w[1].dx(1), w[2].dx(2),
        w[0].dx(1) + w[1].dx(0), w[1].dx(2) + w[2].dx(1), w[0].dx(2) + w[2].dx(0),
    ])


def _voigt_stress_tensor(sig_v, dim):
    """dim Voigt stress -> (dim,dim) tensor."""
    if dim == 2:
        return np.array([[sig_v[0], sig_v[2]], [sig_v[2], sig_v[1]]])
    return np.array([[sig_v[0], sig_v[3], sig_v[5]],
                     [sig_v[3], sig_v[1], sig_v[4]],
                     [sig_v[5], sig_v[4], sig_v[2]]])


def _voigt_vol(ev, dim):
    """Volumetric part of an engineering-Voigt strain vector (the in-plane trace in 2D,
    matching the B-bar correction applied to the residual strain in ``_total_strain``)."""
    t = sum(ev[i] for i in range(dim)) / dim
    if dim == 2:
        return ufl.as_vector([t, t, 0])
    return ufl.as_vector([t, t, t, 0, 0, 0])


def _bbar_jacobian_form(V, dim, qd, C_q, C_c):
    """The B-bar-consistent Jacobian form.

    The residual evaluates stress at the B-bar strain (volumetric part replaced by its cell mean),
    so its exact derivative acts on ``dev eps(u) + mean_cell(vol eps(u))`` -- not on the raw
    ``eps(u)``. An inconsistent Jacobian costs Newton its quadratic convergence
    (tools/profile_plastic.py). The dev part integrates at the full rule against the
    quadrature-point tangent ``C_q``; the cell-mean vol part reduces to the 1-point (centroid) rule
    against the cell-averaged tangent ``C_c`` (exact for a cell-constant tangent on the rectangular
    Q1 grid, where the centroid value equals the cell mean). The form is slightly nonsymmetric (test
    side stays raw, as in the residual); LU doesn't care.
    """
    u_tr, v_te = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx_q = ufl.dx(metadata={"quadrature_degree": qd, "quadrature_scheme": "default"})
    dx_1 = ufl.dx(metadata={"quadrature_degree": 1, "quadrature_scheme": "default"})
    ev_u, ev_v = _eps_voigt(u_tr, dim), _eps_voigt(v_te, dim)
    vol_u = _voigt_vol(ev_u, dim)
    return fem.form(ufl.inner(ufl.dot(C_q, ev_u - vol_u), ev_v) * dx_q
                    + ufl.inner(ufl.dot(C_c, vol_u), ev_v) * dx_1)


class _J2Solver:
    """The setup and Newton plumbing the periodic and standard plastic solvers share.

    Builds the quadrature, the per-point moduli and committed plastic state, the Quadrature-space
    coefficients ``sig_q``/``C_q``/``C_c`` and the residual and B-bar Jacobian forms over
    ``self.u``. A subclass assembles ``self.A`` and implements ``_assemble_jacobian``."""

    def __init__(self, prob, model):
        space = prob.space
        if space.dim not in (2, 3):
            raise NotImplementedError("dolfinx plasticity supports 2D/3D only")
        self.space = space
        self.dim = space.dim
        self.nv = 3 if self.dim == 2 else 6
        self.mesh = space.mesh
        self.V = space.V
        self.materials = list(model.materials)

        qd = _QUAD_DEGREE
        self.points, self.weights = basix.make_quadrature(self.mesh.basix_cell(), qd)
        self.npts = len(self.weights)
        self.ncells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        N = self.ncells * self.npts

        mf = prob.matfields
        oci = space.oci
        E = np.asarray(mf.youngs_cell, dtype=complex).real[oci].astype(float)
        nu = np.asarray(mf.nu_cell, dtype=complex).real[oci].astype(float)
        self.mu = np.repeat(E / (2 * (1 + nu)), self.npts)
        self.lam = np.repeat(E * nu / ((1 + nu) * (1 - 2 * nu)), self.npts)
        self.mat_id = np.repeat(np.asarray(mf.mat_of_cell)[oci].astype(int), self.npts)

        cellname = self.mesh.topology.cell_name()
        Qv = basix.ufl.quadrature_element(cellname, value_shape=(self.nv,), degree=qd)
        Qt = basix.ufl.quadrature_element(cellname, value_shape=(self.nv, self.nv), degree=qd)
        Qt1 = basix.ufl.quadrature_element(cellname, value_shape=(self.nv, self.nv), degree=1)
        self.sig_q = fem.Function(fem.functionspace(self.mesh, Qv))
        self.C_q = fem.Function(fem.functionspace(self.mesh, Qt))
        self.C_c = fem.Function(fem.functionspace(self.mesh, Qt1))  # cell-mean tangent

        # the periodic fluctuation u~ or the standard path's full displacement
        self.u = fem.Function(self.V)
        self.eps_expr = fem.Expression(_eps_voigt(self.u, self.dim), self.points)
        self._cells = np.arange(self.ncells, dtype=np.int32)

        v_te = ufl.TestFunction(self.V)
        dx_q = ufl.dx(metadata={"quadrature_degree": qd, "quadrature_scheme": "default"})
        ev_v = _eps_voigt(v_te, self.dim)
        self.a_form = _bbar_jacobian_form(self.V, self.dim, qd, self.C_q, self.C_c)
        self.L_form = fem.form(ufl.inner(self.sig_q, ev_v) * dx_q)

        # committed state (persists across steps)
        self.eps_p = np.zeros((N, 6))
        self.p = np.zeros(N)

        self._fill_coeffs(np.zeros((N, self.nv)), commit=False)  # elastic C_q to allocate A
        self._C_assembled = self.C_q.x.array.copy()  # tangent the assembled A is built from

    # -- coefficient refresh -------------------------------------------------
    def _total_strain(self, E_voigt=None):
        """Total engineering strain at the quad points: ``E + eps(u)`` (N,nv), real, with the
        mean-dilatation B-bar (volumetric strain replaced by its per-cell mean; for rectangular
        Q1/hex the cell mean == centroid value, i.e. the standard B-bar). ``E_voigt`` is the
        periodic path's macro strain; without it the strain is ``eps(u)``."""
        ev = self.eps_expr.eval(self.mesh, self._cells)  # (ncells, npts, nv), complex
        eps = np.asarray(ev).real
        if E_voigt is not None:
            eps = eps + E_voigt
        vol = eps[:, :, :self.dim].sum(axis=2)           # volumetric strain (sum of normals)
        w = self.weights
        vbar = (vol * w).sum(axis=1, keepdims=True) / w.sum()
        corr = (vbar - vol) / self.dim
        for k in range(self.dim):
            eps[:, :, k] += corr
        return eps.reshape(-1, self.nv)

    def _fill_coeffs(self, eps_total, commit, tangent=True):
        """Refresh the residual stress ``sig_q`` (always) and, when ``tangent``, the consistent
        algorithmic tangent ``C_q``/``C_c``. Returns the stress (N,nv)."""
        if tangent:
            C, sig, eps_p_new, p_new = _tangent(eps_total, self.mu, self.lam, self.eps_p,
                                                self.p, self.mat_id, self.materials, self.dim)
            self.C_q.x.array[:] = C.reshape(-1)
            w = self.weights
            Cc = (C.reshape(self.ncells, self.npts, self.nv, self.nv)
                  * w[None, :, None, None]).sum(axis=1) / w.sum()
            self.C_c.x.array[:] = Cc.reshape(-1)
        else:
            sig, eps_p_new, p_new = _return_map(eps_total, self.mu, self.lam, self.eps_p,
                                                self.p, self.mat_id, self.materials, self.dim)
        self.sig_q.x.array[:] = sig.reshape(-1)
        if commit:
            self.eps_p, self.p = eps_p_new, p_new
        return sig

    def _refresh_tangent(self):
        """``modified_newton``'s Jacobian refresh: the tangent at the last residual's strain,
        reassembled only when it changed."""
        self._fill_coeffs(self._eps_total, commit=False, tangent=True)
        if not np.array_equal(self.C_q.x.array, self._C_assembled):
            self.A.zeroEntries()
            self._assemble_jacobian()
            self.A.assemble()  # values-only refill; PETSc refactors on the next solve
            self.ksp.setOperators(self.A)
            self._C_assembled = self.C_q.x.array.copy()


class _PlasticSolver(_J2Solver):
    """Newton over the periodic MPC + complex LU, with numpy return mapping.

    Persistent across a sim's plastic Static steps: ``solve`` ramps the macro strain from the
    previously committed ``E_current`` to the new step target (so reversal/cyclic patterns
    accumulate plastic state)."""

    def __init__(self, prob, model):
        super().__init__(prob, model)
        self.mpc = prob.mpc
        self.bcs = prob.center_bcs
        self.E_current = np.zeros(self.nv)
        self._shear_pairs = _SHEAR_PAIRS[self.dim]
        self._ramp = FreeLateralRamp(lambda axis: axis)
        self._r_ref = 0.0  # largest initial Newton residual seen (convergence reference)

        self.A = dolfinx_mpc.assemble_matrix(self.a_form, self.mpc, bcs=self.bcs)
        self.A.assemble()
        self.b = dolfinx_mpc.assemble_vector(self.L_form, self.mpc)
        self.x = self.A.createVecRight()
        self.ksp = lu_ksp(self.A, self.mesh.comm)

    def _assemble_jacobian(self):
        dolfinx_mpc.assemble_matrix(self.a_form, self.mpc, bcs=self.bcs, A=self.A)

    def _average(self, sig_v):
        """Volume-averaged stress tensor (dim,dim) from a (N,nv) Voigt field (uniform grid)."""
        w = np.tile(self.weights, self.ncells)
        denom = self.ncells * self.weights.sum()
        sbar_v = (w[:, None] * sig_v).sum(axis=0) / denom
        return _voigt_stress_tensor(sbar_v, self.dim)

    def _average_stress(self, E_voigt):
        """Volume-averaged stress (dim,dim) of the current fluctuation at ``E_voigt``."""
        return self._average(self._fill_coeffs(self._total_strain(E_voigt), commit=False,
                                               tangent=False))

    # -- one Newton solve at a fixed macro strain ----------------------------
    def _newton(self, E_voigt):
        """Equilibrate the fluctuation at macro strain ``E_voigt`` by ``modified_newton``, judged
        against the largest initial residual this solver has seen. Returns whether the residual met
        its tolerance."""
        def residual():
            self._eps_total = self._total_strain(E_voigt)
            self._fill_coeffs(self._eps_total, commit=False, tangent=False)
            with self.b.localForm() as bl:
                bl.set(0.0)
            dolfinx_mpc.assemble_vector(self.L_form, self.mpc, self.b)
            dolfinx_mpc.apply_lifting(self.b, [self.a_form], [self.bcs], self.mpc)
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fempetsc.set_bc(self.b, self.bcs)
            self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            return self.b

        converged, self._r_ref = modified_newton(residual, self._refresh_tangent, self.ksp,
                                                 self.x, self.u, mpc=self.mpc, r_ref=self._r_ref)
        return converged

    # -- public entry --------------------------------------------------------
    def _target_voigt(self, loading):
        """The step's target macro strain (engineering Voigt): driven components imposed, the
        rest carried from the current committed value (held normals stay 0; free updated below)."""
        target = self.E_current.copy()
        for (i, j), val in loading.imposed.items():
            if i == j:
                target[i] = val
            else:
                target[self.dim + self._shear_pairs.index((min(i, j), max(i, j)))] = 2.0 * val
        return target

    def solve(self, loading, n_incr):
        """Ramp the macro strain from the committed ``E_current`` to this step's target over
        ``n_incr`` increments (committing plastic state each), solving any ``free`` components so
        their average stress vanishes. Returns the readODB row and advances ``E_current``."""
        def commit(E):
            self._fill_coeffs(self._total_strain(E), commit=True, tangent=False)

        E = self._ramp.run(self.E_current, self._target_voigt(loading),
                           [b for (b, _) in loading.free], n_incr, self._newton,
                           self._average_stress, self.mu.mean() + self.lam.mean(), commit)
        self.E_current = E.copy()
        sbar = self._average(_return_map(self._total_strain(E), self.mu, self.lam, self.eps_p,
                                         self.p, self.mat_id, self.materials, self.dim)[0])
        return periodic_row(loading, sbar)


class _StandardPlasticSolver(_J2Solver):
    """J2 plasticity for the standard (non-periodic) path: direct face Dirichlet, no MPC.

    Macro loading enters through the prescribed face displacements (so the total strain is just
    ``eps(u)``, no fluctuation/E split); the reaction is the internal force ``int sig:eps(v)``
    summed at the drive face -- the nonlinear analogue of the linear path's ``K_full * u``. Standard
    cells are fully confined (single Static step), so there is no free root-find."""

    def __init__(self, prob, model, sim):
        from . import _standard

        super().__init__(prob, model)
        space = self.space
        self.bcs, drive_nodes = _standard._parse_bcs(sim, space, *_standard.dof_maps(prob))
        self._drive_flat = _standard.drive_dofs(space, drive_nodes)

        self.A = fempetsc.assemble_matrix(self.a_form, bcs=self.bcs)
        self.A.assemble()
        self.r = fempetsc.assemble_vector(self.L_form)
        self.x = self.A.createVecRight()
        self.ksp = lu_ksp(self.A, self.mesh.comm)

    def _assemble_jacobian(self):
        fempetsc.assemble_matrix(self.A, self.a_form, bcs=self.bcs)

    def _residual(self):
        """Internal-force vector ``R(u) = int sig_q : eps(v)``, assembled into ``self.r``."""
        with self.r.localForm() as rl:
            rl.set(0.0)
        fempetsc.assemble_vector(self.r, self.L_form)
        self.r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return self.r

    def solve(self):
        """Solve the single Static step, commit the plastic state, and return the readODB row: the
        internal force and displacement summed at the drive face.

        Raises
        ------
        ConvergenceError
            If the Newton solve does not converge.
        """
        fempetsc.set_bc(self.u.x.petsc_vec, self.bcs)   # u carries the prescribed face displ.
        self.u.x.scatter_forward()

        def residual():
            self._eps_total = self._total_strain()
            self._fill_coeffs(self._eps_total, commit=False, tangent=False)
            r = self._residual()                                 # R(u) = internal force
            fempetsc.set_bc(r, self.bcs, x0=self.u.x.petsc_vec, alpha=-1.0)  # 0 at Dirichlet
            r.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            return r

        converged, _ = modified_newton(residual, self._refresh_tangent, self.ksp, self.x, self.u)
        if not converged:
            raise ConvergenceError("Newton solve did not converge")
        self._fill_coeffs(self._total_strain(), commit=True, tangent=False)
        return face_row(self._residual().getArray(), self.u.x.array, self._drive_flat)


def make_solver(prob, model):
    """Build a persistent periodic plastic solver (reused across a sim's plastic Static steps)."""
    return _PlasticSolver(prob, model)


def make_standard_solver(prob, model, sim):
    """Build a standard (non-periodic) plastic solver for a single plastic Static step."""
    return _StandardPlasticSolver(prob, model, sim)
