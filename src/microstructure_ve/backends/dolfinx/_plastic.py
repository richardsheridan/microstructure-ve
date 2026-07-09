"""Small-strain J2 (von Mises) isotropic-hardening plasticity for ``Static`` steps.

The linear backend (``_homogenize``) superposes per-unit-macro-strain solves, valid only for a
linear material. A ``Plastic`` response is history/path-dependent, so a Static step carrying one
needs a genuine nonlinear solve. This module provides it for the periodic, corner-driven
homogenization in **2D (plane strain) and 3D**, including multi-step sims where plastic state
persists across consecutive plastic Static steps (load/reverse hysteresis).

Design constraints (see the plan / design doc):

* **Complex-scalar PETSc build** -- UFL yield conditionals on complex scalars are ill-defined,
  so the radial return runs in **numpy at the quadrature points** (real arithmetic) and is fed
  back as Quadrature-space coefficients: the stress ``sig_q`` (residual) and algorithmic tangent
  ``C_q`` (Jacobian). Assembled complex matrices/vectors carry real values.
* **Hand-rolled Newton** over the proven ``dolfinx_mpc`` assemble + complex-LU path (mirrors
  ``_solver.LuSolver``); ``dolfinx_mpc.NonlinearProblem``/SNES is skipped in complex builds and
  would not let us refresh the return-map coefficients between iterations.
* **B-bar** (mean-dilatation) on the strain removes Q1/Q1-hex volumetric locking from J2's
  near-isochoric plastic flow, matching the elastic path's selective reduced integration.
* Quadrature degree 2 -> 4 Gauss points on a quad (CPE4) / 8 on a hex (C3D8).

The macro strain ``E`` enters through the *total* strain ``eps(E.x + u~) = E + eps(u~)`` (the
fluctuation formulation the linear path also uses); driven components are imposed and any
``free`` lateral components are found by a small outer root-find so their volume-averaged stress
vanishes. Reported row mirrors the readODB columns: ``[1.0, RF_Real..., 0..., U...]`` (zero loss).
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

_QUAD_DEGREE = 2          # 2x2(x2) Gauss = ABAQUS CPE4 (4 pts) / C3D8 (8 pts)
_FD_STEP = 1e-7           # finite-difference step for the (consistent) algorithmic tangent
_NEWTON_TOL = 1e-10       # absolute floor on the reduced residual norm
_NEWTON_RTOL = 1e-9       # relative to the first iteration's residual
_MAX_NEWTON = 50
_MAX_OUTER = 25           # free-lateral root-find iterations
_OUTER_TOL = 1e-9         # |sigma-bar| on a free component (relative to a stress scale)
_MACRO_FD_STEP = 1e-6     # macro-strain perturbation for the Jacobian bootstrap
_OUTER_FD_RETRY = 8       # outer iterations on a Broyden Jacobian before an FD refresh
_FROZEN_RATIO = 0.5       # refresh tangent+factorization when an iteration shrinks the
                          # residual by less than this (modified Newton: a frozen-tangent
                          # back-substitution costs ~3% of a refactor, so even a mediocre
                          # linear rate beats refactoring every iteration)

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
    """Radial-return a vectorized Gauss-point batch (pure -- safe to re-call for the tangent).

    ``eps_v``: ``(N, nv)`` total engineering Voigt strain (nv=3 in 2D, 6 in 3D). ``mu``/``lam``:
    ``(N,)``. ``eps_p_old``: ``(N, 6)`` tensor plastic strain. ``p_old``: ``(N,)``. Returns
    ``(sig_v (N,nv), eps_p_new (N,6), p_new (N,))``.
    """
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

    yielded = dp > 0
    safe_q = np.where(q > 0, q, 1.0)
    scale = np.where(yielded, 1.0 - 3 * mu * dp / safe_q, 1.0)
    sig_new = d * scale[:, None]
    sig_new[:, 0] += mean
    sig_new[:, 1] += mean
    sig_new[:, 2] += mean
    fac = np.where(yielded, 1.5 * dp / safe_q, 0.0)        # d eps_p = (3/2)(s/q) dp
    eps_p_new = eps_p_old + fac[:, None] * d
    return _tensor_stress_to_voigt(sig_new, dim), eps_p_new, p_old + dp


def _tangent(eps_v, mu, lam, eps_p_old, p_old, mat_id, materials, dim):
    """Algorithmic tangent ``dsig/deps`` (N,nv,nv) by finite differences of the return map,
    plus the base return-map products ``(C, sig0, eps_p_new, p_new)``. FD gives the
    *consistent* tangent (incl. hardening/segment effects); the base state is passed through
    so callers don't re-run the return map for the same strain."""
    sig0, eps_p_new, p_new = _return_map(eps_v, mu, lam, eps_p_old, p_old,
                                         mat_id, materials, dim)
    nv = eps_v.shape[1]
    C = np.empty((eps_v.shape[0], nv, nv))
    for j in range(nv):
        pert = eps_v.copy()
        pert[:, j] += _FD_STEP
        sigj, _, _ = _return_map(pert, mu, lam, eps_p_old, p_old, mat_id, materials, dim)
        C[:, :, j] = (sigj - sig0) / _FD_STEP
    return C, sig0, eps_p_new, p_new


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

    The residual evaluates stress at the B-bar strain (volumetric part replaced by its cell
    mean), so its exact derivative acts on ``dev eps(u) + mean_cell(vol eps(u))`` -- not on the
    raw ``eps(u)``. An inconsistent Jacobian costs Newton its quadratic convergence (observed:
    linear decay at ~0.34/iter, ~25 iterations per solve). The dev part integrates at the full
    rule against the quadrature-point tangent ``C_q``; the cell-mean vol part reduces to the
    1-point (centroid) rule against the cell-averaged tangent ``C_c`` (exact for a cell-constant
    tangent on the rectangular Q1 grid, where the centroid value equals the cell mean). The
    form is slightly nonsymmetric (test side stays raw, as in the residual); LU doesn't care.
    """
    u_tr, v_te = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx_q = ufl.dx(metadata={"quadrature_degree": qd, "quadrature_scheme": "default"})
    dx_1 = ufl.dx(metadata={"quadrature_degree": 1, "quadrature_scheme": "default"})
    ev_u, ev_v = _eps_voigt(u_tr, dim), _eps_voigt(v_te, dim)
    vol_u = _voigt_vol(ev_u, dim)
    return fem.form(ufl.inner(ufl.dot(C_q, ev_u - vol_u), ev_v) * dx_q
                    + ufl.inner(ufl.dot(C_c, vol_u), ev_v) * dx_1)


class _PlasticSolver:
    """Hand-rolled Newton over the periodic MPC + complex LU, with numpy return mapping.

    Persistent across a sim's plastic Static steps: ``solve`` ramps the macro strain from the
    previously committed ``E_current`` to the new step target (so reversal/cyclic patterns
    accumulate plastic state)."""

    def __init__(self, prob, model):
        space = prob.space
        if space.dim not in (2, 3):
            raise NotImplementedError("dolfinx plasticity supports 2D/3D only")
        self.space = space
        self.dim = space.dim
        self.nv = 3 if self.dim == 2 else 6
        self.mesh = space.mesh
        self.V = space.V
        self.mpc = prob.mpc
        self.bcs = prob.center_bcs
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

        self.u = fem.Function(self.V)  # the periodic fluctuation u~ (persists across steps)
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
        self.E_current = np.zeros(self.nv)
        self._shear_pairs = _SHEAR_PAIRS[self.dim]

        # macro Jacobian for the free-lateral root-find, kept across outer iterations,
        # increments, and steps (Broyden-updated; FD-bootstrapped when absent or stale)
        self._J_macro = None
        self._J_free_slots = None
        self._r_ref = 0.0  # largest initial Newton residual seen (convergence reference)

        self._fill_coeffs(np.zeros((N, self.nv)), commit=False)  # elastic C_q to allocate A
        self.A = dolfinx_mpc.assemble_matrix(self.a_form, self.mpc, bcs=self.bcs)
        self.A.assemble()
        self._C_assembled = self.C_q.x.array.copy()  # tangent the assembled A was built from
        self.b = dolfinx_mpc.assemble_vector(self.L_form, self.mpc)
        self.x = self.A.createVecRight()
        self.du = fem.Function(self.V)
        self.ksp = PETSc.KSP().create(self.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")

    # -- coefficient refresh -------------------------------------------------
    def _total_strain(self, E_voigt):
        """Total engineering strain at the quad points: ``E + eps(u~)`` (N,nv), real, with the
        mean-dilatation B-bar (volumetric strain replaced by its per-cell mean; for rectangular
        Q1/hex the cell mean == centroid value, i.e. the standard B-bar)."""
        ev = self.eps_expr.eval(self.mesh, self._cells)  # (ncells, npts, nv), complex
        eps = np.asarray(ev).real + E_voigt
        vol = eps[:, :, :self.dim].sum(axis=2)           # volumetric strain (sum of normals)
        w = self.weights
        vbar = (vol * w).sum(axis=1, keepdims=True) / w.sum()
        corr = (vbar - vol) / self.dim
        for k in range(self.dim):
            eps[:, :, k] += corr
        return eps.reshape(-1, self.nv)

    def _fill_coeffs(self, eps_total, commit, tangent=True):
        """Refresh the residual stress ``sig_q`` (always) and, when ``tangent``, the FD
        algorithmic tangent ``C_q``/``C_c`` (1+nv return maps instead of 1)."""
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
        return self._average(sig)

    def _average(self, sig_v):
        """Volume-averaged stress tensor (dim,dim) from a (N,nv) Voigt field (uniform grid)."""
        w = np.tile(self.weights, self.ncells)
        denom = self.ncells * self.weights.sum()
        sbar_v = (w[:, None] * sig_v).sum(axis=0) / denom
        return _voigt_stress_tensor(sbar_v, self.dim)

    # -- one Newton solve at a fixed macro strain ----------------------------
    def _newton(self, E_voigt):
        """Modified Newton: the tangent (and its LU factorization) is refreshed on the first
        correcting iteration and whenever the frozen-tangent residual reduction stalls below
        ``_FROZEN_RATIO``; in between, iterations reuse the factorization (back-substitution
        only, ~3% of a refactor). A step that grows the residual is backtracked (halved,
        residual-only re-check): return-map branch switching can trap an undamped Newton in a
        two-cycle. The converged answer is set by the residual and tolerance, which are
        unchanged."""
        r0 = rprev = None
        last_du = None
        halvings = 0
        for _ in range(_MAX_NEWTON):
            eps_total = self._total_strain(E_voigt)
            self._fill_coeffs(eps_total, commit=False, tangent=False)
            with self.b.localForm() as bl:
                bl.set(0.0)
            dolfinx_mpc.assemble_vector(self.L_form, self.mpc, self.b)
            dolfinx_mpc.apply_lifting(self.b, [self.a_form], [self.bcs], self.mpc)
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fempetsc.set_bc(self.b, self.bcs)
            self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            rnorm = self.b.norm()
            if r0 is None:
                r0 = rnorm
                # convergence is judged against the largest initial residual seen so far, not
                # this call's r0: warm-started re-solves (outer root-find, FD bootstrap) begin
                # essentially converged and must not grind to the absolute floor
                self._r_ref = max(self._r_ref, r0)
            if rnorm <= _NEWTON_TOL + _NEWTON_RTOL * (self._r_ref or 1.0):
                break

            if rprev is not None and rnorm > rprev and last_du is not None and halvings < 8:
                last_du *= 0.5                      # backtrack: retreat half of the last step
                self.u.x.array[:] = self.u.x.array + last_du
                halvings += 1
                continue
            halvings = 0

            if rprev is None or rnorm > _FROZEN_RATIO * rprev:
                self._fill_coeffs(eps_total, commit=False, tangent=True)
                if not np.array_equal(self.C_q.x.array, self._C_assembled):
                    self.A.zeroEntries()
                    dolfinx_mpc.assemble_matrix(self.a_form, self.mpc, bcs=self.bcs, A=self.A)
                    self.A.assemble()  # values-only refill; PETSc refactors on the next solve
                    self.ksp.setOperators(self.A)
                    self._C_assembled = self.C_q.x.array.copy()
            self.ksp.solve(self.b, self.x)            # x = A^{-1} R
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            du = self.du
            fempetsc.assign(self.x, du)
            self.mpc.homogenize(du)
            self.mpc.backsubstitution(du)
            last_du = du.x.array.copy()
            self.u.x.array[:] = self.u.x.array - last_du
            rprev = rnorm
        return self._fill_coeffs(self._total_strain(E_voigt), commit=False, tangent=False)

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
        target = self._target_voigt(loading)
        free = list(loading.free)
        free_slots = [b for (b, _) in free]
        free_val = np.array([self.E_current[s] for s in free_slots])
        E_start = self.E_current.copy()
        if free_slots != self._J_free_slots:
            self._J_macro, self._J_free_slots = None, free_slots

        E = target.copy()
        for inc in range(1, n_incr + 1):
            frac = inc / n_incr
            E_drv = E_start + frac * (target - E_start)
            prev_val = prev_resid = None  # secant pairs are only valid at a fixed E_drv
            for it in range(_MAX_OUTER):
                E = E_drv.copy()
                for k, s in enumerate(free_slots):
                    E[s] = free_val[k]
                sbar = self._newton(E)
                if not free:
                    break
                resid = np.array([sbar[b, b] for (b, _) in free])
                scale = max(np.abs(target).max() * (self.mu.mean() + self.lam.mean()), 1.0)
                if np.max(np.abs(resid)) <= _OUTER_TOL * scale:
                    break
                if self._J_macro is not None and prev_val is not None:
                    dv = free_val - prev_val  # Broyden rank-1 secant update
                    denom = float(dv @ dv)
                    if denom > 0.0:
                        self._J_macro += np.outer(
                            (resid - prev_resid) - self._J_macro @ dv, dv) / denom
                if self._J_macro is None or it == _OUTER_FD_RETRY:
                    self._J_macro = self._macro_jac_fd(E, free_slots, resid)
                prev_val, prev_resid = free_val.copy(), resid
                free_val = free_val - np.linalg.solve(self._J_macro, resid)
            self._fill_coeffs(self._total_strain(E), commit=True, tangent=False)

        self.E_current = E.copy()
        sbar = self._average(_return_map(self._total_strain(E), self.mu, self.lam, self.eps_p,
                                         self.p, self.mat_id, self.materials, self.dim)[0])
        a0 = loading.primary_axis
        RF = sbar[:, a0] * loading.cross_area
        U = np.zeros(self.dim)
        U[loading.primary_dof - 1] = loading.drive_value
        return [1.0] + list(RF.real) + [0.0] * self.dim + list(U)

    def _macro_jac_fd(self, E, free_slots, base_r):
        """FD bootstrap of the macroscopic Jacobian d(sigma-bar_free)/d(E_free): one warm-started
        inner Newton per free component, against the already-converged residual ``base_r`` at
        ``E``. No restore solve -- the outer loop's next ``_newton`` re-converges from the
        perturbed warm start. Called once, then kept fresh by Broyden updates in ``solve``."""
        n = len(free_slots)
        J = np.empty((n, n))
        for k, s in enumerate(free_slots):
            Ep = E.copy()
            Ep[s] += _MACRO_FD_STEP
            pert = self._newton(Ep)
            J[:, k] = (np.array([pert[t, t] for t in free_slots]) - base_r) / _MACRO_FD_STEP
        return J


class _StandardPlasticSolver:
    """J2 plasticity for the standard (non-periodic) path: direct face Dirichlet, no MPC.

    Macro loading enters through the prescribed face displacements (so the total strain is just
    ``eps(u)``, no fluctuation/E split); the reaction is the internal force ``int sig:eps(v)``
    summed at the drive face -- the nonlinear analogue of the linear path's ``K_full * u``.
    Standard cells are fully confined (single Static step), so there is no free root-find."""

    def __init__(self, prob, model, sim):
        from . import _standard

        space = prob.space
        if space.dim not in (2, 3):
            raise NotImplementedError("dolfinx plasticity supports 2D/3D only")
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

        self.u = fem.Function(self.V)  # the FULL displacement
        self.eps_expr = fem.Expression(_eps_voigt(self.u, self.dim), self.points)
        self._cells = np.arange(self.ncells, dtype=np.int32)

        v_te = ufl.TestFunction(self.V)
        dx_q = ufl.dx(metadata={"quadrature_degree": qd, "quadrature_scheme": "default"})
        ev_v = _eps_voigt(v_te, self.dim)
        self.a_form = _bbar_jacobian_form(self.V, self.dim, qd, self.C_q, self.C_c)
        self.L_form = fem.form(ufl.inner(self.sig_q, ev_v) * dx_q)

        self.eps_p = np.zeros((N, 6))
        self.p = np.zeros(N)

        if prob.vc_spaces is None:
            prob.vc_spaces, prob.inv_maps = _standard._build_dof_maps(space)
        self.bcs, drive_nodes = _standard._parse_bcs(sim, space, prob.vc_spaces, prob.inv_maps)
        bs = self.V.dofmap.index_map_bs
        self._drive_flat = [space.block_of_node[drive_nodes] * bs + c for c in range(self.dim)]

        self._fill_coeffs(np.zeros((N, self.nv)))  # elastic C_q to allocate A
        self.A = fempetsc.assemble_matrix(self.a_form, bcs=self.bcs)
        self.A.assemble()
        self._C_assembled = self.C_q.x.array.copy()  # tangent the assembled A was built from
        self.x = self.A.createVecRight()
        self.ksp = PETSc.KSP().create(self.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")

    def _strain(self):
        ev = self.eps_expr.eval(self.mesh, self._cells)
        eps = np.asarray(ev).real
        vol = eps[:, :, :self.dim].sum(axis=2)
        w = self.weights
        vbar = (vol * w).sum(axis=1, keepdims=True) / w.sum()
        corr = (vbar - vol) / self.dim
        for k in range(self.dim):
            eps[:, :, k] += corr
        return eps.reshape(-1, self.nv)

    def _fill_coeffs(self, eps_total, commit=False, tangent=True):
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

    def _residual(self):
        """Internal-force vector ``R(u) = int sig_q : eps(v)`` (a fresh assembled PETSc vec)."""
        r = fempetsc.assemble_vector(self.L_form)
        r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return r

    def solve(self):
        fempetsc.set_bc(self.u.x.petsc_vec, self.bcs)   # u carries the prescribed face displ.
        self.u.x.scatter_forward()
        r0 = rprev = None
        last_du = None
        halvings = 0
        for _ in range(_MAX_NEWTON):
            eps_total = self._strain()
            self._fill_coeffs(eps_total, tangent=False)
            r = self._residual()                                 # R(u) = internal force
            fempetsc.set_bc(r, self.bcs, x0=self.u.x.petsc_vec, alpha=-1.0)  # 0 at Dirichlet
            r.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            rnorm = r.norm()
            if r0 is None:
                r0 = rnorm
            if rnorm <= _NEWTON_TOL + _NEWTON_RTOL * (r0 or 1.0):
                r.destroy()
                break
            if rprev is not None and rnorm > rprev and last_du is not None and halvings < 8:
                last_du *= 0.5                      # backtrack: retreat half of the last step
                self.u.x.array[:] = self.u.x.array + last_du
                halvings += 1
                r.destroy()
                continue
            halvings = 0
            if rprev is None or rnorm > _FROZEN_RATIO * rprev:  # modified Newton (see _newton)
                self._fill_coeffs(eps_total, tangent=True)
                if not np.array_equal(self.C_q.x.array, self._C_assembled):
                    self.A.zeroEntries()
                    fempetsc.assemble_matrix(self.A, self.a_form, bcs=self.bcs)
                    self.A.assemble()  # values-only refill; PETSc refactors on the next solve
                    self.ksp.setOperators(self.A)
                    self._C_assembled = self.C_q.x.array.copy()
            self.ksp.solve(r, self.x)            # x = A^{-1} R
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            last_du = self.x.array.copy()
            self.u.x.array[:] = self.u.x.array - last_du
            r.destroy()
            rprev = rnorm
        # commit + reaction = internal force (no BC modification) summed at the drive face
        self._fill_coeffs(self._strain(), commit=True, tangent=False)
        fint = self._residual()
        f = fint.getArray()
        u = self.u.x.array
        RF = [float(f[flat].sum().real) for flat in self._drive_flat]
        U = [float(u[flat].sum().real) for flat in self._drive_flat]
        fint.destroy()
        return [1.0] + RF + [0.0] * self.dim + U


def make_solver(prob, model):
    """Build a persistent periodic plastic solver (reused across a sim's plastic Static steps)."""
    return _PlasticSolver(prob, model)


def make_standard_solver(prob, model, sim):
    """Build a standard (non-periodic) plastic solver for a single plastic Static step."""
    return _StandardPlasticSolver(prob, model, sim)


def solve_static_plastic(prob, loading, model, n_incr=None):
    """Solve one standalone plastic ``Static`` step; return its readODB row.

    ``n_incr`` defaults to 1 for a fully-prescribed (confined) macro strain (monotonic
    proportional loading is increment-independent for the radial return) and to several for a
    free-lateral case. For multi-step sims build one ``make_solver`` and call ``.solve`` per step.
    """
    if n_incr is None:
        n_incr = 1 if not list(loading.free) else 20
    return make_solver(prob, model).solve(loading, n_incr)
