"""Finite-strain (total-Lagrangian) hyperelasticity for ``Static`` steps (NLGEOM).

The ``*HYPERELASTIC`` responses (``ArrudaBoyce`` / ``Polynomial`` / ``ReducedPolynomial``)
are path-independent strain-energy materials, meaningful only at finite strain, so a Static
step carrying one needs a geometrically nonlinear solve: ``F = I + H + grad(w)`` with the
macro displacement gradient ``H`` (NON-symmetric -- the corner-driven PBC imposes the raw
``H[i,a] = value/L_a``, so a shear drive is *simple* shear, not the symmetrized small-strain
``E``) and the periodic fluctuation ``w`` through the same ``dolfinx_mpc`` constraint the
linear/plastic paths use.

Design notes (mirroring ``_plastic``, the reference for every convention here):

* **Energy formulation, UFL-differentiated.** The strain energy is written per ABAQUS
  conventions in the deviatoric invariants ``I1b = J^(-2/3) tr(C)``, ``I2b = J^(-4/3) I2``
  and ``J = det F``; the residual is ``inner(P, grad(v))`` with the first Piola-Kirchhoff
  stress ``P = diff(psi, F)`` (``F`` a ``ufl.variable``) and the Jacobian is
  ``ufl.derivative`` of the residual -- consistency is automatic (the plastic path's
  inconsistent-tangent trap cannot occur by construction).
* **Selective reduced integration** stands in for ABAQUS's B-bar mean dilatation: the
  volumetric energy ``U(J)`` integrates on the 1-point (centroid) rule, the deviatoric part
  on the full degree-2 rule. Locking comes only from the volumetric term (the deviatoric
  invariants are volume-independent), so this matches CPE4/C3D8 closely.
* **Complex-scalar PETSc build**: the energy contains ``J**(-2/3)`` / ``ln(J)`` whose
  complex branches are noise-sensitive, so the fluctuation ``w`` is held REAL (its imaginary
  part is zeroed after every Newton update); every fractional power then evaluates on a
  real-positive ``J`` and the assembled complex objects carry real values -- the same
  invariant ``_plastic`` relies on.
* **2D is plane strain**: ``F`` is embedded 3x3 with ``F[2,2] = 1`` so the out-of-plane
  stretch enters the invariants exactly.
* **Reactions are nominal (PK1)**: the drive-corner constraint force is conjugate to the
  corner displacement, ``RF = P-bar[:, a0] * cross_area`` with the volume-average PK1
  assembled on the same SRI measures as the residual (``dPi/dH[i,a] = V0 * P-bar[i,a]``;
  the face force ``int P.N dA0`` is the physical force ABAQUS reports). A free lateral
  corner normal means ``P-bar[b,b] = 0``, solved by the same FD-bootstrapped Broyden outer
  root-find as ``_plastic._PlasticSolver.solve`` (kept in sync by hand -- the loop is short
  and the residuals differ: average PK1 here, average Cauchy there).

All phases must be hyperelastic of the same model class (``_loading.can_run`` enforces it:
an ``*Elastic`` phase under NLGEOM is hypoelastic in ABAQUS, which this formulation cannot
reproduce). Reported row mirrors readODB: ``[1.0, RF_Real..., 0..., U...]`` (zero loss).
"""
from __future__ import annotations

import numpy as np

import ufl
from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc as fempetsc
import dolfinx_mpc

from microstructure_ve.constitutive import ArrudaBoyce, Polynomial, ReducedPolynomial

_NEWTON_TOL = 1e-10       # absolute floor on the residual norm
_NEWTON_RTOL = 1e-9       # relative to the largest initial residual seen
_MAX_NEWTON = 50
_MAX_OUTER = 25           # free-lateral root-find iterations
_OUTER_TOL = 1e-9         # |P-bar| on a free component (relative to a stress scale)
_MACRO_FD_STEP = 1e-6     # macro-gradient perturbation for the Jacobian bootstrap
_OUTER_FD_RETRY = 8       # outer iterations on a Broyden Jacobian before an FD refresh
_FROZEN_RATIO = 0.5       # refresh Jacobian+factorization when an iteration shrinks the
                          # residual by less than this (modified Newton, as in _plastic)

# Arruda-Boyce Langevin-expansion coefficients C_i, i = 1..5
_AB_C = (0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0, 519.0 / 673750.0)

HYPER_TYPES = (ArrudaBoyce, ReducedPolynomial, Polynomial)


def _poly_pairs(n):
    """ABAQUS data-line (i, j) order for a Polynomial of order ``n``: k = i+j from 1 to n,
    i decreasing within each k."""
    return [(i, k - i) for k in range(1, n + 1) for i in range(k, -1, -1)]


def _coefficient_table(responses):
    """Per-material coefficient columns for the (single) hyperelastic model class.

    Returns ``(kind, dev_cols, invd_cols)``: ``dev_cols`` is ``(n_mat, n_dev)`` -- the
    deviatoric coefficients in a fixed layout (ReducedPolynomial: C10..CN0; Polynomial:
    the ABAQUS pair order up to the largest N; ArrudaBoyce: the 5 combined coefficients
    ``a_i = mu*C_i/lm^(2i-2)``) -- and ``invd_cols`` is ``(n_mat, n_vol)`` holding **1/Di**
    (0 where Di == 0, i.e. that volumetric term is absent for the phase, matching the
    ABAQUS convention that a zero Di drops the term).
    """
    kind = type(responses[0])
    if any(type(r) is not kind for r in responses):
        raise NotImplementedError("all hyperelastic phases must share one model class")

    if kind is ArrudaBoyce:
        dev = np.array([[r.mu * _AB_C[i] / r.lm ** (2 * i) for i in range(5)]
                        for r in responses])
        d = np.array([[r.d_coeffs[0]] for r in responses])
    elif kind is ReducedPolynomial:
        nmax = max(r.n for r in responses)
        dev = np.zeros((len(responses), nmax))
        d = np.zeros((len(responses), nmax))
        for m, r in enumerate(responses):
            dev[m, :r.n] = r.c
            d[m, :r.n] = r.d_coeffs
    elif kind is Polynomial:
        nmax = max(r.n for r in responses)
        pairs = _poly_pairs(nmax)
        dev = np.zeros((len(responses), len(pairs)))
        d = np.zeros((len(responses), nmax))
        for m, r in enumerate(responses):
            lookup = dict(zip(_poly_pairs(r.n), r.c))
            dev[m] = [lookup.get(p, 0.0) for p in pairs]
            d[m, :r.n] = r.d_coeffs
    else:
        raise NotImplementedError(f"unsupported hyperelastic response {kind.__name__}")

    with np.errstate(divide="ignore"):
        invd = np.where(d != 0.0, 1.0 / np.where(d != 0.0, d, 1.0), 0.0)
    return kind, dev, invd


def _dg0_fields(mesh, mat_of_cell, oci, columns):
    """One DG0 Function per column, filled per cell through the pixel->dolfinx cell map.

    Columns that are zero for every material return ``None`` (their energy term is absent
    everywhere, so it is not built at all).
    """
    DG0 = fem.functionspace(mesh, ("DG", 0))
    fns = []
    for k in range(columns.shape[1]):
        if not np.any(columns[:, k]):
            fns.append(None)
            continue
        f = fem.Function(DG0)
        f.x.array[:] = columns[:, k][mat_of_cell][oci]
        fns.append(f)
    return fns


def _grad3(w, dim):
    """Displacement gradient of ``w`` embedded 3x3 (plane strain: out-of-plane row/col 0)."""
    G = ufl.grad(w)
    if dim == 2:
        return ufl.as_matrix([[G[0, 0], G[0, 1], 0],
                              [G[1, 0], G[1, 1], 0],
                              [0, 0, 0]])
    return G


def _const3(H, dim):
    """The macro-gradient Constant embedded 3x3 (plane strain: zero out-of-plane)."""
    if dim == 2:
        return ufl.as_matrix([[H[0, 0], H[0, 1], 0],
                              [H[1, 0], H[1, 1], 0],
                              [0, 0, 0]])
    return H


def _energy(kind, dev_fns, invd_fns, F):
    """(psi_dev, psi_vol) UFL scalars for the model class from the DG0 coefficient fields.

    ``F`` must be a ``ufl.variable`` so callers can take ``P = diff(psi, F)``.
    """
    J = ufl.det(F)
    C = F.T * F
    I1 = ufl.tr(C)
    I2 = (I1 ** 2 - ufl.tr(C * C)) / 2
    I1b = J ** (-2.0 / 3.0) * I1
    I2b = J ** (-4.0 / 3.0) * I2

    if kind is ArrudaBoyce:
        psi_dev = sum(a * (I1b ** (i + 1) - 3 ** (i + 1))
                      for i, a in enumerate(dev_fns) if a is not None)
        psi_vol = invd_fns[0] * ((J * J - 1) / 2 - ufl.ln(J))
        return psi_dev, psi_vol

    if kind is ReducedPolynomial:
        psi_dev = sum(c * (I1b - 3) ** (k + 1)
                      for k, c in enumerate(dev_fns) if c is not None)
    else:  # Polynomial
        pairs = _poly_pairs(_order_from_pairs(len(dev_fns)))
        psi_dev = sum(c * (I1b - 3) ** i * (I2b - 3) ** j
                      for (i, j), c in zip(pairs, dev_fns) if c is not None)
    psi_vol = sum(invd * (J - 1) ** (2 * (i + 1))
                  for i, invd in enumerate(invd_fns) if invd is not None)
    return psi_dev, psi_vol


def _order_from_pairs(n_pairs):
    """Invert ``len(_poly_pairs(N)) = N*(N+3)/2``."""
    for N in range(1, 7):
        if N * (N + 3) // 2 == n_pairs:
            return N
    raise ValueError(f"not a Polynomial pair count: {n_pairs}")


class _FormsMixin:
    """Shared UFL construction: energy, residual, Jacobian, and average-PK1 forms.

    ``self.disp`` is the displacement field (periodic fluctuation ``w`` or full ``u``),
    ``self.H`` the macro-gradient Constant (zero for the standard path).
    """

    def _build_forms(self, model, prob):
        mesh, dim = self.mesh, self.dim
        responses = [m.response for m in model.materials]
        kind, dev_cols, invd_cols = _coefficient_table(responses)
        mf = prob.matfields
        mat_of_cell = np.asarray(mf.mat_of_cell).astype(int)
        oci = prob.space.oci
        dev_fns = _dg0_fields(mesh, mat_of_cell, oci, dev_cols)
        invd_fns = _dg0_fields(mesh, mat_of_cell, oci, invd_cols)
        # outer-tolerance stress scale (initial Young's modulus, volume-averaged enough)
        self._E_scale = float(np.mean([r.youngs for r in responses]))

        self.H = fem.Constant(mesh, np.zeros((dim, dim), dtype=PETSc.ScalarType))
        F = ufl.variable(ufl.Identity(3) + _const3(self.H, dim) + _grad3(self.disp, dim))
        psi_dev, psi_vol = _energy(kind, dev_fns, invd_fns, F)
        P_dev = ufl.diff(psi_dev, F)
        P_vol = ufl.diff(psi_vol, F)

        dx2 = ufl.dx(metadata={"quadrature_degree": 2})
        dx1 = ufl.dx(metadata={"quadrature_degree": 1})  # centroid: SRI ~ B-bar mean dilatation
        v = ufl.TestFunction(self.V)
        du = ufl.TrialFunction(self.V)
        gv = _grad3(v, dim)
        R = ufl.inner(P_dev, gv) * dx2 + ufl.inner(P_vol, gv) * dx1
        self.res_form = fem.form(R)
        self.jac_form = fem.form(ufl.derivative(R, self.disp, du))
        # volume-average PK1 entries (assembled on the residual's own SRI measures, so the
        # reported reaction is exactly conjugate to what Newton converged)
        self.P_forms = [[fem.form(P_dev[i, j] * dx2 + P_vol[i, j] * dx1)
                         for j in range(dim)] for i in range(dim)]
        self.vol0 = prob.space.area

    def _avg_P(self):
        """Volume-average PK1 (dim, dim), real."""
        return np.array([[complex(fem.assemble_scalar(self.P_forms[i][j])).real
                          for j in range(self.dim)] for i in range(self.dim)]) / self.vol0


class _HyperelasticSolver(_FormsMixin):
    """Total-Lagrangian Newton over the periodic MPC + complex LU (periodic cells).

    Persistent across a sim's hyperelastic Static steps (``H_current`` and ``w`` carry, so a
    later step warm-starts from the previous converged state -- the response itself is
    path-independent)."""

    def __init__(self, prob, model):
        space = prob.space
        if space.dim not in (2, 3):
            raise NotImplementedError("dolfinx hyperelasticity supports 2D/3D only")
        self.space = space
        self.dim = space.dim
        self.mesh = space.mesh
        self.V = space.V
        self.mpc = prob.mpc
        self.bcs = prob.center_bcs

        self.disp = fem.Function(self.V)  # the periodic fluctuation w (kept real)
        self._build_forms(model, prob)

        self.A = dolfinx_mpc.assemble_matrix(self.jac_form, self.mpc, bcs=self.bcs)
        self.A.assemble()
        self.b = dolfinx_mpc.assemble_vector(self.res_form, self.mpc)
        self.x = self.A.createVecRight()
        self.du = fem.Function(self.V)
        self.ksp = PETSc.KSP().create(self.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")

        self.H_current = np.zeros((self.dim, self.dim))
        self._J_macro = None       # free-lateral macro Jacobian (Broyden-updated)
        self._J_free_slots = None
        self._r_ref = 0.0          # largest initial Newton residual seen

    def _set_H(self, H_num):
        self.H.value[:] = H_num

    # -- one Newton solve at a fixed macro gradient ---------------------------
    def _newton(self):
        """Modified Newton with step-halving backtracking, mirroring ``_plastic._newton``
        (same constants, same ``_r_ref`` warm-start reference). The tangent refresh is a UFL
        Jacobian assembly instead of a return-map coefficient fill."""
        r0 = rprev = None
        last_du = None
        halvings = 0
        for _ in range(_MAX_NEWTON):
            with self.b.localForm() as bl:
                bl.set(0.0)
            dolfinx_mpc.assemble_vector(self.res_form, self.mpc, self.b)
            dolfinx_mpc.apply_lifting(self.b, [self.jac_form], [self.bcs], self.mpc)
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fempetsc.set_bc(self.b, self.bcs)
            self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            rnorm = self.b.norm()
            # imaginary residual => some det(F) went non-positive (complex branch of the
            # fractional powers): element inversion, handled like a growing residual
            inverted = (not np.isfinite(rnorm)) or (
                np.abs(np.asarray(self.b.getArray()).imag).max() > 1e-8 * max(rnorm, 1.0))
            if not inverted:
                if r0 is None:
                    r0 = rnorm
                    self._r_ref = max(self._r_ref, r0)
                if rnorm <= _NEWTON_TOL + _NEWTON_RTOL * (self._r_ref or 1.0):
                    break

            grew = rprev is not None and rnorm > rprev
            if (inverted or grew) and last_du is not None and halvings < 8:
                last_du *= 0.5                      # backtrack: retreat half of the last step
                self.disp.x.array[:] = self.disp.x.array.real + last_du
                halvings += 1
                continue
            halvings = 0

            if rprev is None or rnorm > _FROZEN_RATIO * rprev:
                self.A.zeroEntries()
                dolfinx_mpc.assemble_matrix(self.jac_form, self.mpc, bcs=self.bcs, A=self.A)
                self.A.assemble()  # values-only refill; PETSc refactors on the next solve
                self.ksp.setOperators(self.A)
            self.ksp.solve(self.b, self.x)
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            du = self.du
            fempetsc.assign(self.x, du)
            self.mpc.homogenize(du)
            self.mpc.backsubstitution(du)
            last_du = du.x.array.real.copy()
            # keep w real: fractional powers of det(F) must see a real-positive J
            self.disp.x.array[:] = self.disp.x.array.real - last_du
            rprev = rnorm

    # -- public entry ----------------------------------------------------------
    def _target_H(self, loading):
        """The step's target macro gradient: driven entries imposed (raw, non-symmetric),
        everything else carried from the committed state (pinned entries stay 0)."""
        target = self.H_current.copy()
        for (i, a), g in loading.H_imposed.items():
            target[i, a] = g
        return target

    def solve(self, loading, n_incr):
        """Ramp the macro gradient from the committed ``H_current`` to this step's target
        over ``n_incr`` increments (Newton warm-started across increments), solving any
        ``free`` lateral diagonal entries so their average PK1 vanishes. Returns the readODB
        row and advances ``H_current``. Mirrors ``_plastic._PlasticSolver.solve``."""
        target = self._target_H(loading)
        free_slots = [b for (b, _) in loading.free]
        free_val = np.array([self.H_current[b, b] for b in free_slots])
        H_start = self.H_current.copy()
        if free_slots != self._J_free_slots:
            self._J_macro, self._J_free_slots = None, free_slots

        Hc = target.copy()
        for inc in range(1, n_incr + 1):
            frac = inc / n_incr
            H_drv = H_start + frac * (target - H_start)
            prev_val = prev_resid = None  # secant pairs are only valid at a fixed H_drv
            for it in range(_MAX_OUTER):
                Hc = H_drv.copy()
                for k, b in enumerate(free_slots):
                    Hc[b, b] = free_val[k]
                self._set_H(Hc)
                self._newton()
                if not free_slots:
                    break
                Pbar = self._avg_P()
                resid = np.array([Pbar[b, b] for b in free_slots])
                scale = max(np.abs(target).max() * self._E_scale, 1.0)
                if np.max(np.abs(resid)) <= _OUTER_TOL * scale:
                    break
                if self._J_macro is not None and prev_val is not None:
                    dv = free_val - prev_val  # Broyden rank-1 secant update
                    denom = float(dv @ dv)
                    if denom > 0.0:
                        self._J_macro += np.outer(
                            (resid - prev_resid) - self._J_macro @ dv, dv) / denom
                if self._J_macro is None or it == _OUTER_FD_RETRY:
                    self._J_macro = self._macro_jac_fd(Hc, free_slots, resid)
                prev_val, prev_resid = free_val.copy(), resid
                free_val = free_val - np.linalg.solve(self._J_macro, resid)

        self.H_current = Hc.copy()
        Pbar = self._avg_P()
        a0 = loading.primary_axis
        RF = Pbar[:, a0] * loading.cross_area  # nominal force on the reference cross-section
        U = np.zeros(self.dim)
        U[loading.primary_dof - 1] = loading.drive_value
        return [1.0] + list(RF) + [0.0] * self.dim + list(U)

    def _macro_jac_fd(self, Hc, free_slots, base_r):
        """FD bootstrap of d(P-bar_free)/d(H_free): one warm-started inner Newton per free
        slot against the already-converged ``base_r``. No restore solve (the outer loop's
        next ``_newton`` re-converges from the perturbed warm start)."""
        n = len(free_slots)
        J = np.empty((n, n))
        for k, s in enumerate(free_slots):
            Hp = Hc.copy()
            Hp[s, s] += _MACRO_FD_STEP
            self._set_H(Hp)
            self._newton()
            Pp = self._avg_P()
            J[:, k] = (np.array([Pp[t, t] for t in free_slots]) - base_r) / _MACRO_FD_STEP
        self._set_H(Hc)
        return J


class _StandardHyperelasticSolver(_FormsMixin):
    """Finite strain for the standard (non-periodic) path: direct face Dirichlet, no MPC.

    The macro loading enters through the prescribed face displacements (``F = I + grad(u)``,
    no fluctuation/H split -- ``self.H`` stays zero); the drive ramps over ``n_incr``
    increments by scaling the Dirichlet values. The reaction is the assembled nonlinear
    residual (internal nominal force) summed at the drive face -- the finite-strain analogue
    of ``_StandardPlasticSolver``'s internal force. Standard cells are fully confined
    (``_standard_can_run``), so there is no free-lateral root-find."""

    def __init__(self, prob, model, sim):
        from . import _standard

        space = prob.space
        if space.dim not in (2, 3):
            raise NotImplementedError("dolfinx hyperelasticity supports 2D/3D only")
        self.space = space
        self.dim = space.dim
        self.mesh = space.mesh
        self.V = space.V

        self.disp = fem.Function(self.V)  # the FULL displacement u (kept real)
        self._build_forms(model, prob)

        if prob.vc_spaces is None:
            prob.vc_spaces, prob.inv_maps = _standard._build_dof_maps(space)
        self.bcs, drive_nodes = _standard._parse_bcs(sim, space, prob.vc_spaces, prob.inv_maps)
        bs = self.V.dofmap.index_map_bs
        self._drive_flat = [space.block_of_node[drive_nodes] * bs + c for c in range(self.dim)]

        # full drive values per bc, so each increment can rescale the bc Functions in place
        # (the drive is applied through Newton's lifting, not by pre-setting face dofs --
        # a directly-imposed face jump against a lagging interior inverts the first element
        # row, J < 0, and the fractional powers of J go onto the complex branch)
        self._bc_gs = [(bc, bc.g.x.array.real.copy()) for bc in self.bcs]

        self.A = fempetsc.assemble_matrix(self.jac_form, bcs=self.bcs)
        self.A.assemble()
        self.x = self.A.createVecRight()
        self.ksp = PETSc.KSP().create(self.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")

    def _residual(self):
        """Assembled internal (nominal) force vector ``R(u)`` -- a fresh PETSc vec."""
        r = fempetsc.assemble_vector(self.res_form)
        r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return r

    def solve(self, n_incr):
        r_ref = 0.0
        for inc in range(1, n_incr + 1):
            frac = inc / n_incr
            for bc, g_full in self._bc_gs:
                bc.g.x.array[:] = frac * g_full
            r0 = rprev = None
            last_du = None
            halvings = 0
            for _ in range(_MAX_NEWTON):
                # Newton with the increment's BC delta carried through lifting: du solves
                # J du = R with du = u - g at the Dirichlet dofs, so u_new = u - du lands
                # exactly on the (rescaled) bc values and the first iterate of an increment
                # is the smooth linearized response to the bc change.
                r = self._residual()
                fempetsc.apply_lifting(r, [self.jac_form], bcs=[self.bcs],
                                       x0=[self.disp.x.petsc_vec], alpha=-1.0)
                r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                fempetsc.set_bc(r, self.bcs, x0=self.disp.x.petsc_vec, alpha=-1.0)
                rnorm = r.norm()
                # a residual with an imaginary part means some det(F) went non-positive
                # (fractional powers jumped onto the complex branch): element inversion,
                # handled like a growing residual (backtrack)
                inverted = (not np.isfinite(rnorm)) or (
                    np.abs(np.asarray(r.getArray()).imag).max() > 1e-8 * max(rnorm, 1.0))
                if not inverted:
                    if r0 is None:
                        r0 = rnorm
                        r_ref = max(r_ref, r0)
                    if rnorm <= _NEWTON_TOL + _NEWTON_RTOL * (r_ref or 1.0):
                        r.destroy()
                        break
                grew = rprev is not None and rnorm > rprev
                if (inverted or grew) and last_du is not None and halvings < 8:
                    last_du *= 0.5                  # backtrack: retreat half of the last step
                    self.disp.x.array[:] = self.disp.x.array.real + last_du
                    halvings += 1
                    r.destroy()
                    continue
                halvings = 0
                if rprev is None or rnorm > _FROZEN_RATIO * rprev:  # modified Newton
                    self.A.zeroEntries()
                    fempetsc.assemble_matrix(self.A, self.jac_form, bcs=self.bcs)
                    self.A.assemble()
                    self.ksp.setOperators(self.A)
                self.ksp.solve(r, self.x)
                self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                   mode=PETSc.ScatterMode.FORWARD)
                last_du = self.x.array.real.copy()
                self.disp.x.array[:] = self.disp.x.array.real - last_du
                r.destroy()
                rprev = rnorm
        # reaction = internal force (no BC masking) summed at the drive face
        fint = self._residual()
        f = fint.getArray()
        u = self.disp.x.array
        RF = [float(f[flat].sum().real) for flat in self._drive_flat]
        U = [float(u[flat].sum().real) for flat in self._drive_flat]
        fint.destroy()
        return [1.0] + RF + [0.0] * self.dim + U


def make_solver(prob, model):
    """Build a persistent periodic finite-strain solver (reused across Static steps)."""
    return _HyperelasticSolver(prob, model)


def make_standard_solver(prob, model, sim):
    """Build a standard (non-periodic) finite-strain solver for a single Static step."""
    return _StandardHyperelasticSolver(prob, model, sim)
