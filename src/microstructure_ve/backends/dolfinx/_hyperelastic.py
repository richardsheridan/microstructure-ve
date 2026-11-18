"""Finite-strain (total-Lagrangian) hyperelasticity for ``Static`` steps (NLGEOM).

The ``*HYPERELASTIC`` responses (``ArrudaBoyce`` / ``Polynomial`` / ``ReducedPolynomial``) are
path-independent strain-energy materials, meaningful only at finite strain, so a Static step
carrying one needs a geometrically nonlinear solve: ``F = I + H + grad(w)`` with the macro
displacement gradient ``H`` (non-symmetric -- the corner-driven PBC imposes the raw
``H[i,a] = value/L_a``, so a shear drive is *simple* shear, not the symmetrized small-strain ``E``)
and the periodic fluctuation ``w`` through the same ``dolfinx_mpc`` constraint the linear/plastic
paths use.

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
* **Complex-scalar PETSc build**: the energy contains ``J**(-2/3)`` / ``ln(J)`` whose complex
  branches are noise-sensitive, so the fluctuation ``w`` is held real (its imaginary part is zeroed
  after every Newton update); every fractional power then evaluates on a real-positive ``J`` and the
  assembled complex objects carry real values -- the same invariant ``_plastic`` relies on.
* **2D is plane strain**: ``F`` is embedded 3x3 with ``F[2,2] = 1`` so the out-of-plane
  stretch enters the invariants exactly.
* **Reactions are nominal (PK1)**: the drive-corner constraint force is conjugate to the corner
  displacement, ``RF = P-bar[:, a0] * cross_area`` with the volume-average PK1 assembled on the same
  SRI measures as the residual (``dPi/dH[i,a] = V0 * P-bar[i,a]``; the face force ``int P.N dA0`` is
  the physical force ABAQUS reports). A free lateral corner normal means ``P-bar[b,b] = 0``, solved
  with the fluctuation in one Newton iteration by ``_nonlinear.FreeEntries``, as the plastic path
  does (on average PK1 here, average Cauchy there).

All phases must be hyperelastic of the same model class (``_loading.check_supported`` enforces it:
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

from microstructure_ve.constitutive import (
    ArrudaBoyce,
    NeoHookean,
    Polynomial,
    ReducedPolynomial,
)

from ._nonlinear import FreeEntries, face_row, free_tol, modified_newton, periodic_row, ramp
from ._solver import ConvergenceError, lu_ksp

# Arruda-Boyce Langevin-expansion coefficients C_i, i = 1..5
_AB_C = (0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0, 519.0 / 673750.0)

def _poly_pairs(n):
    """ABAQUS data-line (i, j) order for a Polynomial of order ``n``: k = i+j from 1 to n,
    i decreasing within each k."""
    return [(i, k - i) for k in range(1, n + 1) for i in range(k, -1, -1)]


def _coefficient_table(responses):
    """Per-material coefficient columns for the (single) hyperelastic model class.

    Returns
    -------
    kind : type
        The model class shared by every response.
    dev_cols : numpy.ndarray
        ``(n_mat, n_dev)`` deviatoric coefficients in a fixed layout (ReducedPolynomial: C10..CN0;
        Polynomial: the ABAQUS pair order up to the largest N; ArrudaBoyce: the 5 combined
        coefficients ``a_i = mu*C_i/lm^(2i-2)``; NeoHookean: ``mu0``).
    invd_cols : numpy.ndarray
        ``(n_mat, n_vol)`` holding ``1/Di``, 0 where ``Di == 0`` (the ABAQUS convention that a zero
        Di drops that volumetric term). For NeoHookean, which has no D series, the one column is
        ``lam`` itself.

    Raises
    ------
    NotImplementedError
        If the responses mix model classes or the class is not a supported hyperelastic one.
    """
    kind = type(responses[0])
    if any(type(r) is not kind for r in responses):
        raise NotImplementedError("all hyperelastic phases must share one model class")

    if kind is NeoHookean:
        # the coupled form's volumetric column is lambda itself, not 1/Di
        dev = np.array([[r.mu0] for r in responses])
        lam = np.array([[r.lam] for r in responses])
        return kind, dev, lam
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


def _dg0_fields(mesh, mat_of_cell, oci, columns, keep_zero=False):
    """One DG0 Function per column, filled per cell through the pixel->dolfinx cell map.

    Columns that are zero for every material return ``None`` (their energy term is absent
    everywhere, so it is not built at all), unless ``keep_zero`` is set, in which case every column
    gets a Function.
    """
    DG0 = fem.functionspace(mesh, ("DG", 0))
    fns = []
    for k in range(columns.shape[1]):
        if not keep_zero and not np.any(columns[:, k]):
            fns.append(None)
            continue
        f = fem.Function(DG0)
        f.x.array[:] = columns[:, k][mat_of_cell][oci]
        fns.append(f)
    return fns


def _embed3(G, dim):
    """A ``(dim, dim)`` gradient embedded 3x3 (plane strain: out-of-plane row/col 0)."""
    if dim == 2:
        return ufl.as_matrix([[G[0, 0], G[0, 1], 0],
                              [G[1, 0], G[1, 1], 0],
                              [0, 0, 0]])
    return G


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

    if kind is NeoHookean:
        # coupled compressible form: the full I1 (not I1b) plus the -mu*ln(J) coupling; the
        # invd slot carries lambda directly (see _coefficient_table)
        mu, lam = dev_fns[0], invd_fns[0]
        psi_dev = mu / 2 * (I1 - 3) - mu * ufl.ln(J)
        psi_vol = lam / 2 * ((J * J - 1) / 2 - ufl.ln(J))
        return psi_dev, psi_vol

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
    """Shared setup and UFL construction: energy, residual, Jacobian, and average-PK1 forms.

    ``self.disp`` is the displacement field (periodic fluctuation ``w`` or full ``u``, kept real),
    ``self.H`` the macro-gradient Constant (zero for the standard path).
    """

    def __init__(self, prob, model):
        space = prob.space
        if space.dim not in (2, 3):
            raise NotImplementedError("dolfinx hyperelasticity supports 2D/3D only")
        self.space = space
        self.dim = space.dim
        self.mesh = space.mesh
        self.V = space.V
        self.disp = fem.Function(self.V)
        self._build_forms(model, prob)

    def _build_forms(self, model, prob):
        mesh, dim = self.mesh, self.dim
        responses = [m.response for m in model.materials]
        kind, dev_cols, invd_cols = _coefficient_table(responses)
        mf = prob.matfields
        mat_of_cell = np.asarray(mf.mat_of_cell).astype(int)
        oci = prob.space.oci
        dev_fns = _dg0_fields(mesh, mat_of_cell, oci, dev_cols)
        # NeoHookean's energy reads its lambda field unconditionally (its volumetric term is
        # not one of an optional Di series), so the field is built even when lambda = 0
        # in every phase
        invd_fns = _dg0_fields(mesh, mat_of_cell, oci, invd_cols,
                               keep_zero=kind is NeoHookean)
        # free-entry tolerance stress scale: the mean initial Young's modulus of the materials
        self._E_scale = float(np.mean([r.youngs for r in responses]))

        self.H = fem.Constant(mesh, np.zeros((dim, dim), dtype=PETSc.ScalarType))
        F = ufl.variable(ufl.Identity(3) + _embed3(self.H, dim)
                         + _embed3(ufl.grad(self.disp), dim))
        psi_dev, psi_vol = _energy(kind, dev_fns, invd_fns, F)
        P_dev = ufl.diff(psi_dev, F)
        P_vol = ufl.diff(psi_vol, F)

        # TODO: reproduce ABAQUS's in-plane F-bar pairing to close the hyperelastic parity
        # gap (_HYPER_RTOL in tests/test_matrix_parity.py); it needs per-iteration
        # DG0 Jbar/pbar coefficient fields in place of this selective reduced integration.
        dx2 = ufl.dx(metadata={"quadrature_degree": 2})
        dx1 = ufl.dx(metadata={"quadrature_degree": 1})  # centroid: SRI ~ B-bar mean dilatation
        v = ufl.TestFunction(self.V)
        du = ufl.TrialFunction(self.V)
        gv = _embed3(ufl.grad(v), dim)
        R_dev = ufl.inner(P_dev, gv) * dx2
        R_vol = ufl.inner(P_vol, gv) * dx1
        # A 1-form mixing quadrature degrees compiles to one rule, so the residual is two
        # forms assembled in two passes, see
        # docs/dolfinx_backend_design.md#quadrature-merge-in-1-forms.
        # TODO: check for the merge on the periodic path too; the Clapeyron identity in
        # tests/test_mechanical_mnist.py covers the standard path only.
        self.res_forms = (fem.form(R_dev), fem.form(R_vol))
        self.jac_form = fem.form(ufl.derivative(R_dev + R_vol, self.disp, du))
        # total strain energy on the residual's own SRI measures (so the reported energy
        # is exactly the potential Newton minimized); zero at the undeformed state
        self.energy_form = fem.form(psi_dev * dx2 + psi_vol * dx1)
        # volume-average PK1 entries (assembled on the residual's own SRI measures, so the
        # reported reaction is exactly conjugate to what Newton converged)
        self.P_forms = [[fem.form(P_dev[i, j] * dx2 + P_vol[i, j] * dx1)
                         for j in range(dim)] for i in range(dim)]
        # F depends on H only through H + grad(disp), so a derivative with respect to H[b, b]
        # is one with respect to disp along the field x_b e_b. Each part is its own form, as
        # for the residual.
        self._dir = fem.Function(self.V)
        self._dR_forms = [fem.form(ufl.derivative(R, self.disp, self._dir))
                          for R in (R_dev, R_vol)]
        self._dP_forms = [[fem.form(ufl.derivative(Pd[b, b] * dx, self.disp, self._dir))
                           for Pd, dx in ((P_dev, dx2), (P_vol, dx1))] for b in range(dim)]
        self.vol0 = prob.space.area

    def _refresh_jacobian(self):
        """``modified_newton``'s Jacobian refresh: reassemble at the current displacement."""
        self.A.zeroEntries()
        self._assemble_jacobian()
        self.A.assemble()  # values-only refill; PETSc refactors on the next solve
        self.ksp.setOperators(self.A)

    def _avg_P(self):
        """Volume-average PK1 (dim, dim), real."""
        return np.array([[complex(fem.assemble_scalar(self.P_forms[i][j])).real
                          for j in range(self.dim)] for i in range(self.dim)]) / self.vol0

    def strain_energy(self):
        """Total strain energy of the current converged state (real scalar)."""
        return float(complex(fem.assemble_scalar(self.energy_form)).real)


class _HyperelasticSolver(_FormsMixin):
    """Total-Lagrangian Newton over the periodic MPC + complex LU (periodic cells).

    Persistent across a sim's hyperelastic Static steps (``H_current`` and ``w`` carry, so a
    later step warm-starts from the previous converged state -- the response itself is
    path-independent)."""

    def __init__(self, prob, model):
        super().__init__(prob, model)  # self.disp is the periodic fluctuation w
        self.mpc = prob.mpc
        self.bcs = prob.center_bcs

        self.A = dolfinx_mpc.assemble_matrix(self.jac_form, self.mpc, bcs=self.bcs)
        self.A.assemble()
        self.b = dolfinx_mpc.assemble_vector(self.res_forms[0], self.mpc)
        self._b_vol = dolfinx_mpc.assemble_vector(self.res_forms[1], self.mpc)
        self.x = self.A.createVecRight()
        self.ksp = lu_ksp(self.A, self.mesh.comm)

        self.H_current = np.zeros((self.dim, self.dim))
        self._r_ref = 0.0          # largest initial Newton residual seen
        mode = fem.Function(self.V)
        self._modes = []           # the field x_b e_b, whose gradient is the unit H[b, b]
        for b in range(self.dim):
            mode.interpolate(lambda x, b=b: np.stack(
                [x[b] if c == b else np.zeros_like(x[b]) for c in range(self.dim)]))
            self._modes.append(mode.x.array.real.copy())

    def _assemble_jacobian(self):
        dolfinx_mpc.assemble_matrix(self.jac_form, self.mpc, bcs=self.bcs, A=self.A)

    def _residual(self):
        """The constrained residual at the current fluctuation, in ``self.b``."""
        # each part gets its own vector: dolfinx_mpc.assemble_vector zeroes its target
        # (docs/dolfinx_backend_design.md#vector-assembly-into-one-target)
        dolfinx_mpc.assemble_vector(self.res_forms[0], self.mpc, self.b)
        dolfinx_mpc.assemble_vector(self.res_forms[1], self.mpc, self._b_vol)
        with self.b.localForm() as bl, self._b_vol.localForm() as vl:
            bl.array[:] += vl.array
        dolfinx_mpc.apply_lifting(self.b, [self.jac_form], [self.bcs], self.mpc)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fempetsc.set_bc(self.b, self.bcs)
        self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return self.b

    def _free_stress(self, axes):
        """Average PK1 normal entries on ``axes`` at the current state."""
        return [complex(fem.assemble_scalar(self.P_forms[b][b])).real / self.vol0 for b in axes]

    def _free_stress_derivative(self, y, axes):
        """Derivative of ``_free_stress`` along the fluctuation direction ``y``."""
        self._dir.x.array[:] = y
        return np.array([sum(complex(fem.assemble_scalar(f)).real for f in self._dP_forms[b])
                         for b in axes]) / self.vol0

    def _macro_jacobian(self, axes):
        """``FreeEntries``'s ``(B, D)`` for the macro gradient entries ``H[b, b]``, ``b`` in
        ``axes``."""
        B, D = [], []
        for c in axes:
            self._dir.x.array[:] = self._modes[c]
            col, vol = self.b.duplicate(), self._b_vol
            dolfinx_mpc.assemble_vector(self._dR_forms[0], self.mpc, col)
            dolfinx_mpc.assemble_vector(self._dR_forms[1], self.mpc, vol)
            with col.localForm() as cl, vol.localForm() as vl:
                cl.array[:] += vl.array
            col.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fempetsc.set_bc(col, self.bcs)
            col.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            B.append(col)
            D.append(self._free_stress_derivative(self._modes[c], axes))
        return B, np.column_stack(D)

    # -- one Newton solve at a macro gradient ---------------------------------
    def _newton(self, H, free=None):
        """Equilibrate the fluctuation at macro gradient ``H`` by ``modified_newton``, solving
        the ``free`` entries of ``H`` in place, held real and judged against the largest initial
        residual this solver has seen. Returns whether the residual met its tolerance."""
        def residual():
            self.H.value[:] = H
            return self._residual()

        converged, self._r_ref = modified_newton(residual, self._refresh_jacobian,
                                                 self.ksp, self.x, self.disp, mpc=self.mpc,
                                                 real=True, r_ref=self._r_ref, free=free)
        return converged

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
        row, whose reaction is the nominal force on the reference cross-section, and advances
        ``H_current``."""
        target = self._target_H(loading)
        axes = [b for (b, _) in loading.free]
        tol = free_tol(target, self._E_scale)

        def equilibrate(H):
            free = None
            if axes:
                free = FreeEntries(H, [(b, b) for b in axes], axes, tol, self.vol0,
                                   self._free_stress, self._free_stress_derivative,
                                   self._macro_jacobian)
            return self._newton(H, free)

        Hc = ramp(self.H_current, target, [(b, b) for b in axes], n_incr, equilibrate)
        self.H_current = Hc.copy()
        return periodic_row(loading, self._avg_P())


class _StandardHyperelasticSolver(_FormsMixin):
    """Finite strain for the standard (non-periodic) path: direct face Dirichlet, no MPC.

    The macro loading enters through the prescribed face displacements (``F = I + grad(u)``, no
    fluctuation/H split -- ``self.H`` stays zero); the drive ramps over ``n_incr`` increments by
    scaling the Dirichlet values. The reaction is the assembled nonlinear residual (internal nominal
    force) summed at the drive face -- the finite-strain analogue of ``_StandardPlasticSolver``'s
    internal force. Standard cells are fully confined (``_loading._standard_check``), so there are no
    free lateral entries.

    Persistent across a sim's Static steps: ``set_step`` re-parses each step's drives and
    the next ``solve`` ramps from the committed (previous converged) boundary values to the
    new targets, warm-starting Newton from the previous state -- the response itself is
    path-independent, so multistep and single-step land on the same converged states."""

    def __init__(self, prob, model, sim):
        super().__init__(prob, model)  # self.disp is the full displacement u

        self._prob = prob
        self._sim = sim
        self.set_step(sim.steps[0])

        self.A = fempetsc.assemble_matrix(self.jac_form, bcs=self.bcs)
        self.A.assemble()
        self.r = fempetsc.assemble_vector(self.res_forms[0])
        self.x = self.A.createVecRight()
        self.ksp = lu_ksp(self.A, self.mesh.comm)

    def set_step(self, step):
        """Re-parse ``step``'s BCs; the next ``solve`` ramps from the committed state.

        The ramp start for each bc is read off the current converged displacement at that bc's dofs
        (the converged u is the previous step's committed boundary value, since the drive is applied
        through Newton's lifting -- see ``solve``), so no cross-step value bookkeeping is needed."""
        from . import _standard

        entries, drive_nodes = _standard._parse_bcs_ex(
            self._sim, self.space, *_standard.dof_maps(self._prob), step)
        self.bcs = [bc for bc, _, _ in entries]
        self._drive_flat = _standard.drive_dofs(self.space, drive_nodes)
        # (bc, g_start, g_target) per bc: each increment interpolates the bc Function in
        # place between the committed value and this step's target, applied through
        # Newton's lifting (docs/dolfinx_backend_design.md#standard-bc-drive-through-lifting)
        u = self.disp.x.array.real
        self._bc_gs = []
        for bc, flat_dofs, vc_dofs in entries:
            g_start = np.zeros_like(bc.g.x.array.real)
            g_start[vc_dofs] = u[flat_dofs]
            self._bc_gs.append((bc, g_start, bc.g.x.array.real.copy()))

    def _assemble_jacobian(self):
        fempetsc.assemble_matrix(self.A, self.jac_form, bcs=self.bcs)

    def _residual(self):
        """Assembled internal (nominal) force vector ``R(u)``, in ``self.r``."""
        with self.r.localForm() as rl:
            rl.set(0.0)
        fempetsc.assemble_vector(self.r, self.res_forms[0])
        fempetsc.assemble_vector(self.r, self.res_forms[1])
        self.r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return self.r

    def _lifted_residual(self):
        """The Newton residual with the increment's BC delta carried through lifting: ``du``
        solves ``J du = R`` with ``du = u - g`` at the Dirichlet dofs, so ``u_new = u - du``
        lands exactly on the (rescaled) bc values and the first iterate of an increment is
        the smooth linearized response to the bc change."""
        r = self._residual()
        fempetsc.apply_lifting(r, [self.jac_form], bcs=[self.bcs],
                               x0=[self.disp.x.petsc_vec], alpha=-1.0)
        r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fempetsc.set_bc(r, self.bcs, x0=self.disp.x.petsc_vec, alpha=-1.0)
        return r

    def solve(self, n_incr):
        """Ramp the bc values from the committed state to this step's targets over ``n_incr``
        increments and return the readODB row: the internal force and displacement summed
        at the drive face.

        Raises
        ------
        ConvergenceError
            If the Newton solve of some increment does not converge.
        """
        r_ref = 0.0
        for inc in range(1, n_incr + 1):
            frac = inc / n_incr
            for bc, g_start, g_target in self._bc_gs:
                bc.g.x.array[:] = g_start + frac * (g_target - g_start)
            converged, r_ref = modified_newton(self._lifted_residual, self._refresh_jacobian,
                                               self.ksp, self.x, self.disp, real=True,
                                               r_ref=r_ref)
            if not converged:
                raise ConvergenceError(
                    f"Newton solve did not converge in increment {inc}/{n_incr}")
        return face_row(self._residual().getArray(), self.disp.x.array, self._drive_flat)


def make_solver(prob, model):
    """Build a persistent periodic finite-strain solver (reused across Static steps)."""
    return _HyperelasticSolver(prob, model)


def make_standard_solver(prob, model, sim):
    """Build a persistent standard (non-periodic) finite-strain solver.

    Reused across a sim's Static steps: call ``set_step`` before each ``solve`` so the
    ramp runs from the committed state to that step's prescribed values."""
    return _StandardHyperelasticSolver(prob, model, sim)
