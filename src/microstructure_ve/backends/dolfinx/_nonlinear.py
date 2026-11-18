"""The drivers the nonlinear ``Static`` solvers share (``_plastic``, ``_hyperelastic``).

``modified_newton`` equilibrates the field at a macro state, solving any free lateral macro entries
(``FreeEntries``) in the same iteration; ``ramp`` steps the macro state over increments. Each
solver supplies its residual, Jacobian refresh and, for free entries, their average stress and its
derivatives; the iteration and its tolerances live here. ``ramp`` raises when an increment fails;
``modified_newton`` reports convergence and leaves the raise to its caller, and the standard
solvers, which have no free lateral entries, run their own increments around it. ``periodic_row``
and ``face_row`` build the readODB row of a converged step.
"""
from __future__ import annotations

import numpy as np

from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc as fempetsc

from ._solver import ConvergenceError

_NEWTON_TOL = 1e-10       # absolute floor on the residual norm
_NEWTON_RTOL = 1e-9       # relative to the convergence reference (see modified_newton)
_MAX_NEWTON = 50
_MAX_HALVINGS = 8         # consecutive step halvings before a growing step is taken anyway
_FROZEN_RATIO = 0.5       # refresh tangent+factorization when an iteration shrinks the
                          # residual by less than this (tools/profile_plastic.py)
_INVERSION_TOL = 1e-8     # imaginary residual, relative to max(|R|, 1), read as inversion
_FREE_TOL = 1e-9          # average stress on a free entry, relative to a stress scale


def modified_newton(residual, refresh_jacobian, ksp, x, field, *, mpc=None, real=False,
                    r_ref=0.0, free=None):
    """Drive ``residual`` to zero by modified Newton on ``field``, with step halving.

    Each iteration solves ``J du = R`` and updates ``field <- field - du``. ``refresh_jacobian`` is
    called on the first correcting iteration and whenever an iteration shrinks the residual by less
    than ``_FROZEN_RATIO``; in between, iterations reuse the current factorization. A step that
    grows the residual is halved back, at most ``_MAX_HALVINGS`` times in a row: return-map branch
    switching can trap an undamped Newton in a two-cycle. The converged answer is set by the
    residual and tolerance alone.

    With ``free``, its macro entries are unknowns of the same iteration: the step solves the
    bordered system of the field and the free entries by a Schur complement on the one
    factorization of ``J`` (``FreeEntries``), a halving retreats both, and convergence also needs
    the free entries' average stress within ``free.tol``.

    Parameters
    ----------
    residual : callable
        ``residual() -> PETSc.Vec``, the residual at the current ``field`` with its Dirichlet rows
        set to the correction they impose.
    refresh_jacobian : callable
        ``refresh_jacobian()`` brings the operator of ``ksp`` up to date with the iterate of the
        last ``residual`` call, or leaves it alone when it would not change.
    ksp : PETSc.KSP
        The linear solver of ``J du = R``.
    x : PETSc.Vec
        The solution vector of ``ksp``.
    field : dolfinx.fem.Function
        The iterate, updated in place.
    mpc : dolfinx_mpc.MultiPointConstraint, optional
        The constraint whose reduced dofs ``x`` holds; without it ``x`` is in ``field``'s layout.
    real : bool
        Hold ``field`` real, and treat a non-finite residual or one with an imaginary part as an
        inverted element, backtracked like a growing residual.
    r_ref : float
        The convergence reference before this call. The call's first residual raises it to at least
        that residual, and convergence is ``|R| <= _NEWTON_TOL + _NEWTON_RTOL * r_ref``. A caller
        that passes the returned reference back in judges a warm-started re-solve, which begins
        essentially converged, against the largest initial residual seen rather than its own, so it
        does not grind to the absolute floor.
    free : FreeEntries, optional
        Free macro entries to solve together with ``field``; ``residual`` must read them.

    Returns
    -------
    converged : bool
        Whether the residual (and the free entries' stress) met its tolerance within
        ``_MAX_NEWTON`` iterations.
    r_ref : float
        The convergence reference after this call.
    """
    du = fem.Function(field.function_space) if mpc is not None else None

    def solve(rhs):
        """``J^{-1} rhs`` in ``field``'s layout."""
        ksp.solve(rhs, x)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        if mpc is None:
            step = x.array
        else:
            fempetsc.assign(x, du)
            mpc.homogenize(du)
            mpc.backsubstitution(du)
            step = du.x.array
        return (step.real if real else step).copy()

    first = True
    rprev = last_du = None
    halvings = 0
    for _ in range(_MAX_NEWTON):
        b = residual()
        rnorm = b.norm()
        g = free.stress() if free is not None else None
        inverted = real and ((not np.isfinite(rnorm)) or (
            np.abs(np.asarray(b.getArray()).imag).max() > _INVERSION_TOL * max(rnorm, 1.0)))
        if not inverted:
            if first:
                first = False
                r_ref = max(r_ref, rnorm)
            if (rnorm <= _NEWTON_TOL + _NEWTON_RTOL * (r_ref or 1.0)
                    and (free is None or free.converged(g))):
                return True, r_ref
        merit = rnorm if free is None else free.merit(rnorm, g)

        grew = rprev is not None and merit > rprev
        if (inverted or grew) and last_du is not None and halvings < _MAX_HALVINGS:
            last_du *= 0.5                      # backtrack: retreat half of the last step
            field.x.array[:] = _values(field, real) + last_du
            if free is not None:
                free.retreat()
            halvings += 1
            continue
        halvings = 0

        if rprev is None or merit > _FROZEN_RATIO * rprev:
            refresh_jacobian()
            if free is not None:
                free.linearize(solve)
        last_du = solve(b)
        if free is not None:
            try:
                last_du = free.step(last_du, g)
            except np.linalg.LinAlgError:
                return False, r_ref
        field.x.array[:] = _values(field, real) - last_du
        rprev = merit
    return False, r_ref


def _values(field, real):
    """The dof values of ``field``, their real part when ``real``."""
    return field.x.array.real if real else field.x.array


def periodic_row(loading, avg_stress):
    """The readODB row of a converged periodic ``Static`` step, with zero loss.

    Parameters
    ----------
    loading : _loading.MacroLoading
        The step's loading; it names the reported axis, dof and drive.
    avg_stress : numpy.ndarray
        The real volume-averaged stress ``(dim, dim)`` conjugate to the drive.

    Returns
    -------
    list of float
        ``[1.0, RF..., 0..., U...]`` with ``RF = avg_stress[:, primary_axis] * cross_area`` and
        ``U`` the drive's corner displacement.
    """
    dim = loading.dim
    RF = avg_stress[:, loading.primary_axis] * loading.cross_area
    U = np.zeros(dim)
    U[loading.primary_dof - 1] = loading.drive_value
    return [1.0] + list(RF.real) + [0.0] * dim + list(U)


def face_row(force, disp, drive_dofs):
    """The readODB row of a converged standard ``Static`` step, with zero loss.

    Parameters
    ----------
    force : numpy.ndarray
        The internal force vector, without Dirichlet masking.
    disp : numpy.ndarray
        The converged displacement dof values.
    drive_dofs : list of numpy.ndarray
        Per component, the dofs of the drive face (``_standard.drive_dofs``).

    Returns
    -------
    list of float
        ``[1.0, RF..., 0..., U...]``, each ``RF`` and ``U`` summed over the drive face.
    """
    RF = [float(force[dofs].sum().real) for dofs in drive_dofs]
    U = [float(disp[dofs].sum().real) for dofs in drive_dofs]
    return [1.0] + RF + [0.0] * len(drive_dofs) + U


def free_tol(target, stress_scale):
    """The tolerance on a free entry's average stress for a step driven to macro state
    ``target``, given the modulus ``stress_scale`` that turns a macro strain into a stress."""
    return _FREE_TOL * max(np.abs(target).max() * stress_scale, 1.0)


class FreeEntries:
    """Free lateral macro entries, solved with the field in one ``modified_newton`` iteration.

    The linearized system of the field step ``du`` and the free entries' step ``de`` is ::

        J du + B de = R
        G du + D de = g

    with ``R`` the field residual, ``g`` the free entries' average normal stress, ``B = dR/de``,
    ``G = dg/dfield`` and ``D = dg/de``. The step is ``field <- field - du``, ``e <- e - de``.
    Eliminating ``du`` on the factorization of ``J``: ``x1 = J^{-1} R``, ``X2 = J^{-1} B`` (one
    back-substitution per free entry, when the Jacobian is refreshed), ``S = D - G X2``,
    ``de = S^{-1} (g - G x1)`` and ``du = x1 - X2 de``.

    Parameters
    ----------
    macro : numpy.ndarray
        The macro state; its free entries are updated in place.
    index : list
        The index into ``macro`` of the entry conjugate to the average normal stress on each axis.
    axes : list of int
        The axes whose average normal stress must vanish.
    tol : float
        The tolerance on that stress (``free_tol``).
    volume : float
        The cell volume, which turns an average stress into a force for the step's merit.
    stress : callable
        ``stress(axes)`` is the average normal stress on ``axes`` at the last residual.
    stress_derivative : callable
        ``stress_derivative(y, axes)`` is its derivative with respect to the field, in the
        direction of the dof array ``y`` in the field's layout, at the last Jacobian refresh.
    macro_jacobian : callable
        ``macro_jacobian(axes) -> (B, D)`` at the last Jacobian refresh: ``B`` a list of
        constrained residual derivatives (``PETSc.Vec``, one per free entry), ``D`` the
        ``(n, n)`` derivative of the stress with respect to the free entries.
    """

    def __init__(self, macro, index, axes, tol, volume, stress, stress_derivative,
                 macro_jacobian):
        self.macro = macro
        self.index = index
        self.axes = axes
        self.tol = tol
        self.volume = volume
        self._stress = stress
        self._stress_derivative = stress_derivative
        self._macro_jacobian = macro_jacobian
        self._X2 = self._S = self._last_de = None

    def stress(self):
        """The free entries' average normal stress at the last residual."""
        return np.asarray(self._stress(self.axes), dtype=float)

    def converged(self, g):
        """Whether the stress ``g`` meets ``tol``."""
        return np.max(np.abs(g)) <= self.tol

    def merit(self, rnorm, g):
        """The size of the bordered residual, the stress taken as a force on the cell."""
        return float(np.hypot(rnorm, self.volume * np.linalg.norm(g)))

    def linearize(self, solve):
        """Factor-side work of a Jacobian refresh: ``X2`` and the Schur complement ``S``."""
        B, D = self._macro_jacobian(self.axes)
        self._X2 = np.array([solve(col) for col in B])
        GX2 = np.column_stack([self._stress_derivative(x2, self.axes) for x2 in self._X2])
        self._S = np.asarray(D, dtype=float) - GX2

    def step(self, x1, g):
        """Take the free entries' step for the field solve ``x1 = J^{-1} R``; return ``du``.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the Schur complement is singular.
        """
        de = np.linalg.solve(self._S, g - self._stress_derivative(x1, self.axes))
        self._move(-de)
        self._last_de = de
        return x1 - self._X2.T @ de

    def retreat(self):
        """Retreat half of the last free-entry step, alongside the field's halving."""
        self._last_de = 0.5 * self._last_de
        self._move(self._last_de)

    def _move(self, delta):
        for k, i in enumerate(self.index):
            self.macro[i] += delta[k]


def ramp(start, target, index, n_incr, equilibrate, commit=None):
    """Ramp the macro state from ``start`` to ``target`` in ``n_incr`` equal increments.

    Parameters
    ----------
    start : numpy.ndarray
        The committed macro state.
    target : numpy.ndarray
        This step's macro state; its free entries are solved for.
    index : list
        The index into the macro state of each free entry; each increment starts them from the
        last converged values.
    n_incr : int
        The number of increments.
    equilibrate : callable
        ``equilibrate(M) -> bool`` solves at macro state ``M``, updating its free entries in place,
        and returns whether the solve met its tolerance.
    commit : callable, optional
        ``commit(M)`` is called after each converged increment.

    Returns
    -------
    numpy.ndarray
        The converged macro state of the last increment.

    Raises
    ------
    ConvergenceError
        If the solve of some increment does not converge.
    """
    M = start.copy()
    for inc in range(1, n_incr + 1):
        free_val = [M[i] for i in index]
        M = start + inc / n_incr * (target - start)
        for i, val in zip(index, free_val):
            M[i] = val
        if not equilibrate(M):
            raise ConvergenceError(f"Newton solve did not converge in increment {inc}/{n_incr}")
        if commit is not None:
            commit(M)
    return M
