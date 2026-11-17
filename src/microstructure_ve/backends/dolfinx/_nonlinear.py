"""The drivers the nonlinear ``Static`` solvers share (``_plastic``, ``_hyperelastic``).

``modified_newton`` equilibrates the field at a fixed macro state; ``FreeLateralRamp`` ramps the
macro state over increments and finds the free lateral entries by a root-find on the average
stress. Each solver supplies its residual, Jacobian and average stress; the iteration, its
tolerances and its failure raises live here.
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
                          # residual by less than this (modified Newton: a frozen-tangent
                          # back-substitution costs ~3% of a refactor, so even a mediocre
                          # linear rate beats refactoring every iteration)
_INVERSION_TOL = 1e-8     # imaginary residual, relative to max(|R|, 1), read as inversion
_MAX_OUTER = 25           # free-lateral root-find iterations
_OUTER_TOL = 1e-9         # average stress on a free entry, relative to a stress scale
_MACRO_FD_STEP = 1e-6     # macro-state perturbation for the Jacobian bootstrap
_OUTER_FD_RETRY = 8       # outer iterations on a Broyden Jacobian before an FD refresh


def modified_newton(residual, refresh_jacobian, ksp, x, field, *, mpc=None, real=False,
                    r_ref=0.0):
    """Drive ``residual`` to zero by modified Newton on ``field``, with step halving.

    Each iteration solves ``J du = R`` and updates ``field <- field - du``. The Jacobian and
    its LU factorization are refreshed on the first correcting iteration and whenever an
    iteration shrinks the residual by less than ``_FROZEN_RATIO``; in between, iterations
    reuse the factorization. A step that grows the residual is halved back, at most
    ``_MAX_HALVINGS`` times in a row: return-map branch switching can trap an undamped Newton
    in a two-cycle. The converged answer is set by the residual and tolerance alone.

    Parameters
    ----------
    residual : callable
        ``residual() -> PETSc.Vec``, the residual at the current ``field`` with its Dirichlet
        rows set to the correction they impose.
    refresh_jacobian : callable
        ``refresh_jacobian()`` refills the operator of ``ksp`` at the iterate of the last
        ``residual`` call.
    ksp : PETSc.KSP
        The linear solver of ``J du = R``.
    x : PETSc.Vec
        The solution vector of ``ksp``.
    field : dolfinx.fem.Function
        The iterate, updated in place.
    mpc : dolfinx_mpc.MultiPointConstraint, optional
        The constraint whose reduced dofs ``x`` holds; without it ``x`` is in ``field``'s
        layout.
    real : bool
        Hold ``field`` real, and treat a non-finite residual or one with an imaginary part as
        an inverted element, backtracked like a growing residual.
    r_ref : float
        The convergence reference before this call. The call's first residual raises it to at
        least that residual, and convergence is ``|R| <= _NEWTON_TOL + _NEWTON_RTOL * r_ref``.
        A caller that passes the returned reference back in judges a warm-started re-solve,
        which begins essentially converged, against the largest initial residual seen rather
        than its own, so it does not grind to the absolute floor.

    Returns
    -------
    converged : bool
        Whether the residual met its tolerance within ``_MAX_NEWTON`` iterations.
    r_ref : float
        The convergence reference after this call.
    """
    du = fem.Function(field.function_space) if mpc is not None else None
    first = True
    rprev = last_du = None
    halvings = 0
    for _ in range(_MAX_NEWTON):
        b = residual()
        rnorm = b.norm()
        inverted = real and ((not np.isfinite(rnorm)) or (
            np.abs(np.asarray(b.getArray()).imag).max() > _INVERSION_TOL * max(rnorm, 1.0)))
        if not inverted:
            if first:
                first = False
                r_ref = max(r_ref, rnorm)
            if rnorm <= _NEWTON_TOL + _NEWTON_RTOL * (r_ref or 1.0):
                return True, r_ref

        grew = rprev is not None and rnorm > rprev
        if (inverted or grew) and last_du is not None and halvings < _MAX_HALVINGS:
            last_du *= 0.5                      # backtrack: retreat half of the last step
            field.x.array[:] = _values(field, real) + last_du
            halvings += 1
            continue
        halvings = 0

        if rprev is None or rnorm > _FROZEN_RATIO * rprev:
            refresh_jacobian()
        ksp.solve(b, x)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        if mpc is None:
            step = x.array
        else:
            fempetsc.assign(x, du)
            mpc.homogenize(du)
            mpc.backsubstitution(du)
            step = du.x.array
        last_du = (step.real if real else step).copy()
        field.x.array[:] = _values(field, real) - last_du
        rprev = rnorm
    return False, r_ref


def _values(field, real):
    """The dof values of ``field``, their real part when ``real``."""
    return field.x.array.real if real else field.x.array


def root_step(J, resid):
    """Newton step ``J^{-1} resid`` of a free-lateral root-find.

    Raises
    ------
    ConvergenceError
        If ``J`` is singular, so the root-find cannot take its step.
    """
    try:
        return np.linalg.solve(J, resid)
    except np.linalg.LinAlgError as e:
        raise ConvergenceError(f"free-lateral root-find did not converge: {e}") from e


class FreeLateralRamp:
    """Ramp a macro state over increments, solving its free entries for zero average stress.

    The free entries are found by an outer root-find whose macro Jacobian is bootstrapped by
    finite differences and kept current by Broyden rank-1 updates across outer iterations,
    increments and calls. It is dropped when the set of free axes changes.

    Parameters
    ----------
    free_index : callable
        ``free_index(axis)`` is the index into the macro state of the entry conjugate to the
        average normal stress on ``axis``.
    """

    def __init__(self, free_index):
        self._free_index = free_index
        self._J = None
        self._axes = None

    def run(self, start, target, free_axes, n_incr, equilibrate, average_stress, stress_scale,
            commit=None):
        """Ramp from ``start`` to ``target`` in ``n_incr`` equal increments.

        Parameters
        ----------
        start : numpy.ndarray
            The committed macro state.
        target : numpy.ndarray
            This step's macro state; its free entries are replaced by the root-find.
        free_axes : list of int
            The axes whose average normal stress must vanish.
        n_incr : int
            The number of increments.
        equilibrate : callable
            ``equilibrate(M) -> bool`` solves at macro state ``M`` and returns whether the
            solve met its tolerance.
        average_stress : callable
            ``average_stress(M)`` is the volume-averaged stress ``(dim, dim)`` of the last
            solve at ``M``.
        stress_scale : float
            The modulus that turns ``max |target|`` into the stress scale of ``_OUTER_TOL``.
        commit : callable, optional
            ``commit(M)`` is called after each converged increment.

        Returns
        -------
        numpy.ndarray
            The converged macro state of the last increment.

        Raises
        ------
        ConvergenceError
            If the root-find or a solve does not converge in some increment.
        """
        idx = [self._free_index(b) for b in free_axes]
        free_val = np.array([start[i] for i in idx])
        if free_axes != self._axes:
            self._J, self._axes = None, free_axes

        M = target.copy()
        for inc in range(1, n_incr + 1):
            frac = inc / n_incr
            M_drv = start + frac * (target - start)
            prev_val = prev_resid = None  # secant pairs are only valid at a fixed M_drv
            for it in range(_MAX_OUTER):
                M = M_drv.copy()
                for k, i in enumerate(idx):
                    M[i] = free_val[k]
                converged = equilibrate(M)
                if not free_axes:
                    break
                sbar = average_stress(M)
                resid = np.array([sbar[b, b] for b in free_axes])
                scale = max(np.abs(target).max() * stress_scale, 1.0)
                if np.max(np.abs(resid)) <= _OUTER_TOL * scale:
                    break
                if self._J is not None and prev_val is not None:
                    dv = free_val - prev_val  # Broyden rank-1 secant update
                    denom = float(dv @ dv)
                    if denom > 0.0:
                        self._J += np.outer((resid - prev_resid) - self._J @ dv, dv) / denom
                if self._J is None or it == _OUTER_FD_RETRY:
                    self._J = self._jacobian_fd(M, free_axes, idx, resid, equilibrate,
                                                average_stress)
                prev_val, prev_resid = free_val.copy(), resid
                free_val = free_val - root_step(self._J, resid)
            else:
                raise ConvergenceError(
                    f"free-lateral root-find did not converge in increment {inc}/{n_incr}")
            if not converged:
                raise ConvergenceError(
                    f"Newton solve did not converge in increment {inc}/{n_incr}")
            if commit is not None:
                commit(M)
        return M

    @staticmethod
    def _jacobian_fd(M, free_axes, idx, base_r, equilibrate, average_stress):
        """FD bootstrap of the macro Jacobian: one warm-started solve per free entry, against
        the residual ``base_r`` already converged at ``M``. No restore solve: the outer loop's
        next solve re-converges from the perturbed warm start."""
        n = len(free_axes)
        J = np.empty((n, n))
        for k, i in enumerate(idx):
            Mp = M.copy()
            Mp[i] += _MACRO_FD_STEP
            equilibrate(Mp)
            sbar = average_stress(Mp)
            J[:, k] = (np.array([sbar[b, b] for b in free_axes]) - base_r) / _MACRO_FD_STEP
        return J
