"""The linear solve: one MUMPS LU factorization reused across frequencies and RHS.

The direct solver is MUMPS, a multifrontal sparse LU; why it and not PETSc's native serial LU is in
docs/dolfinx_backend_design.md#10-iterative-solver-gmresilu-crossover-for-large-3d-adopted. The
matrix is re-assembled per frequency (same sparsity -> LU re-factors values only); the
per-unit-strain right-hand sides are cheap back-substitutions. MUMPS factors on the native thread
pools, which every in-process backend call caps at its ``threads`` (the ``_run`` module docstring).

Cancellation. A direct LU ``ksp.solve()`` is a single C call with no Python breakpoint
and no monitor callback (PREONLY+LU never fires ``setMonitor``), so it can only be
cancelled *between* whole solves -- the orchestration loop in ``_run.py`` polls the user
``cancel`` predicate between frequencies. That is acceptable only while a single solve
stays short; ``select_solver_kind`` predicts the LU time from problem size and, above
``LU_TIME_S``, picks ``IterativeSolver`` (GMRES+ILU) instead, whose per-KSP-iteration
monitor *can* poll the callback (sub-second latency).
"""
from __future__ import annotations

from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc as fempetsc
import dolfinx_mpc

from . import _arithmetic as arithmetic


class Cancelled(RuntimeError):
    """Raised when a user ``cancel()`` predicate returns True during a sweep or solve."""


class ConvergenceError(RuntimeError):
    """Raised when a solve ends without meeting its convergence test; no result is returned."""


class ScalarBuildError(RuntimeError):
    """Raised when a solve needs complex arithmetic and this process's PETSc is the real build."""


COMPLEX_BUILD = arithmetic.is_complex_scalar(PETSc.ScalarType)
"""True iff this process's PETSc is the complex-scalar build."""


def require_complex_build():
    """Raise ``ScalarBuildError`` unless this process can solve in complex arithmetic."""
    if not COMPLEX_BUILD:
        raise ScalarBuildError(arithmetic.refusal_message())


# --- LU cost model -----------------------------------------------------------------
# A power law in ndof per dimension, coefficient and exponent fit to the single-core MUMPS times
# of tools/bench_solvers.py; rerun it to recalibrate on other hardware.
# ndof = n_nodes * dim for the vector P1 space.
LU_TIME_S = 10.0  # crossover budget: above this predicted LU time, prefer the iterative solver
# TODO: _COST was fit at one thread in the complex build; the LU-versus-iterative choice of
# select_solver_kind ignores both the threads cap and the scalar build.
_COST = {2: (1.70e-6, 1.21), 3: (6.96e-7, 1.61)}  # dim -> (coefficient, exponent)


def predict_lu_seconds(ndof, dim):
    """Predict the single-core LU factor+solve time (seconds) for ``ndof`` unknowns.

    See the module ``_COST`` calibration note. Treat the prediction as an order-of-magnitude
    crossover signal, not a timer.
    """
    if dim not in _COST:
        raise ValueError(f"unsupported dim {dim!r}; expected 2 or 3")
    coeff, expo = _COST[dim]
    return coeff * float(ndof) ** expo


def select_solver_kind(ndof, dim):
    """Pick ``"lu"`` or ``"iterative"`` for ``solver="auto"``.

    - **2D -> always LU.** GMRES+ILU is slower where it converges and diverges on large 2D
      meshes, so ``auto`` never picks it here.
    - **3D -> iterative above the LU budget** ``LU_TIME_S``, where LU fill explodes and
      GMRES+ILU stays cheap.

    The benchmark behind both rules is in
    docs/dolfinx_backend_design.md#10-iterative-solver-gmresilu-crossover-for-large-3d-adopted. The
    complex-symmetric system has no AMG (GAMG/hypre are real-only), so iterative is
    ILU-preconditioned; callers wanting it in 2D anyway can force ``solver="iterative"``.
    """
    if dim == 2:
        return "lu"
    return "lu" if predict_lu_seconds(ndof, dim) <= LU_TIME_S else "iterative"


def lu_ksp(A, comm):
    """A direct solver for ``A``: a MUMPS LU factorization (see the module docstring)."""
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    return ksp


class LuSolver:
    """Direct (``lu_ksp``) solve of the periodic fluctuation per unit macro strain, refactored per
    frequency."""

    def __init__(self, space, forms, matfields, mpc, bcs):
        self.space = space
        self.forms = forms
        self.matfields = matfields
        self.mpc = mpc
        self.bcs = bcs
        V = space.V
        self.uh = [fem.Function(V) for _ in range(len(forms.L_forms))]

        matfields.set_moduli_elastic()  # placeholder values to allocate A
        self.A = dolfinx_mpc.assemble_matrix(forms.a_form, mpc, bcs=bcs)
        self.A.assemble()
        self.b = dolfinx_mpc.assemble_vector(forms.L_forms[0], mpc)
        self.x = self.A.createVecRight()
        self.ksp = lu_ksp(self.A, space.mesh.comm)

    def _refactor(self):
        self.A.zeroEntries()
        dolfinx_mpc.assemble_matrix(self.forms.a_form, self.mpc, bcs=self.bcs, A=self.A)
        self.A.assemble()
        self.ksp.setOperators(self.A)  # same nonzero pattern -> values-only refactor

    def reassemble(self, f):
        """Re-fill the modulus fields at frequency ``f`` and refactor the matrix."""
        self.matfields.set_moduli(f)
        self._refactor()

    def reassemble_elastic(self):
        """Re-fill with the real *Elastic moduli (Static step) and refactor."""
        self.matfields.set_moduli_elastic()
        self._refactor()

    def solve(self, j):
        """Solve the fluctuation for unit macro strain ``j`` into ``uh[j]``."""
        L_form = self.forms.L_forms[j]
        uout = self.uh[j]
        b, x = self.b, self.x
        with b.localForm() as bl:
            bl.set(0.0)
        dolfinx_mpc.assemble_vector(L_form, self.mpc, b)
        dolfinx_mpc.apply_lifting(b, [self.forms.a_form], [self.bcs], self.mpc)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fempetsc.set_bc(b, self.bcs)
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.ksp.solve(b, x)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        fempetsc.assign(x, uout)
        self.mpc.homogenize(uout)
        self.mpc.backsubstitution(uout)
        return uout


class IterativeSolver(LuSolver):
    """GMRES + ILU iterative solver for meshes above the LU time budget, and the cancellable
    path (the KSP monitor polls ``cancel`` every iteration -- unlike LU's single
    uninterruptible C call, which is cancellable only between frequencies).

    Inherits LuSolver's MPC assemble/solve plumbing unchanged (``reassemble`` already calls
    ``setOperators``, so the ILU is rebuilt per frequency); only the KSP/PC is swapped.

    A frequency-domain system is **complex-symmetric** (complex moduli), so PETSc's algebraic multigrid
    (GAMG/hypre) is unavailable -- those are real-only -- and CG is invalid. This uses GMRES with an
    ILU preconditioner, which works in complex but does not coarsen like AMG, so it can stagnate on
    the hardest large-3D cases. ``solve`` raises ``ConvergenceError`` on non-convergence (it never
    returns silently-wrong numbers); ``petsc_options`` lets a caller swap the preconditioner (e.g.
    ``pc_type='bjacobi'``/``'asm'``, more ILU fill) or one can fall back to ``solver='lu'``.
    """

    _DEFAULT_OPTS = {
        "ksp_type": "gmres",
        "ksp_rtol": 1e-8,
        "ksp_max_it": 1000,
        "ksp_gmres_restart": 100,
        "pc_type": "ilu",
    }

    def __init__(self, space, forms, matfields, mpc, bcs, cancel=None, petsc_options=None):
        super().__init__(space, forms, matfields, mpc, bcs)  # builds A, b, x, ksp (preonly/lu)
        opts = dict(self._DEFAULT_OPTS)
        if petsc_options:
            opts.update(petsc_options)
        # Configure the inherited KSP via a unique options prefix (the dolfinx_mpc
        # LinearProblem pattern), then scrub the keys so they don't accumulate in the global
        # PETSc options database across solver instances.
        prefix = f"msve_iter_{id(self)}_"
        self.ksp.setOptionsPrefix(prefix)
        db = PETSc.Options()
        for k, v in opts.items():
            db[prefix + k] = v
        self.ksp.setFromOptions()
        for k in opts:
            db.delValue(prefix + k)

        self._cancel = cancel
        # petsc4py marshals exceptions raised in a callback back out of ksp.solve(), so a
        # Cancelled raised here surfaces from the solve and unwinds cleanly.
        self.ksp.setMonitor(self._cancel_monitor)

    def _cancel_monitor(self, ksp, its, rnorm):
        if self._cancel is not None and self._cancel():
            raise Cancelled("cancelled by callback")

    def solve(self, j):
        uout = super().solve(j)  # same MPC assemble/solve/back-substitute; iterative KSP
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise ConvergenceError(
                f"IterativeSolver did not converge (KSP reason {reason}). The complex-"
                "symmetric system has no AMG preconditioner (GAMG/hypre are real-only), so "
                "this uses GMRES+ILU, which can stagnate on hard/large 3D meshes. Try a "
                "different `petsc_options` (e.g. pc_type='bjacobi' or 'asm', more ILU fill), "
                "a smaller mesh, or solver='lu'."
            )
        return uout
