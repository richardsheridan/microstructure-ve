"""The linear solve: one native-LU factorization reused across frequencies and RHS.

PETSc's built-in serial LU is used (MUMPS is ~25x slower on these small complex
systems). The matrix is re-assembled per frequency (same sparsity -> LU re-factors
values only); the per-unit-strain right-hand sides are cheap back-substitutions.
"""
from __future__ import annotations

from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc as fempetsc
import dolfinx_mpc


class Solver:
    def __init__(self, space, forms, matfields, mpc, bcs):
        self.space = space
        self.forms = forms
        self.matfields = matfields
        self.mpc = mpc
        self.bcs = bcs
        V = space.V
        self.uh = [fem.Function(V) for _ in range(space.dim)]

        matfields.set_moduli(1.0)  # placeholder values to allocate A
        self.A = dolfinx_mpc.assemble_matrix(forms.a_form, mpc, bcs=bcs)
        self.A.assemble()
        self.b = dolfinx_mpc.assemble_vector(forms.L_forms[0], mpc)
        self.x = self.A.createVecRight()
        self.ksp = PETSc.KSP().create(space.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")  # native serial LU (~25x faster than MUMPS here)

    def reassemble(self, f):
        """Re-fill the modulus fields at frequency ``f`` and refactor the matrix."""
        self.matfields.set_moduli(f)
        self.A.zeroEntries()
        dolfinx_mpc.assemble_matrix(self.forms.a_form, self.mpc, bcs=self.bcs, A=self.A)
        self.A.assemble()
        self.ksp.setOperators(self.A)  # same nonzero pattern -> values-only refactor

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
