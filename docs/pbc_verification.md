# Verifying the periodic constraint

How the corner-driven `PeriodicBoundaryConstraint` was verified against the constraint
it replaced, and what that says about verifying any reformulated constraint system.

## The two formulations

The original `OldPeriodicBoundaryCondition` drove the cell through a virtual `DRIVE`
node, hard-wired `Y0 ≡ Y1`, and could impose confined loading only. It had been
validated by `.inp`-text equivalence to an earlier artifact.

The corner-driven constraint carries the macro strain on the reference corners (X0Y0,
X1Y0, X0Y1 in 2D). The constraint never eliminates those corners; the far corner X1Y1
is a dependent node. Confined or free-lateral loading follows only from how the caller
fixes or drives the corners.

## Compare the observable, not the deck

`.inp`-text equivalence is the wrong oracle for a reformulated constraint system: the
equations differ algebraically by design. The right oracle is the physical observable,
the homogenized `E*(f)`, which is invariant when two correct constraint sets encode the
same loading. The DOLFINx backend is verified against ABAQUS the same way
(`dolfinx_backend_design.md`, section 6).

## Which columns are comparable

The drive node set is `X1Y0`. The reaction work-conjugate to the applied displacement is
the x column only (`RF_Real1`, `RF_Imag1`, `U1`). The lateral (y) reaction lives on a
different corner in each formulation and is not comparable, so only the macro x columns
are compared.

## A structural check

The coefficients of a pure periodic `*Equation` sum to zero, which is translation
invariance. The corner-driven equation has coefficients (1, -1, -1, 1). The old
`DriveEquation` summed to 1.

## Result

Over the 30-frequency example sweep, the corner-driven constraint with confined corners
reproduces the old formulation's x modulus to about 7e-8. With the lateral corner
floating the cell is about 0.75x as stiff, the Poisson relief.

`example/example-youngs.tsv`, where it survives untracked, is a stale oracle from the
old formulation: the same curve shape with a scale offset of about 5.4% from later fixes
to that formulation. It is a historical reference only.

## Where the harness lives

The comparison harness `verify_pbc.py` and both formulations exist on the `pbc` branch
only. The old class, the harness and the separate equation classes were deleted when
boundary conditions became `BoundaryCondition(target, Fixed | Prescribed)`, so the
old-versus-new comparison is runnable only from that branch. On current branches the
corner-driven scheme is pinned by the ABAQUS parity suite
(`tests/test_matrix_parity.py`).
