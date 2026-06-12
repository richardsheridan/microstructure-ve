# Design: a DOLFINx/FEniCSx backend for microstructure-ve

Status: design document — no backend code yet. First implementation milestone: **example.py
parity** (the 2D corner-driven-PBC harmonic viscoelastic sweep reproducing the ABAQUS `E*(f)`
within tolerance).

Every load-bearing API claim below was smoke-tested on this host against the pinned stack in
the `fenicsx` conda env (created 2026-06-12):

| package | version |
| --- | --- |
| fenics-dolfinx | 0.10.0 |
| dolfinx_mpc | 0.10.5 |
| petsc / petsc4py | 3.25.2 (**complex** build; `PETSc.ScalarType == complex128`) |
| fenics-basix / fenics-ufl | 0.10.0 / 2025.2.1 |
| python / numpy | 3.12.13 / 2.4.6 |

Recreate with:
```
conda create -n fenicsx -c conda-forge python=3.12 fenics-dolfinx dolfinx_mpc 'petsc=*=*complex*' scipy
```

## 0. Implementation status (executed 2026-06-12)

Backend implemented in `dolfinx_backend.py`; driver `example_dolfinx.py`. **One design claim
broke in practice and forced a formulation change**, found by execution:

- **`dolfinx_mpc` silently ignores Dirichlet BCs on MPC *master* dofs.** The corner-driven
  reuse (§4.4) drives the macro strain by Dirichlet-ing reference corners that are exactly
  those masters — so the solve ran, KSP converged, but the corners were never pinned (garbage
  field). The smoke test missed it because it only checked the MPC *relation*, not the
  prescribed corner values.
- **Pivot (standard periodic homogenization):** split `u = E_macro·x + u_per`; a *pure-periodic*
  2-term MPC on the fluctuation `u_per` (slave=image, built from our disjoint `nsets` — exact,
  validated), the macro strain as a RHS source `-∫σ(E_macro):ε(v)`, and one interior node
  pinned for rigid translation. This imposes **confined** loading (E fully prescribed). It is a
  different *route* to the same physical BVP as the confined corner-driven ABAQUS case.
- **Also required:** dolfinx 0.10 `petsc_options` only take effect under a `petsc_options_prefix`;
  without it the solver silently falls back and BCs aren't enforced.

**Implemented features** (`dolfinx_backend.run(sim, freqs, lateral=, bbar=)`):
- **B-bar** — selective reduced integration on the volumetric term (deviatoric full quad +
  `κ·tr(ε_u)·tr(ε_v)` at 1-point quad, `κ = λ + 2μ/d`). Matches ABAQUS CPE4's B-bar formulation;
  removes Q1 volumetric locking.
- **Free-lateral** (`lateral="free"`) — E_yy floats so σ̄_yy = 0, done by **superposition** of the
  E_xx-only and E_yy-only unit solves. The stiffness is identical for both, so the second solve is
  a free back-substitution. Confined (`lateral="confined"`) prescribes E_yy = 0.
- **Performance** — one factorization reused across all frequencies and both RHS. The real
  bottleneck was the solver: **MUMPS is pathologically slow (~9 s) on this small 5202-dof complex
  system; PETSc's built-in serial LU does it in 0.37 s** (25×). Switched to native LU → the full
  30-frequency sweep dropped from **270 s to 15 s (0.5 s/frequency)**.

**Verification result** (50×50 RVE, 30-frequency complex viscoelastic sweep, vs ABAQUS, macro
x-columns of the homogenized response; B-bar on):

| loading | oracle | storage/loss max rel diff | dolfinx/ABAQUS ratio |
| --- | --- | --- | --- |
| confined | CPE4 (full) | **0.08% / 0.14%** | 1.0001 ± 0.0002 |
| free-lateral | CPE4 (full) | **0.08% / 0.14%** | 1.0001 ± 0.0002 |
| confined (no B-bar) | CPE4 | 2.2% | 1.0199 ± 0.0008 |
| confined | CPE4R (reduced) | 3.7% | 1.025 ± 0.006 |

With B-bar, dolfinx matches ABAQUS **CPE4 to <0.1% across all 30 frequencies** for both confined and
free-lateral loading — solver-tolerance agreement of the homogenized `E*(f)`. U1 exact (2e-8). The
earlier ~2% (no B-bar) was confirmed to be pure volumetric locking, and the ~3–4% vs CPE4R is the
expected reduced-vs-full integration gap. Homogeneous analytic checks are exact: confined
σ̄_xx = (λ+2μ)·ε_xx, free-lateral σ̄_xx = 4μ(λ+μ)/(λ+2μ)·ε_xx (plane-strain uniaxial stress),
fluctuation ~1e-18.

**Remaining follow-ups:**
- **Frequency-level parallelism (recommended near-term lever).** The 30-frequency sweep is
  embarrassingly parallel — independent serial solves sharing one mesh. Splitting frequencies across
  cores (`multiprocessing`, or MPI ranks each owning a subset on their *own serial mesh*) gives
  near-linear speedup (~15 s → ~4 s on 4 cores) with **zero distributed-mesh rework**. Note: ABAQUS
  does *not* do this — `cpus=N` parallelizes each frequency's assembly+direct-solve and steps through
  frequencies sequentially (the domain/solver axis, not the frequency axis), so this is an axis ABAQUS
  leaves on the table.
- **MPI domain decomposition — low priority until large 3D meshes.** Only pays off for ≫10⁵–10⁶ dofs;
  on the current ~5k-dof system it would add communication overhead for no gain. Requires rework:
  rank-local node↔dof map (the global `ravel_multi_index` assumption breaks), `comm.allreduce` on the
  volume-averaged-stress and area integrals, a distributed solver (superlu_dist / parallel MUMPS), and
  testing the periodic MPC across partition boundaries.
- 3D; non-square cells; reuse the *symbolic* factorization across frequencies (native LU re-factors in
  0.37 s, so minor).

## 1. Motivation

- **License-free solves.** ABAQUS runs consume DSLS tokens (~25/solve) and serialize behind the
  license server; DOLFINx solves are free and parallel-friendly.
- **CI-able physics tests.** The PBC verification strategy (compare physical observables, not
  `.inp` text) currently needs ABAQUS in the loop. A DOLFINx backend lets equivalence tests run
  in CI on any machine.
- **Fast iteration for 2D RVEs.** The example RVE (2500 quads, 30 frequencies) is interactive-scale
  for a direct sparse solver.

ABAQUS remains the validated reference backend; nothing in the `.inp` path changes.

## 2. Two directions considered

**A. DOLFINx as a second backend** — keep the existing dataclasses (`GridNodes`, `GridElements`,
`ElementSet`, materials, BCs, `Step`) as the solver-neutral specification and add a builder that
constructs DOLFINx objects from them. **Recommended.**

**B. "Hijack" DOLFINx specifications to emit `.inp`** — author simulations in DOLFINx-native terms
and serialize to ABAQUS input. **Rejected:** DOLFINx has no declarative simulation spec to hijack;
simulations are imperative Python over meshes and UFL forms. Only the *mesh* has a serializable
form, and `meshio` already writes ABAQUS mesh files. Materials, `*Equation` constraints, and steps
have no DOLFINx-native serialized representation — any exporter would have to invent a spec layer,
which is exactly what our dataclasses already are. Direction B collapses into Direction A.

The deciding structural fact for A: the over-constraint validation work (commit `5c825d2` on the
`pbc` branch) refactored constraints into *enumerable data* — `PeriodicBoundaryCondition` builds
its `SequentialDifferenceEquation` objects eagerly in `__post_init__`, and every constraint/BC
exposes `dependent_dofs()` / `prescribed_dofs()` as `(node_inds, dof)` groups. `dolfinx_mpc`'s
general multi-point-constraint API consumes essentially the same shape (slave → {master: coeff}).
The `.inp` emitter and the DOLFINx builder become two consumers of one constraint data model.

## 3. Architecture

- **New module `dolfinx_backend.py`**, importing `dolfinx` lazily (top-level import of
  `microstructure_ve` stays numpy-only; the msve env never needs dolfinx).
- A **builder consuming `Simulation`/`Model`**, not per-class `to_dolfinx` methods. DOLFINx
  construction is not keyword-sequential the way `.inp` emission is: the mesh must exist before
  function spaces, spaces before constraints, constraints before assembly. One
  `run(sim, freqs) -> results` (plus finer-grained pieces for testing) mirrors how dolfinx is used.
- `Model.__post_init__`'s `validate_constraints` runs identically for both backends — the same
  construction-time over-constraint guarantees apply before any solve.

## 4. Mapping (each row smoke-tested where marked ✓)

### 4.1 Mesh: `GridNodes`/`GridElements` → `dolfinx.mesh.create_mesh` ✓

- Coordinates: `np.indices(shape)[::-1] * scale`, raveled — same arrays `GridNodes.to_inp` writes.
- Connectivity: the same slicing `GridElements.to_inp` uses, **without the counterclockwise swap**.
  DOLFINx quadrilaterals use tensor-product ("zigzag") vertex ordering — exactly what
  `itertools.product` produces *before* the swap at `microstructure_ve.py` ~:119. Verified: mesh
  assembles with exact total area `(n·scale)²`.
- **dolfinx 0.10 API trap:** `create_mesh(comm, cells, e, x)` — the coordinate element comes
  *before* the coordinates (0.9 had `(comm, cells, x, e)`). Smoke test hit this.
- **Cells are reordered even in serial** ✓: `mesh.topology.original_cell_index` is *not* the
  identity in 0.10. Anything cell-indexed must map through it (see §4.3).
- Node↔dof map ✓: for P1 vector spaces (`functionspace(mesh, ("Lagrange", 1, (2,)))`), match
  `V.tabulate_dof_coordinates()` to grid coordinates by `rint(xy / scale)` — exact on a structured
  grid. Round-trips all eight 2D `GridNodes.nsets`. Blocked dof for (node, component) =
  `2 * block + comp`.
- The ABAQUS virtual node (`GridNodes.virtual_node`) has no dolfinx counterpart and is not needed
  (it only serves `OldPeriodicBoundaryCondition`'s DRIVE scheme).

### 4.2 Elements: CPE4R ↔ Q1 quadrilaterals (see §5 for parity)

Plane strain: the UFL form must use the plane-strain Lamé parameter
`λ = Eν/((1+ν)(1−2ν))`, `μ = E/(2(1+ν))` ✓ (matches CPE4x semantics; CPS4R would need plane stress).

### 4.3 Materials: `ElementSet` → DG0 fields; factor `complex_modulus(freqs)` out of `to_inp`

- Per-pixel materials become cellwise-constant ("DG", 0) complex fields. **Must assign through
  `original_cell_index`** ✓: `dg0.x.array[:] = per_original_cell_values[oci]` (our element order is
  the raveled pixel order). Verified against `compute_midpoints` — naïve order silently scrambles
  the microstructure.
- **Refactor prerequisite in `microstructure_ve.py`:** extract the master-curve evaluation from
  `TabularViscoelasticMaterial.to_inp` into a reusable method, used by both backends:

  ```python
  def complex_modulus(self, freqs):
      """E*(f) at the excitation frequencies, after shift/broadening."""
      table_freq = self.apply_shift()
      wg, _ = self.normalize_constant_nu_modulus()
      wgr = np.interp(freqs, table_freq, wg.real)   # see interpolation note below
      wgi = np.interp(freqs, table_freq, wg.imag)
      return self.youngs * ((1.0 - wgi) + 1j * wgr)
  ```

  This inverts ABAQUS's tabular convention (`ℜωg* = E_loss/E_long`, `ℑωg* = 1 − E_storage/E_long`,
  with `E_long = *Elastic` modulus = `self.youngs`), so by construction both backends sample the
  *same* shifted/broadened master curve. Plain `Material` → constant real `E`.
- **Interpolation-scheme parity risk:** ABAQUS interpolates `*VISCOELASTIC, FREQUENCY=TABULAR`
  linearly in frequency between table points. The master curve (R10 data) is dense, and the
  `Dynamic` sweep should be evaluated with the same rule (`np.interp` on linear f, not log-f) to
  match. Quantify during implementation; if residual disagreement at low f exceeds tolerance,
  this is the first suspect.
- Constant real ν per material (the constant-ν normalization is already the only path
  `to_inp` uses); density is not needed (§4.5).

### 4.4 Constraints: PBC equations → `dolfinx_mpc` general constraints ✓

Each `SequentialDifferenceEquation` row `u_dep − u_img − u_refHi + u_refLo = 0` becomes a slave
with three masters: `u_dep = u_img + u_refHi − u_refLo`, i.e. masters `{img: +1, refHi: +1,
refLo: −1}` — generated directly from `PeriodicBoundaryCondition.equations` (`nsets[0][i]` slave;
`nsets[1][i]`, `nsets[2]`, `nsets[3]` masters), per component:

```python
mpc = dolfinx_mpc.MultiPointConstraint(V)
# keys are coordinate arrays serialized with .tobytes(); coords computed by the
# same i*scale arithmetic as the mesh geometry -> bitwise-identical floats
for comp in range(dim):
    mpc.create_general_constraint(slave_master_dict, comp, comp)
mpc.finalize()
```

Verified ✓ on the corner relation: complex LU solve via `dolfinx_mpc.LinearProblem(a, L, mpc,
bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})`; constraint residual exactly 0.

- The reference corners (X0Y0, X1Y0, X0Y1) appear only as masters — same invariant
  `validate_constraints` enforces; a model that passes validation maps cleanly onto an MPC.
- Scaling note: the coordinate-bytes dict API is comfortable at example scale (198 equations).
  For very large grids, `dolfinx_mpc`'s lower-level `mpc_data` arrays accept (slave dof, master
  dofs, coeffs) directly — our node↔dof map already provides the dofs, skipping coordinate lookup.
- `FixedBoundaryCondition` / `DisplacementBoundaryCondition` → per-component `dirichletbc` via
  `locate_dofs_geometrical((V.sub(c), Vc), pred)` ✓ (or directly via the node↔dof map). The
  model-level zero baseline + step displacement collapse to one Dirichlet value per frequency
  sweep (perturbation step semantics: the step value is the harmonic amplitude).
- Complex-mode UFL note ✓: `ufl.inner(a, b)` conjugates its second argument itself — do **not**
  add explicit `conj` in forms.

### 4.5 Steps: `Dynamic` + `Step(perturbation=True)` → frequency loop, quasi-static

For each of the `f_count` log-spaced frequencies: rebuild the DG0 modulus fields from
`complex_modulus(f)`, assemble `K(f) u = 0` with the Dirichlet drive, solve. **Omit the mass
matrix:** ABAQUS `*STEADY STATE DYNAMICS, DIRECT` includes `−ω²M`, but with these densities the
inertia-to-stiffness ratio `ρω²L²/E` is ≤ 5.5e−9 at 1e5 Hz (ρ = 2650 kg/m³, L = 0.125 µm,
E ≈ 3 GPa) — below the ~7e−8 noise floor measured in the PBC equivalence runs. Document, don't
model. (If ever needed: unit consistency — lengths µm, stress MPa, density kg/µm³ — must be
rechecked before adding `−ω²M`.)

Only the material fields change between frequencies; the MPC, Dirichlet structure, and sparsity
are frequency-independent and can be set up once.

### 4.6 Observable: volume-averaged complex stress → `E*(f)` tsv

`σ̄ = (1/V) ∫ σ(u) dx` (complex), assembled per frequency. By Hill–Mandel/divergence theorem this
equals the X1Y0 corner-reaction route used with ABAQUS (`readODB.py example X1Y0`):
`E*(f) = σ̄ₓₓ / ε̄ₓₓ` with `ε̄ₓₓ = δ/Lx`. Volume averaging avoids fiddly reaction extraction from
the MPC-condensed system. Write a tsv mirroring `readODB.py`'s columns
(`frequency, RF_Real*, RF_Imag*, U*`, with RF ≡ σ̄·area) so `verify_pbc.compare`'s
macro-x-column comparison works unchanged.

## 5. Element parity: the oracle needs CPE4, not CPE4R

`CPE4R` is reduced-integration with hourglass stabilization; DOLFINx Q1 assembles full
integration. On a coarse RVE these differ beyond solver tolerance, so the tight oracle is:

- extend `GridElements`'s allowed 2D types with plain **`CPE4`** (full integration, plane strain) —
  a one-line validation change;
- generate a CPE4 ABAQUS run of the example as the matched oracle (~30 s solve, license needed
  once);
- expect near-solver-tolerance agreement of `E*(f)` against DOLFINx Q1 (same discretization);
- the existing CPE4R results remain a *mesh-converged* comparison only (document the expected
  small gap rather than chasing it).

## 6. Verification protocol (for the implementation PR)

Same physical-observable strategy as the PBC swap (`.inp`-text equivalence is meaningless across
solvers; `E*(f)` is the oracle):

1. Smoke suite (already passing) kept as `tests/` or scratch: complex build, mesh ordering,
   node↔dof round-trip, 4-term MPC solve, `original_cell_index` DG0 assignment.
2. ABAQUS CPE4 oracle: regenerate example with CPE4, solve, `readODB.py example_cpe4 X1Y0` →
   oracle tsv (one-time, msve + abaqus).
3. DOLFINx run of the identical `Simulation` → tsv; compare macro x-columns over all 30
   frequencies with `verify_pbc.compare`; set `rtol` empirically (expect ≲1e−6; investigate
   anything worse than 1e−4, starting with the interpolation rule of §4.3).
4. Confined-vs-free-lateral cross-check inside DOLFINx alone: free-lateral ~0.75× softer
   (known Poisson-relief factor from the PBC verification).
5. msve regression: `example.py` still runs and `example.inp` is unchanged (the only
   `microstructure_ve.py` edits are the `complex_modulus` factoring and the `CPE4` allowance).

## 7. Risks

- **API churn** (realized twice during smoke testing): 0.10 reordered `create_mesh` arguments and
  reorders cells in serial. Pin the env; re-run the smoke suite on any dolfinx upgrade.
- **Tabular interpolation parity** (§4.3) — most likely source of small `E*(f)` mismatch.
- **MPC + complex at scale** — validated at toy scale; the 198-equation example is the real test.
  Fall back to `mpc_data` arrays if the coordinate-dict API gets slow on big grids.
- **Serial-first** — the node↔dof and `original_cell_index` mappings as designed assume one rank.
  MPI distribution renumbers locally; out of scope for the parity milestone (declare backend
  serial-only initially).
- **Units** — the implied system (µm, MPa, kg/µm³) is consistent for quasi-statics; revisit only
  if inertia or non-harmonic dynamics are ever added.

## 8. Milestone plan (example.py parity)

1. `complex_modulus(freqs)` factoring in `microstructure_ve.py` + `CPE4` in `GridElements`
   (+ regenerate nothing; `.inp` unaffected — verify byte-identical example.inp).
2. `dolfinx_backend.py`: mesh + node↔dof map + DG0 materials (smoke-validated patterns above).
3. Constraints: `PeriodicBoundaryCondition.equations` → MPC; Dirichlet corners.
4. Frequency loop + σ̄ observable + tsv writer.
5. `example_dolfinx.py` mirroring `example.py`; run protocol §6; commit oracle + results.

Each step has an isolated check; nothing merges until §6 passes end-to-end.
