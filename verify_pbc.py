"""Verify the corrected PeriodicBoundaryCondition against OldPeriodicBoundaryCondition.

Builds the *same* RVE (mesh, materials, step) three ways, differing only in the
boundary-condition block:

    old          - OldPeriodicBoundaryCondition, driven via the virtual DRIVE node
    new_confined - PeriodicBoundaryCondition, corners constrained to reproduce old's
                   confined uniaxial loading (zero lateral strain)
    new_free     - PeriodicBoundaryCondition, lateral corner free (Poisson contraction)

`old` and `new_confined` encode the *same* macroscopic loading, so an ABAQUS solve
of each must yield identical reaction forces / displacements at the drive node. That
equivalence -- not any .inp text diff -- is the oracle for the reformulated constraints.

Usage:
    python verify_pbc.py gen        # write example_old.inp / example_new.inp (+ sanity check)
    python verify_pbc.py gen_free   # write example_free.inp
    python verify_pbc.py compare    # after ABAQUS+readODB: assert RF/U allclose
"""
import sys

import numpy as np

from microstructure_ve import (
    Heading,
    GridNodes,
    GridElements,
    ElementSet,
    TabularViscoelasticMaterial,
    Material,
    periodic_assign_intph,
    load_viscoelasticity,
    FixedBoundaryCondition,
    DisplacementBoundaryCondition,
    OldPeriodicBoundaryCondition,
    PeriodicBoundaryCondition,
    Dynamic,
    Step,
    NodeSet,
    Model,
    Simulation,
)

scale = 0.0025
displacement = 0.005
layers = 5


def build_common():
    """Mesh + materials + dynamic step shared by every BC mode."""
    ms_img = np.load("ms.npy")
    intph_img = periodic_assign_intph(ms_img, [layers])

    freq, youngs_cplx = load_viscoelasticity("PMMA_shifted_R10_data.txt")
    youngs_plat = youngs_cplx[0].real

    nodes = GridNodes.from_matl_img(intph_img, scale)
    elements = GridElements(nodes, type="CPE4R")
    filler_elset, intph_elset, mat_elset = ElementSet.from_matl_img(intph_img)

    materials = [
        Material(filler_elset, density=2.65e-15, youngs=5e5, poisson=0.15),
        TabularViscoelasticMaterial(
            intph_elset,
            density=1.18e-15,
            poisson=0.35,
            shift=-4.0,
            youngs=youngs_plat,
            freq=freq,
            youngs_cplx=youngs_cplx,
            left_broadening=1.8,
            right_broadening=1.5,
        ),
        TabularViscoelasticMaterial(
            mat_elset,
            density=1.18e-15,
            poisson=0.35,
            youngs=youngs_plat,
            freq=freq,
            youngs_cplx=youngs_cplx,
            shift=-6.0,
        ),
    ]
    dyn = Dynamic(f_initial=1e-7, f_final=1e5, f_count=30, bias=1)
    return nodes, elements, materials, dyn


def build_sim(mode):
    nodes, elements, materials, dyn = build_common()

    if mode == "old":
        # Old PBC drives a virtual node and carries its own model-level
        # DRIVE,1,1,0.0 baseline (it subclasses DisplacementBoundaryCondition).
        drive = NodeSet("DRIVE", [nodes.virtual_node])
        bcs = [
            OldPeriodicBoundaryCondition(
                nodes=nodes, nset=drive, first_dof=1, last_dof=1, displacement=0.0
            )
        ]
        model_nsets = [drive]
    else:
        # Corner-driven periodic BCs. The corners X0Y0, X1Y0, X0Y1 are the macro
        # DOFs that PeriodicBoundaryCondition leaves unconstrained; X1Y0 is driven.
        x0y0 = nodes.nsets["X0Y0"]
        x0y1 = nodes.nsets["X0Y1"]
        drive = nodes.nsets["X1Y0"]
        bcs = [
            PeriodicBoundaryCondition(nodes=nodes),
            FixedBoundaryCondition(x0y0, dofs=[1, 2]),  # pin origin
            FixedBoundaryCondition(drive, dofs=[2]),  # no xy shear
        ]
        if mode == "new_confined":
            # zero lateral strain -> matches OldPeriodicBoundaryCondition
            bcs.append(FixedBoundaryCondition(x0y1, dofs=[1, 2]))
        elif mode == "new_free":
            # traction-free lateral -> Poisson contraction allowed
            bcs.append(FixedBoundaryCondition(x0y1, dofs=[1]))
        else:
            raise ValueError("unknown mode", mode)
        # explicit baseline matching old's built-in model-level DRIVE,1,1,0.0
        bcs.append(DisplacementBoundaryCondition(drive, 1, 1, 0.0))
        model_nsets = []

    # Every mode drives dof 1 of the same `drive` node by `displacement` in the step.
    disp_bc = DisplacementBoundaryCondition(
        drive, first_dof=1, last_dof=1, displacement=displacement
    )
    model = Model(
        nodes=nodes,
        nsets=model_nsets,
        elements=elements,
        materials=materials,
        bcs=bcs,
    )
    step = Step(subsections=[dyn, disp_bc], perturbation=True)
    sim = Simulation(
        heading=Heading("PBC verification RVE"), model=model, steps=[step]
    )
    return sim, nodes


def _parse_equations(inp_path):
    """Yield lists of (token, dof, coeff) for each *Equation block in the file."""
    with open(inp_path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().upper() == "*EQUATION":
            n = int(lines[i + 1])
            terms = []
            for j in range(i + 2, i + 2 + n):
                tok, dof, coeff = (s.strip() for s in lines[j].split(","))
                terms.append((tok, int(dof), float(coeff)))
            yield terms
            i += 2 + n
        else:
            i += 1


def sanity_check(inp_path, nodes):
    """ABAQUS-free pre-flight on a *new* PBC .inp (pure periodic constraints).

    Invariants: every equation's coefficients sum to 0 (translation-invariance);
    no (dependent token, dof) is constrained twice; the macro reference corners
    are never a dependent (first) term.
    """
    ref_corners = {"X0Y0", "X1Y0", "X0Y1"}
    # node numbers of the reference corners, to also catch them as integer deps
    ref_nums = {str(nodes.nsets[c].node_inds[0]) for c in ref_corners}

    problems = []
    seen_dep = set()
    n_eq = 0
    for terms in _parse_equations(inp_path):
        n_eq += 1
        coeff_sum = sum(t[2] for t in terms)
        if abs(coeff_sum) > 1e-9:
            problems.append(f"coeffs sum to {coeff_sum} (not 0): {terms}")
        dep_tok, dep_dof, _ = terms[0]
        key = (dep_tok, dep_dof)
        if key in seen_dep:
            problems.append(f"{key} is a dependent term in >1 equation")
        seen_dep.add(key)
        if dep_tok in ref_corners or dep_tok in ref_nums:
            problems.append(f"reference corner {dep_tok} used as dependent term")

    print(f"  {inp_path}: {n_eq} *Equation blocks")
    if problems:
        print("  SANITY FAILURES:")
        for p in problems[:10]:
            print("    -", p)
        return False
    print("  sanity OK (coeffs sum to 0, no double-constraint, corners free)")
    return True


def compare(old_tsv, new_tsv, rtol=1e-4, atol=1e-12):
    """Assert the homogenized x-response matches between old and new-confined.

    Columns are: frequency, RF_Real1, RF_Real2, RF_Imag1, RF_Imag2, U1, U2.
    Only the x-direction macro DOF (the driven direction) is a formulation-
    independent observable: the storage/loss force RF*_1 work-conjugate to the
    applied U1. The lateral reaction RF*_2 lives on a different node in each
    scheme (old: virtual DRIVE has no y-coupling -> 0; new: X1Y0 carries the
    confining force), so it is NOT comparable and is excluded by design.
    """
    macro = [1, 3, 5]  # RF_Real1 (storage), RF_Imag1 (loss), U1 (applied disp)
    old = np.loadtxt(old_tsv, skiprows=1)
    new = np.loadtxt(new_tsv, skiprows=1)
    if old.shape != new.shape:
        raise SystemExit(f"shape mismatch: old {old.shape} vs new {new.shape}")
    old = old[np.argsort(old[:, 0])]
    new = new[np.argsort(new[:, 0])]
    if not np.allclose(old[:, 0], new[:, 0], rtol=1e-6):
        raise SystemExit("frequency columns differ")
    ok = np.allclose(old[:, macro], new[:, macro], rtol=rtol, atol=atol)
    maxrel = np.max(
        np.abs(old[:, macro] - new[:, macro]) / (np.abs(old[:, macro]) + atol)
    )
    print(f"max relative diff in macro x-response (RF_Real1, RF_Imag1, U1): {maxrel:.3e}")
    if ok:
        print("PASS: new-confined reproduces old x-modulus -> PBC verified")
    else:
        print("FAIL: macro x-response differs beyond tolerance")
    return ok


def write_inp(mode, path):
    """Build `mode` and write its .inp; sanity-check the pure-periodic new modes."""
    sim, nodes = build_sim(mode)
    with open(path, "w", encoding="ascii") as f:
        sim.to_inp(f)
    print(f"wrote {path} ({mode})")
    if mode.startswith("new"):  # old's DriveEquation isn't pure-periodic, so skip it
        sanity_check(path, nodes)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if cmd == "gen":
        write_inp("old", "example_old.inp")
        write_inp("new_confined", "example_new.inp")
    elif cmd == "gen_free":
        write_inp("new_free", "example_free.inp")
    elif cmd == "compare":
        old_tsv = sys.argv[2] if len(sys.argv) > 2 else "example_old-reaction-force.tsv"
        new_tsv = sys.argv[3] if len(sys.argv) > 3 else "example_new-reaction-force.tsv"
        ok = compare(old_tsv, new_tsv)
        sys.exit(0 if ok else 1)
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
