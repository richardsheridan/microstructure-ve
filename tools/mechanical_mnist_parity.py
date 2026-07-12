"""Mechanical-MNIST parity study: solve the 20 committed sample bitmaps, compare psi.

For every committed sample bitmap (tests/data/mechanical_mnist/), run the uniaxial-
extension benchmark through the DOLFINx finite-strain NeoHookean solver at each
requested refinement (elements per pixel edge) and compare the per-step strain-energy
change against the benchmark's committed summary rows.

The benchmark's summary files are fixed-point with 5 decimals, so comparisons carry an
absolute floor of 1e-5 (their d=0.001 column rounds to exactly 0). The reported
relative errors use max(|theirs|, 1e-5) in the denominator.

Run under the fenicsx env::

    python tools/mechanical_mnist_parity.py --refine 1 2 3 --set all --json out.json
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests._mechanical_mnist import DISP_VALS, load_bitmap, load_psi, mm_simulation  # noqa: E402

QUANT = 1e-5  # the summary files are %0.5f fixed-point; half-ULP absolute floor


def run_bitmap(run, bitmap, refine, n_incr, flip, independent):
    """(13,) strain-energy change per displacement step for one bitmap."""
    if independent:
        # 13 single-step sims (path independence makes this equivalent; slower, no
        # warm start): a cross-check for the multistep ramp
        psi = []
        for d in DISP_VALS:
            _, e = run.run(mm_simulation(bitmap, disp_vals=[d], refine=refine, flip=flip),
                           return_energy=True, n_incr=max(n_incr, 2 * int(1 + d)))
            psi.append(e[0])
        psi = np.array(psi)
    else:
        _, psi = run.run(mm_simulation(bitmap, refine=refine, flip=flip),
                         return_energy=True, n_incr=n_incr)
    return psi - psi[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refine", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--set", dest="which", choices=("train", "test", "all"), default="all")
    ap.add_argument("--n-incr", type=int, default=2)
    ap.add_argument("--independent", action="store_true",
                    help="13 single-step sims per bitmap instead of the warm-started ramp")
    ap.add_argument("--no-flip", dest="flip", action="store_false",
                    help="skip the vertical bitmap flip (energy is flip-invariant; "
                         "this is an orientation cross-check)")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    from microstructure_ve.backends.dolfinx import _run as run

    sets = ("train", "test") if args.which == "all" else (args.which,)
    results = {"disp_vals": list(DISP_VALS), "quant_floor": QUANT, "runs": []}
    for refine in args.refine:
        errs = []
        print(f"\n=== refine={refine} ({28 * refine}x{28 * refine} elements) ===")
        for which in sets:
            ref_rows = load_psi(which)
            for i in range(10):
                name = f"{which}_data_{i}"
                t0 = time.perf_counter()
                dpsi = run_bitmap(run, load_bitmap(name), refine, args.n_incr,
                                  args.flip, args.independent)
                dt = time.perf_counter() - t0
                ref = ref_rows[i]
                rel = np.abs(dpsi - ref) / np.maximum(np.abs(ref), QUANT)
                errs.append(rel)
                results["runs"].append({
                    "name": name, "refine": refine, "seconds": round(dt, 2),
                    "dpsi": dpsi.tolist(), "ref": ref.tolist(), "rel": rel.tolist(),
                })
                print(f"  {name}: {dt:6.1f}s  max rel = {rel.max():.3e}  "
                      f"(d>=0.1: {rel[3:].max():.3e})")
        errs = np.array(errs)
        print(f"  -- refine={refine} summary over {len(errs)} bitmaps --")
        print("  d       " + "  ".join(f"{d:8.3g}" for d in DISP_VALS[1:]))
        print("  max rel " + "  ".join(f"{v:8.1e}" for v in errs.max(axis=0)[1:]))
        print("  med rel " + "  ".join(f"{v:8.1e}" for v in np.median(errs, axis=0)[1:]))

    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
