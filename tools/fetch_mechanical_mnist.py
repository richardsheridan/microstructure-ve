"""Fetch the Mechanical-MNIST in-repo data subset and commit-ready excerpts.

Downloads from github.com/elejeune11/Mechanical-MNIST (MIT license) into a scratch
directory (default /tmp/mechanical-mnist, kept out of the repo):

- ``generate_dataset/input_data/{train,test}_data_{0..9}.txt`` -- 20 sample 28x28
  MNIST bitmaps (the benchmark's own example subset),
- ``data/summary_psi_{train,test}_all.txt`` -- the uniaxial-extension strain-energy
  summaries (one row per MNIST image x 13 displacement steps, baseline-subtracted).

It validates the bitmaps (28x28, integers 0..255), slices rows 0-9 of each summary
(the rows matching the sample bitmaps -- see the README it writes for the pairing
assumption), and writes only those excerpts into ``tests/data/mechanical_mnist/``.
The full summaries and the zipped image datasets are never committed.

Run under the msve env (stdlib + numpy only)::

    python tools/fetch_mechanical_mnist.py [--scratch DIR]
"""
import argparse
import pathlib
import sys
import urllib.request

import numpy as np

RAW = "https://raw.githubusercontent.com/elejeune11/Mechanical-MNIST/master"
BITMAPS = [f"{which}_data_{i}" for which in ("train", "test") for i in range(10)]
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "tests" / "data" / "mechanical_mnist"

README = """\
# Mechanical-MNIST data excerpts

Source: https://github.com/elejeune11/Mechanical-MNIST (MIT license,
Copyright (c) 2019 Emma Lejeune). Fetched verbatim by ``tools/fetch_mechanical_mnist.py``;
see that script for the exact upstream paths.

- ``{train,test}_data_{0..9}.txt`` -- the benchmark's own 20 sample bitmaps
  (``generate_dataset/input_data/``), 28x28 integers 0..255. These are MNIST images
  0-9 of the respective train/test sets (the first ten test digits render as
  7 2 1 0 4 1 4 9 5 9).
- ``summary_psi_{train,test}_first10.txt`` -- rows 0-9 of the benchmark's
  ``data/summary_psi_{train,test}_all.txt`` (10 x 13 each): total strain energy change
  of the uniaxial-extension FEA at the 13 displacement steps
  d = 0, 0.001, 0.01, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14 (domain height 28, so the
  final step is 50% nominal strain), each row baseline-subtracted (column 0 is 0).

Pairing assumption: sample bitmap ``i`` corresponds to summary row ``i``. This cannot
be proven from the text files alone, but the parity run itself is the empirical check --
a wrong pairing produces O(1) disagreement, not the observed mesh-convergence-level
agreement (see ``tools/mechanical_mnist_parity.py``).

## Parity results (tools/mechanical_mnist_parity.py, all 20 bitmaps)

Max/median relative error of the DOLFINx NeoHookean strain energies vs the committed
rows over the d >= 0.1 steps, by ``refine`` (Q1 elements per pixel edge):

| refine | mesh    | max rel | median rel | ~s/bitmap |
|--------|---------|---------|------------|-----------|
| 1      | 28x28   | 1.5e-2  | 1.0e-2     | 0.5       |
| 2      | 56x56   | 4.7e-3  | 2.5e-3     | 2.5       |
| 3      | 84x84   | 1.8e-3  | 5e-4       | 9         |
| 5      | 140x140 | 2.4e-3  | 1e-3       | 30-170    |

Convergence is ~O(h^2) through refine=3 and plateaus at ~1e-3: the benchmark's own
mref=5 unstructured-P2 reference and its nodal degree-1 E field carry comparable
discretization error, so closer agreement is not expected. The d = 0.001 and 0.01
columns are limited by the summaries' 5-decimal fixed-point output (absolute
quantization 1e-5; our absolute deviations there are below that). Orientation
cross-check: flipped vs unflipped bitmaps agree to ~1e-9 (the clamped top/bottom BCs
make the energy mirror-invariant).
"""


def fetch(path, dest):
    if dest.exists():
        return dest
    url = f"{RAW}/{path}"
    print(f"fetching {url}")
    with urllib.request.urlopen(url) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scratch", type=pathlib.Path,
                    default=pathlib.Path("/tmp/mechanical-mnist"))
    args = ap.parse_args(argv)
    args.scratch.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)

    for name in BITMAPS:
        src = fetch(f"generate_dataset/input_data/{name}.txt",
                    args.scratch / f"{name}.txt")
        img = np.loadtxt(src)
        if img.shape != (28, 28):
            sys.exit(f"{name}: expected 28x28, got {img.shape}")
        if not (np.all(img == np.rint(img)) and img.min() >= 0 and img.max() <= 255):
            sys.exit(f"{name}: expected integer values in 0..255")
        (DEST / f"{name}.txt").write_bytes(src.read_bytes())  # verbatim copy

    for which in ("train", "test"):
        src = fetch(f"data/summary_psi_{which}_all.txt",
                    args.scratch / f"summary_psi_{which}_all.txt")
        psi = np.loadtxt(src)
        n_expected = 60000 if which == "train" else 10000
        if psi.shape != (n_expected, 13):
            sys.exit(f"summary_psi_{which}_all: expected ({n_expected}, 13), "
                     f"got {psi.shape}")
        if not np.allclose(psi[:, 0], 0.0):
            sys.exit(f"summary_psi_{which}_all: column 0 should be the subtracted "
                     "d=0 baseline (all zeros)")
        np.savetxt(DEST / f"summary_psi_{which}_first10.txt", psi[:10], fmt="%.8e")

    (DEST / "README.md").write_text(README, encoding="utf-8")
    print(f"excerpts written to {DEST}")


if __name__ == "__main__":
    main()
