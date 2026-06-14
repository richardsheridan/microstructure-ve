"""Regenerate the committed ABAQUS parity oracles under tests/data/.

The FE-vs-ABAQUS parity tests (tests/test_backend_dolfinx_parity.py) compare the DOLFINx
backend against small ABAQUS reaction-force oracles. This script regenerates those
oracles. It writes one job per oracle under abaqus-work/<name>/ (a flat .inp plus a
generated standalone read_abaqus_odb.py), then -- if ABAQUS is available -- runs each job
and copies the resulting tsv into tests/data/.

Run from the repo root with the msve interpreter::

    /path/to/msve/python tools/make_oracles.py

The ABAQUS solve may abort in wrap-up with a harmless glibc signal-6 on some
abq2019/RHEL8 hosts; the analysis still completes and writes the .odb, which the reader
then extracts. Oracles are tiny, so they commit naturally (no force-add).
"""
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests._helpers import oracle_simulation_2d, oracle_simulation_3d  # noqa: E402
from microstructure_ve.backends.abaqus import write_inp, write_odb_reader  # noqa: E402

ABAQUS = "/var/DassaultSystemes/SIMULIA/Commands/abaqus"

JOBS = [
    ("oracle_2d_confined", lambda: oracle_simulation_2d("confined"), "X1Y0"),
    ("oracle_2d_free", lambda: oracle_simulation_2d("free"), "X1Y0"),
    ("oracle_3d_elastic", oracle_simulation_3d, "X1Y0Z0"),
]


def main():
    work_root = REPO / "abaqus-work"
    data = REPO / "tests" / "data"
    have_abaqus = os.path.exists(ABAQUS)
    for name, build, nset in JOBS:
        wd = work_root / name
        wd.mkdir(parents=True, exist_ok=True)
        write_inp(build(), wd / (name + ".inp"))
        write_odb_reader(wd / "read_abaqus_odb.py")
        print(f"wrote {wd / (name + '.inp')}")
        if not have_abaqus:
            print(f"  ABAQUS not found; run manually: cd {wd} && abaqus job={name} "
                  f"interactive && abaqus python read_abaqus_odb.py {name} {nset}")
            continue
        subprocess.run([ABAQUS, f"job={name}", "interactive"], cwd=wd,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([ABAQUS, "python", "read_abaqus_odb.py", name, nset], cwd=wd,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tsv = wd / (name + "-reaction-force.tsv")
        if tsv.exists():
            shutil.copy(tsv, data / (name + ".tsv"))
            print(f"  -> {data / (name + '.tsv')}")
        else:
            print(f"  !! {tsv} not produced; check the ABAQUS run")


if __name__ == "__main__":
    main()
