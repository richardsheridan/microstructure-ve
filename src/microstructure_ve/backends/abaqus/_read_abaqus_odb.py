"""Extract viscoelastic response from an ABAQUS output ODB to a TSV.

This module is the single source of truth for the standalone reader script that
``microstructure_ve.backends.write_odb_reader`` emits. It is written to run under the
ABAQUS python interpreter (no f-strings, no package imports, numpy + odbAccess only);
``odbAccess``/``abaqusConstants`` are imported at module top, so importing it outside
ABAQUS requires those names to be stubbed in ``sys.modules`` first (the test does this).

Run standalone under ABAQUS python::

    abaqus python read_abaqus_odb.py <job-name> <drive-nodeset>
"""
import numpy as np

from odbAccess import openOdb
from abaqusConstants import NODAL


def read_odb(name, drive_nodeset):
    """Return a list of [frame_value, RF_Real..., RF_Imag..., U_Real...] rows.

    ``frame_value`` is the frame's independent coordinate -- frequency (Hz) for a
    steady-state Dynamic step, step time for a Static (and future transient/plastic) step.
    One row per loaded frame across every step (the zero base frame -- a perturbation base
    state, or the unloaded static initial frame -- is skipped); the reaction force and
    displacement are summed over the nodes of the drive node set. Raises RuntimeError if
    every frame is at frame value 0 (no loaded frames), or if the drive node set yields
    no U/RF values (misnamed set or instance mismatch).
    """
    odb = openOdb(name + ".odb", readOnly=True)
    drive_nset = odb.rootAssembly.instances["PART-1-1"].nodeSets[drive_nodeset.upper()]
    step_results = []
    for step in odb.steps.values():
        frames = [f for f in step.frames if f.frameValue != 0]
        # An auto-incremented *STATIC step (e.g. a finite-strain NLGEOM solve that needed
        # several increments) writes one frame per converged increment; the oracle contract
        # is ONE row per Static step (the FE side reports only the converged step end), so
        # keep the final loaded frame. Dynamic steps keep every frame (one per frequency).
        if frames and step.procedure.upper().startswith("*STATIC"):
            frames = frames[-1:]
        for frame in frames:
            frame_value = frame.frameValue

            U = frame.fieldOutputs["U"].getSubset(region=drive_nset, position=NODAL)
            RF = frame.fieldOutputs["RF"].getSubset(region=drive_nset, position=NODAL)
            if not U.values or not RF.values:
                raise RuntimeError(
                    "drive node set '" + drive_nodeset + "' resolved to no U/RF values"
                    " (misnamed set or instance mismatch?)")
            U_Real = np.zeros_like(U.values[0].data)
            for v in U.values:
                U_Real += v.data

            RF_Real = np.zeros_like(RF.values[0].data)
            RF_Imag = RF_Real.copy()
            for v in RF.values:
                RF_Real += v.data
                if v.conjugateData is not None:  # Dynamic data only
                    RF_Imag += v.conjugateData
            step_results.append(np.concatenate(([frame_value], RF_Real, RF_Imag, U_Real)))

    if not step_results:
        raise RuntimeError("no loaded frames in any step (all at frame value 0)")
    return step_results


def write_tsv(name, rows):
    """Write the reader rows to ``<name>-reaction-force.tsv`` with a column header."""
    ncomp = (len(rows[0]) - 1) // 3  # frame_value + RF_Real + RF_Imag + U per component
    header = ["frame_value"]
    for i in range(1, 1 + ncomp):
        header.append("RF_Real" + str(i))
    for i in range(1, 1 + ncomp):
        header.append("RF_Imag" + str(i))
    for i in range(1, 1 + ncomp):
        header.append("U" + str(i))
    header = "\t".join(header)

    # numpy 1.6 doesn't know how to write headers, so we do it manually
    with open(name + "-reaction-force.tsv", "w") as f:
        f.write(header)
        f.write("\n")
        np.savetxt(f, rows, fmt="%.8e", delimiter="\t")


def main(argv):
    # abaqus trims sys.argv until after "python"
    this_script, name, drive_nodeset = argv
    write_tsv(name, read_odb(name, drive_nodeset))


if __name__ == "__main__":
    import sys

    main(sys.argv)
