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
    """Return a list of [frequency, RF_Real..., RF_Imag..., U_Real...] rows.

    One row per non-zero-frequency frame across every step; the reaction force and
    displacement are summed over the nodes of the drive node set. Raises RuntimeError
    if no dynamic (non-zero-frequency) frames exist.
    """
    odb = openOdb(name + ".odb", readOnly=True)
    drive_nset = odb.rootAssembly.instances["PART-1-1"].nodeSets[drive_nodeset.upper()]
    step_results = []
    for step in odb.steps.values():
        for frame in step.frames:
            frequency = frame.frameValue
            if frequency == 0:
                continue

            U = frame.fieldOutputs["U"].getSubset(region=drive_nset, position=NODAL)
            U_Real = np.zeros_like(U.values[0].data)
            for v in U.values:
                U_Real += v.data

            RF = frame.fieldOutputs["RF"].getSubset(region=drive_nset, position=NODAL)
            RF_Real = np.zeros_like(RF.values[0].data)
            RF_Imag = RF_Real.copy()
            for v in RF.values:
                RF_Real += v.data
                if v.conjugateData is not None:  # Dynamic data only
                    RF_Imag += v.conjugateData
            step_results.append(np.concatenate(([frequency], RF_Real, RF_Imag, U_Real)))

    if not step_results:
        raise RuntimeError("no dynamic frames in any step")
    return step_results


def write_tsv(name, rows):
    """Write the reader rows to ``<name>-reaction-force.tsv`` with a column header."""
    ncomp = (len(rows[0]) - 1) // 3  # frequency + RF_Real + RF_Imag + U per component
    header = ["frequency"]
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
