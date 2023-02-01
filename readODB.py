"""Extract viscoelastic response from abaqus output ODB to TSV"""
import sys

import numpy as np

from odbAccess import openOdb
from abaqusConstants import NODAL

# abaqus trims sys.argv until after "python"
this_script, name, drive_nodeset = sys.argv
# NOTE: This code cannot know what the proper units for frequency or RF are,
#       as it depends on the interpretation of the inputs in the INP file.
# TODO: add an extra column for step number?
odb = openOdb(name + ".odb", readOnly=True)
drive_nset = odb.rootAssembly.instances["PART-1-1"].nodeSets[drive_nodeset.upper()]
step_results = []
for step in odb.steps.values():
    for frame in step.frames:
        frequency = frame.frameValue
        if frequency == 0:
            continue
        RF = frame.fieldOutputs["RF"].getSubset(region=drive_nset, position=NODAL)
        RF_Real = np.zeros_like(RF.values[0].data)
        RF_Imag = RF_Real.copy()
        for v in RF.values:
            RF_Real += v.data
            if v.conjugateData is not None:  # Dynamic data only
                RF_Imag += v.conjugateData
        step_results.append(np.concatenate(([frequency], RF_Real, RF_Imag)))

header = ["frequency"]
for i in range(1, 1 + len(RF_Real)):
    header.append("RF_Real" + str(i))
for i in range(1, 1 + len(RF_Imag)):
    header.append("RF_Imag" + str(i))
header = "\t".join(header)

# Updated to fit numpy 1.6
# print np.__version__
np.savetxt(
    name + "-reaction-force.tsv",
    step_results,
    fmt="%.8e",
    delimiter="\t",
    # comments="",
    # header=header,
)
# add header
filename = name + "-reaction-force.tsv"
with open(filename, 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write(header.rstrip('\r\n') + '\n' + content)