"""Extract viscoelastic response from abaqus output ODB to TSV"""
import sys
import csv

from odbAccess import openOdb
from abaqusConstants import NODAL

# abaqus trims sys.argv until after "python"
this_script, name = sys.argv
tsv = csv.writer(open(name+'-reaction-force.tsv', 'wb'), dialect=csv.excel_tab)
# NOTE: This code cannot know what the proper units for frequency or RF are,
#       as it depends on the interpretation of the inputs in the INP file.
tsv.writerow(('frequency', 'RF_Real', 'RF_Imag'))
odb = openOdb(name+'.odb', readOnly=True)
drive_nset = odb.rootAssembly.instances['PART-1-1'].nodeSets['DRIVE0']
for frame in odb.steps['STEP-1'].frames:
	frequency = frame.frameValue
	if frequency == 0:
		continue
	RF = frame.fieldOutputs['RF'].getSubset(region=drive_nset, position=NODAL)
	RF_Real = 0
	RF_Imag = 0
	for v in RF.values:
		RF_Real += v.data[0]
		RF_Imag += v.conjugateData[0]
	tsv.writerow((frequency, RF_Real, RF_Imag))
