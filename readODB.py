"""Extract viscoelastic response from abaqus output ODB to TSV"""
import sys
import csv

from odbAccess import openOdb
from abaqusConstants import NODAL

# abaqus trims sys.argv until after "python"
if len(sys.argv) < 4:
	this_script, name, displacement = sys.argv
	poisson = 0
else:
	this_script, name, displacement, poisson = sys.argv
displacement = float(displacement)
poisson = float(poisson)
tsv = csv.writer(open(name+'-youngs.tsv', 'wb'), dialect=csv.excel_tab)
tsv.writerow(('frequency (Hz)', 'E_Real (Pa)', 'E_Imag (Pa)'))
odb = openOdb(name+'.odb', readOnly=True)
drive_nset = odb.rootAssembly.instances['PART-1-1'].nodeSets['DRIVE']
for frame in odb.steps['STEP-1'].frames:
	frequency = frame.frameValue
	if frequency == 0:
		continue
	RF = frame.fieldOutputs['RF'].getSubset(region=drive_nset, position=NODAL)
	E_Real = 0
	E_Imag = 0
	for v in RF.values:
		E_Real += v.data[0] / displacement * (1 - poisson**2)
		E_Imag += v.conjugateData[0] / displacement * (1 - poisson**2)
	tsv.writerow((frequency, E_Real, E_Imag))
