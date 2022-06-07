"""Extract viscoelastic response from abaqus output ODB to TSV"""
import sys

from odbAccess import openOdb
from abaqusConstants import NODAL

# abaqus trims sys.argv until after "python"
this_script, name, displacement = sys.argv
displacement = float(displacement)
out = open(name+'-Modulii.tsv', 'w')
out.write('frequency (Hz)\tE_Real (Pa)\tE_Imag (Pa)\n')
odb = openOdb(name+'.odb')
mySteps = odb.steps['STEP-1']
ref_nSet = odb.rootAssembly.instances['PART-1-1'].nodeSets['DRIVE']
numberFrame = len(odb.steps['STEP-1'].frames)
for iFrame in range(1, numberFrame):
	frame = mySteps.frames[iFrame]
	RF_Field = frame.fieldOutputs['RF']
	RF_ref = RF_Field.getSubset(region=ref_nSet, position=NODAL)
	E_Real = 0
	E_Imag = 0
	for v in RF_ref.values:
		E_Real += v.data[0] / displacement
		E_Imag += v.conjugateData[0] / displacement
	frequency = frame.frameValue
	out.write(str(frequency) + '\t' + str(E_Real) + '\t' + str(E_Imag) + '\n')
