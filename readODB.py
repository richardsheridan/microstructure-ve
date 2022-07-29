"""Extract viscoelastic response from abaqus output ODB to TSV"""
import sys
import csv

from odbAccess import openOdb
from abaqusConstants import NODAL
# from poisson_eff import get_poisson

# abaqus trims sys.argv until after "python"
this_script, name, displacement = sys.argv
displacement = float(displacement)

tsv = csv.writer(open(name+'-youngs.tsv', 'wb'), dialect=csv.excel_tab)
tsv.writerow(('frequency (Hz)', 'E_Real (Pa)', 'E_Imag (Pa)'))
odb = openOdb(name+'.odb', readOnly=True)
etype = odb.rootAssembly.instances['PART-1-1'].elements[0].type
drive_nset = odb.rootAssembly.instances['PART-1-1'].nodeSets['DRIVE']
tl_nset = odb.rootAssembly.instances['PART-1-1'].nodeSets['TOPLEFT']
bl_nset = odb.rootAssembly.instances['PART-1-1'].nodeSets['BOTMLEFT']
br_nset = odb.rootAssembly.instances['PART-1-1'].nodeSets['BOTMRIGHT']
for frame in odb.steps['STEP-1'].frames:
	frequency = frame.frameValue
	if frequency == 0:
		continue
	RF = frame.fieldOutputs['RF'].getSubset(region=drive_nset, position=NODAL)
	# compute poisson's ratio by nu = e22/(e22-e11)
	disp_tl = frame.fieldOutputs['U'].getSubset(region=tl_nset, position=NODAL).values[0].data
	disp_bl = frame.fieldOutputs['U'].getSubset(region=bl_nset, position=NODAL).values[0].data
	disp_br = frame.fieldOutputs['U'].getSubset(region=br_nset, position=NODAL).values[0].data
	e11 = disp_br[0]-disp_bl[0]
	e22 = disp_tl[1]-disp_bl[1]
	# compute poisson's ratio on the fly
	poisson = 1.0*e22/(e22-e11)
	# no longer need displacement as input
	displacement = e11
	E_Real = 0
	E_Imag = 0
	for v in RF.values:
		if etype == 'CPE4R':
			E_Real += v.data[0] / displacement * (1 - poisson**2)
			E_Imag += v.conjugateData[0] / displacement * (1 - poisson**2)
		elif etype == 'CPS4R':
			E_Real += v.data[0] / displacement
			E_Imag += v.conjugateData[0] / displacement
	tsv.writerow((frequency, E_Real, E_Imag))
