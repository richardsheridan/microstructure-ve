"""Extract viscoelastic response from abaqus output ODB to txt"""
import sys
from odbAccess import openOdb

# erase logs each time
open('log.txt', 'w').close()
def printf(text):
	with open('log.txt', 'a') as f:
		f.write('\n' + str(text))
	# print >> sys.__stdout__, text

b = int(sys.argv[-2])
q = int(sys.argv[-1])

# b = 1
# q = 1

odbname='Job-%d-%d' %(b, q)
path='./'                    # set odb path here (if in working dir no need to change!)
myodbpath = path + odbname + '.odb'
odb = openOdb(myodbpath)

RF = odb.steps['Step-1'].historyRegions['Node ASSEMBLY.1'].historyOutputs['RF1'].data
U = odb.steps['Step-1'].historyRegions['Node ASSEMBLY.1'].historyOutputs['U1'].data
printf(RF)
printf(U)
outname = odbname + '_E.txt'
with open(outname, 'w') as f:
	f.write(str(U)+'\n')
	f.write(str(RF)+'\n')
