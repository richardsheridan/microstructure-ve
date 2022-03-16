from subprocess import run
from rve_gen import Box

# looping
b = 1
q = 1

# Inputs
# Material properties
box_size = 20.0
mesh_size = 1.0 # Default: 1.0

E_mat = 1e2 # MPa
nu_mat = 0.47

E_inc = 1e3 # MPa
nu_inc = 0.35

# Geometry
n = 10 # number of inclusions
r = 2.0 # average radius of inclusions
variance = 0. # r variance

# Generate the RVE positions and radii
RVE = Box(box_size)
RVE.populateSpheresSequential(n, r, variance)
l = RVE.df.values.tolist()
with open("RVE.pkl", "wb") as fp:   #Pickling
    pickle.dump(l, fp, protocol=2)

# perform simulation
run(f'abaqus python Sphere_ODB_gen.py {nu_inc} {E_inc} {nu_mat} {E_mat} {mesh_size} {box_size} {b} {q}')

# read the ODB
run(f'abaqus python readODB_E.py {b} {q}')

# Extracct E
odbname='Job-%d-%d' %(b, q)
outname = odbname + '_E.txt'
with open(outname, 'r') as f:
    U = f.read()
    RF = f.read()

print(f'U: {U}\nRF: {RF}')

# Calculate E
