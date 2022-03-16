from subprocess import run
from rve_gen import Box
import pickle
import numpy as np
import csv
import os
import glob

# # looping
b = 1 # batch number
loops = 1
with open(f'Batch-{b}_E.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["box_size", "mesh_size", "E_mat", "nu_mat", "E_inc", "nu_inc", "n_inclusion", "r_inclusion", "variance", "Young's Modulus"])

# Inputs
# Material properties
box_size = 20.0
mesh_size = 1.0 # Default: 1.0

E_mat = 1e2 # MPa
nu_mat = 0.47

E_inc = 1e3 # MPa
nu_inc = 0.35

# Geometry
n = 12 # number of inclusions
r = 2.0 # average radius of inclusions
variance = 0. # r variance

for q in range(loops):


    # Generate the RVE positions and radii
    RVE = Box(box_size)
    RVE.populateSpheresSequential(n, r, variance)
    l = RVE.df.values.tolist()
    with open("RVE.pkl", "wb") as fp:   #Pickling
        pickle.dump(l, fp, protocol=2)

    # test
    # run(['abaqus', 'cae'], shell=True)

    # perform simulation
    run(f'abaqus cae noGUI=Sphere_ODB_gen.py -- {nu_inc} {E_inc} {nu_mat} {E_mat} {mesh_size} {box_size} {b} {q}'.split(), shell=True)

    # read the ODB
    run(f'abaqus cae noGUI=readODB_E.py -- {b} {q}'.split(), shell=True)

    # Extracct E
    odbname='Job-%d-%d' %(b, q)
    outname = odbname + '_E.txt'
    text = np.genfromtxt(outname, dtype=str)

    t = np.char.strip(text[0].reshape([-1,2]), "(),").astype(float)[:,0]
    U = np.char.strip(text[0].reshape([-1,2]), "(),").astype(float)[:,1]
    RF = np.char.strip(text[1].reshape([-1,2]), "(),").astype(float)[:,1]
    # print(f't: {t}')
    # print(f'U: {U}')
    # print(f'RF: {RF}')

    # Calculate E
    X = U - U.mean()
    Y = RF - RF.mean()

    slope = (X.dot(Y)) / (X.dot(X))
    E = slope / box_size
    # print(slope)
    # print(f'E: {E}')

    # Get a list of all the file paths that ends with .txt from in specified directory
    fileList = glob.glob('./Job*')
    # print(f'filelist {fileList}')
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    with open(f'Batch-{b}_E.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([box_size, mesh_size, E_mat, nu_mat, E_inc, nu_inc, n, r, variance, E])
    # with open(f'Batch-{b}_E.txt', 'a') as f:
    #     line =
    #     f.write(str(E)+'\n')
