############################################################################
##             Creating Random Inclusions                                 ##
############################################################################

from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
# import random
from array import *

import pickle
import sys
# import math
# import numpy
# import os        # Operating system
# import shutil    # copying or moving files

def partition(sphere, q):
    x_coordinate = sphere[0]
    y_coordinate = sphere[1]
    z_coordinate = sphere[2]
    rad = sphere[3]

    # print sphere
    # print x_coordinate
    # print type(x_coordinate)

    a=rad*sin(pi/6)
    b=rad*cos(pi/6)
## Creating Datum Planes 1
    dp1=mdb.models['Model-%d' %(q)].parts['Part-1'].DatumPlaneByPrincipalPlane(offset=x_coordinate, principalPlane=YZPLANE)
    dp2=mdb.models['Model-%d' %(q)].parts['Part-1'].DatumPlaneByPrincipalPlane(offset=y_coordinate, principalPlane=XZPLANE)

## Creating Partition profile 2
    mdb.models['Model-%d' %(q)].ConstrainedSketch(gridSpacing=2.0, name='__profile__', sheetSize=25.0, transform=
        mdb.models['Model-%d' %(q)].parts['Part-1'].MakeSketchTransform(
        sketchPlane=mdb.models['Model-%d' %(q)].parts['Part-1'].datums[dp1.id], sketchPlaneSide=SIDE1,
        sketchUpEdge=mdb.models['Model-%d' %(q)].parts['Part-1'].edges.findAt((0.0, 0.0, 5.0), ), sketchOrientation=TOP, origin=(x_coordinate,y_coordinate,z_coordinate)))
    mdb.models['Model-%d' %(q)].parts['Part-1'].projectReferencesOntoSketch(filter=COPLANAR_EDGES, sketch=mdb.models['Model-%d' %(q)].sketches['__profile__'])
    c_1=mdb.models['Model-%d' %(q)].sketches['__profile__'].ArcByCenterEnds(center=(0,0), direction=CLOCKWISE, point1=(0,rad), point2=(rad,0))

    c_2=mdb.models['Model-%d' %(q)].sketches['__profile__'].ArcByCenterEnds(center=(0,0), direction=CLOCKWISE, point1=(rad,0), point2=(0,-rad))

    mdb.models['Model-%d' %(q)].sketches['__profile__'].Line(point1=(0,rad), point2=(0,-rad))
    mdb.models['Model-%d' %(q)].parts['Part-1'].PartitionCellBySketch(cells=
        mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((0.1, 0.1,0.1), )), sketch=mdb.models['Model-%d' %(q)].sketches['__profile__'],
        sketchOrientation=TOP, sketchPlane=mdb.models['Model-%d' %(q)].parts['Part-1'].datums[dp1.id], sketchUpEdge=mdb.models['Model-%d' %(q)].parts['Part-1'].edges.findAt((0.0, 0.0, 5.0), ))

## Creating circle for giving path to sweep
    mdb.models['Model-%d' %(q)].ConstrainedSketch(gridSpacing=2.0, name='__profile__', sheetSize=25.0,
        transform=mdb.models['Model-%d' %(q)].parts['Part-1'].MakeSketchTransform(sketchPlane=mdb.models['Model-%d' %(q)].parts['Part-1'].datums[dp2.id],
        sketchPlaneSide=SIDE1,sketchUpEdge=mdb.models['Model-%d' %(q)].parts['Part-1'].edges.findAt((0.0, 0.0, 5.0), ), sketchOrientation=TOP, origin=(x_coordinate, y_coordinate, z_coordinate)))
    mdb.models['Model-%d' %(q)].parts['Part-1'].projectReferencesOntoSketch(filter=COPLANAR_EDGES,sketch=mdb.models['Model-%d' %(q)].sketches['__profile__'])
    c_3=mdb.models['Model-%d' %(q)].sketches['__profile__'].CircleByCenterPerimeter(center=(0, 0), point1=(-rad, 0))

    mdb.models['Model-%d' %(q)].parts['Part-1'].PartitionCellBySketch(cells=mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((0.2,0.2,
        0.2), )), sketch=mdb.models['Model-%d' %(q)].sketches['__profile__'],sketchOrientation=TOP, sketchPlane=
        mdb.models['Model-%d' %(q)].parts['Part-1'].datums[dp2.id], sketchUpEdge=mdb.models['Model-%d' %(q)].parts['Part-1'].edges.findAt((0.0, 0.0, 5.0), ))

## Creating the spherical partition
    m= mdb.models['Model-%d' %(q)].parts['Part-1']
    m.PartitionCellBySweepEdge(sweepPath=m.edges.findAt((x_coordinate+rad, y_coordinate, z_coordinate),),cells=m.cells.findAt((0.2, 0.2, 0.2),),edges=(m.edges.findAt((x_coordinate, y_coordinate-a,z_coordinate+b), ),))
    m.PartitionCellBySweepEdge(sweepPath=m.edges.findAt((x_coordinate+rad, y_coordinate, z_coordinate),),cells=m.cells.findAt((0.2, 0.2, 0.2),),edges=(m.edges.findAt((x_coordinate, y_coordinate+a,z_coordinate+b), ),))

# MAIN

#NEW

# Input Parameters
b = int(sys.argv[-2]) # batch
q = int(sys.argv[-1]) # number of model

# b = 1 # batch
# q = 1 # number of model

box_size = float(sys.argv[-3]) # mm
mesh_size = float(sys.argv[-4]) # Default: 1.0
# box_size = 20.0 # mm
# mesh_size = 1.0 # Default: 1.0

E_mat = float(sys.argv[-5])
nu_mat = float(sys.argv[-6])
# E_mat = 1e2 # MPa
# nu_mat = 0.47

E_inc = float(sys.argv[-7])
nu_inc = float(sys.argv[-8])
# E_inc = 1e3 # MPa
# nu_inc = 0.35

xdisp_BC = 0.1 # mm

with open("RVE.pkl", "rb") as fp:   # Unpickling
    spheres = pickle.load(fp)

# LET'S CREATE MODEL
mdb.Model(modelType=STANDARD_EXPLICIT, name='Model-%d' %(q))

## LETS CREATE MATRIX
mdb.models['Model-%d' %(q)].ConstrainedSketch(name='__profile__', sheetSize=box_size)
mdb.models['Model-%d' %(q)].sketches['__profile__'].sketchOptions.setValues(
    decimalPlaces=4)
mdb.models['Model-%d' %(q)].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
    point2=(box_size, box_size))
mdb.models['Model-%d' %(q)].Part(dimensionality=THREE_D, name='Part-1', type=
    DEFORMABLE_BODY)
mdb.models['Model-%d' %(q)].parts['Part-1'].BaseSolidExtrude(depth=box_size, sketch=
    mdb.models['Model-%d' %(q)].sketches['__profile__'])
del mdb.models['Model-%d' %(q)].sketches['__profile__']

# POPULATE SPHERES
for sphere in spheres:
    partition(sphere, q)

# CREATE MATERIAL-1 (POLYMER MATRIX)
mdb.models['Model-%d' %(q)].Material(name='Matrix')
mdb.models['Model-%d' %(q)].materials['Matrix'].Elastic(table=
    ((E_mat, nu_mat), ))

# CREATE MATERIAL-2 (STIFF INCLUSION)
mdb.models['Model-%d' %(q)].Material(name='Inclusion')
mdb.models['Model-%d' %(q)].materials['Inclusion'].Elastic(table=
    ((E_inc, nu_inc), ))

# CREATE SECTIONS
mdb.models['Model-%d' %(q)].HomogeneousSolidSection(material='Matrix', name='Matrix',
    thickness=None)
mdb.models['Model-%d' %(q)].HomogeneousSolidSection(material='Inclusion', name='Inclusion',
    thickness=None)

# ASSIGN SECTIONS
mdb.models['Model-%d' %(q)].parts['Part-1'].SectionAssignment(offset=0.0,
    offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
    cells=mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((0.1, 0.1, 0.1), ), )), sectionName='Matrix',
    thicknessAssignment=FROM_SECTION)

for sphere in spheres:
    x_coordinate = sphere[0]
    y_coordinate = sphere[1]
    z_coordinate = sphere[2]
    rad = sphere[3]

    mdb.models['Model-%d' %(q)].parts['Part-1'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
        cells=mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((x_coordinate, y_coordinate-0.2*rad, z_coordinate+0.2*rad), ), )),
        sectionName='Inclusion', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-%d' %(q)].parts['Part-1'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
        cells=mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((x_coordinate, y_coordinate+0.2*rad, z_coordinate+0.2*rad), ), )),
        sectionName='Inclusion', thicknessAssignment=FROM_SECTION)

# CREATE INSTANCE
mdb.models['Model-%d' %(q)].rootAssembly.DatumCsysByDefault(CARTESIAN)
mdb.models['Model-%d' %(q)].rootAssembly.Instance(dependent=ON, name='Part-1-1',
    part=mdb.models['Model-%d' %(q)].parts['Part-1'])

# CREATE STEP
mdb.models['Model-%d' %(q)].StaticStep(initialInc=0.01, maxInc=0.1, maxNumInc=10000,
    minInc=1e-12, name='Step-1', previous='Initial')

# MAKE REFERENCE POINT
RFid = mdb.models['Model-%d' %(q)].rootAssembly.ReferencePoint(point=(box_size, box_size/2, box_size/2)).id
mdb.models['Model-%d' %(q)].rootAssembly.Set(name='RP1_Set', referencePoints=(
    mdb.models['Model-%d' %(q)].rootAssembly.referencePoints[RFid], ))
mdb.models['Model-%d' %(q)].rootAssembly.Surface(name='s_Surf-1', side1Faces=
    mdb.models['Model-%d' %(q)].rootAssembly.instances['Part-1-1'].faces.getSequenceFromMask(
    ('[#4 ]', ), ))
# couple RF (RF1_set) to x surface (s_surf-1)
mdb.models['Model-%d' %(q)].Coupling(controlPoint=
    mdb.models['Model-%d' %(q)].rootAssembly.sets['RP1_Set'], couplingType=KINEMATIC,
    influenceRadius=WHOLE_SURFACE, localCsys=None, name='Constraint-1',
    surface=mdb.models['Model-%d' %(q)].rootAssembly.surfaces['s_Surf-1'], u1=ON, u2=
    OFF, u3=OFF, ur1=OFF, ur2=OFF, ur3=OFF)

# LET'S CREATE BOUNDARY CONDITIONS
mdb.models['Model-%d' %(q)].DisplacementBC(amplitude=UNSET, createStepName='Step-1',
    distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
    'BC-1', region=Region(
    faces=mdb.models['Model-%d' %(q)].rootAssembly.instances['Part-1-1'].faces.findAt(
    ((0.0, 1.0, 5.0), ), )), u1=0.0, u2=UNSET, u3=UNSET, ur1=UNSET,
    ur2=UNSET, ur3=UNSET)

mdb.models['Model-%d' %(q)].DisplacementBC(amplitude=UNSET, createStepName='Step-1',
    distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
    'BC-2', region=Region(
    faces=mdb.models['Model-%d' %(q)].rootAssembly.instances['Part-1-1'].faces.findAt(
    ((1.0, 0.0, 5.0), ), )), u1=UNSET, u2=0.0, u3=UNSET, ur1=UNSET,
    ur2=UNSET, ur3=UNSET)

mdb.models['Model-%d' %(q)].DisplacementBC(amplitude=UNSET, createStepName='Step-1',
    distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
    'BC-3', region=Region(
    faces=mdb.models['Model-%d' %(q)].rootAssembly.instances['Part-1-1'].faces.findAt(
    ((5.0, 5.0, 0.0), ), )), u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET,
    ur2=UNSET, ur3=UNSET)

# More WIP
mdb.models['Model-%d' %(q)].DisplacementBC(amplitude=UNSET, createStepName='Step-1',
    distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
    'BC-5', region=mdb.models['Model-%d' %(q)].rootAssembly.sets['RP1_Set'], u1=xdisp_BC,
    u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET)
# End of WIP

# SEED THE PART
mdb.models['Model-%d' %(q)].parts['Part-1'].seedPart(deviationFactor=mesh_size*0.1,
    minSizeFactor=mesh_size*0.1, size=mesh_size)

# SET ELEMENT TYPE
mdb.models['Model-%d' %(q)].parts['Part-1'].setElementType(elemTypes=(ElemType(
    elemCode=C3D8, elemLibrary=STANDARD), ElemType(elemCode=C3D6,
    elemLibrary=STANDARD), ElemType(elemCode=C3D4, elemLibrary=STANDARD)),
    regions=(mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((0.1,0.1,0.1), ), ), ))

for sphere in spheres:
    x_coordinate = sphere[0]
    y_coordinate = sphere[1]
    z_coordinate = sphere[2]
    rad = sphere[3]

    mdb.models['Model-%d' %(q)].parts['Part-1'].setElementType(elemTypes=(ElemType(elemCode=C3D8, elemLibrary=STANDARD), ElemType(elemCode=C3D6,
        elemLibrary=STANDARD), ElemType(elemCode=C3D4, elemLibrary=STANDARD)),
        regions=(mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((x_coordinate,y_coordinate-0.2*rad,z_coordinate+0.2*rad), ), ((x_coordinate,y_coordinate+0.2*rad,z_coordinate+0.2*rad), ), ), ))

mdb.models['Model-%d' %(q)].parts['Part-1'].setMeshControls(elemShape=TET, regions=
    mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((0.1, 0.1, 0.1), ), ), technique=FREE)

for sphere in spheres:
    x_coordinate = sphere[0]
    y_coordinate = sphere[1]
    z_coordinate = sphere[2]
    rad = sphere[3]

    mdb.models['Model-%d' %(q)].parts['Part-1'].setMeshControls(elemShape=TET, regions=
        mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((x_coordinate,y_coordinate-0.2*rad,z_coordinate+0.2*rad), ), ((x_coordinate,y_coordinate+0.2*rad,z_coordinate+0.2*rad), ), ), technique=FREE)
    mdb.models['Model-%d' %(q)].parts['Part-1'].setMeshControls(elemShape=TET, regions=
        mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((x_coordinate,y_coordinate-1.05*rad,z_coordinate), ), ((x_coordinate,y_coordinate+1.05*rad,z_coordinate), ), ), technique=FREE)

# LET'S GENERATE MESH
mdb.models['Model-%d' %(q)].parts['Part-1'].generateMesh()


# # Reimplementing WIP
# # Make reference point
# mdb.models['Model-%d' %(q)].rootAssembly.ReferencePoint(point=(0.0, 0.0, 0.0))
# mdb.models['Model-%d' %(q)].rootAssembly.Set(name='RP1_Set', referencePoints=(
#     mdb.models['Model-%d' %(q)].rootAssembly.referencePoints[8], ))
# mdb.models['Model-%d' %(q)].rootAssembly.Surface(name='s_Surf-1', side1Faces=
#     mdb.models['Model-%d' %(q)].rootAssembly.instances['Part-1-1'].faces.getSequenceFromMask(
#     ('[#4 ]', ), ))
# # couple RF (RP1_Set) to x surface (s_surf-1)
# mdb.models['Model-%d' %(q)].Coupling(controlPoint=
#     mdb.models['Model-%d' %(q)].rootAssembly.sets['RP1_Set'], couplingType=KINEMATIC,
#     influenceRadius=WHOLE_SURFACE, localCsys=None, name='Constraint-1',
#     surface=mdb.models['Model-%d' %(q)].rootAssembly.surfaces['s_Surf-1'], u1=ON, u2=
#     ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

# Hist ouput with RF and Disp
mdb.models['Model-%d' %(q)].HistoryOutputRequest(createStepName='Step-1', name=
    'H-Output-2', rebar=EXCLUDE, region=
    mdb.models['Model-%d' %(q)].rootAssembly.sets['RP1_Set'], sectionPoints=DEFAULT,
    variables=('U1', 'RF1'))


#LET'S CREATE JOBS
mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,
    explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,
    memory=90, memoryUnits=PERCENTAGE, model='Model-%d' %(q), modelPrint=OFF,
    multiprocessingMode=DEFAULT, name='Job-%d-%d' %(b, q) , nodalOutputPrecision=SINGLE,
    numCpus=1, queue=None, scratch='', type=ANALYSIS, userSubroutine='',
    waitHours=0, waitMinutes=0)
mdb.jobs['Job-%d-%d' %(b, q)].writeInput()
mdb.jobs['Job-%d-%d' %(b, q)].submit(consistencyChecking=OFF)
mdb.jobs['Job-%d-%d' %(b, q)].waitForCompletion()


















# WIP
# # Make reference point
# mdb.models['Model-%d' %(q)].rootAssembly.ReferencePoint(point=(0.0, 0.0, 0.0))
# mdb.models['Model-%d' %(q)].rootAssembly.Set(name='RP1_Set', referencePoints=(
#     mdb.models['Model-%d' %(q)].rootAssembly.referencePoints[8], ))
# mdb.models['Model-%d' %(q)].rootAssembly.Surface(name='s_Surf-1', side1Faces=
#     mdb.models['Model-%d' %(q)].rootAssembly.instances['Part-1-1'].faces.getSequenceFromMask(
#     ('[#4 ]', ), ))
# # couple RF (RF1_set) to x surface (s_surf-1)
# mdb.models['Model-%d' %(q)].Coupling(controlPoint=
#     mdb.models['Model-%d' %(q)].rootAssembly.sets['RP1_Set'], couplingType=KINEMATIC,
#     influenceRadius=WHOLE_SURFACE, localCsys=None, name='Constraint-1',
#     surface=mdb.models['Model-%d' %(q)].rootAssembly.surfaces['s_Surf-1'], u1=ON, u2=
#     ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
# # Hist ouput with RF and Disp
# mdb.models['Model-%d' %(q)].HistoryOutputRequest(createStepName='Step-1', name=
#     'H-Output-2', rebar=EXCLUDE, region=
#     mdb.models['Model-%d' %(q)].rootAssembly.sets['RP1_Set'], sectionPoints=DEFAULT,
#     variables=('U1', 'RF1'))





# OLD
# dis=numpy.zeros(1000)
#
# box_size = 20.0
# iterations= 1    # Set number of iterations
# max_incl = 15      # set number of inclusions required
#
# for q in range (1, iterations + 1):
#     # LET'S CREATE MODEL
#     mdb.Model(modelType=STANDARD_EXPLICIT, name='Model-%d' %(q))
#
#     ## LETS CREATE MATRIX
#     mdb.models['Model-%d' %(q)].ConstrainedSketch(name='__profile__', sheetSize=box_size)
#     mdb.models['Model-%d' %(q)].sketches['__profile__'].sketchOptions.setValues(
#         decimalPlaces=4)
#     mdb.models['Model-%d' %(q)].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
#         point2=(box_size, box_size))
#     mdb.models['Model-%d' %(q)].Part(dimensionality=THREE_D, name='Part-1', type=
#         DEFORMABLE_BODY)
#     mdb.models['Model-%d' %(q)].parts['Part-1'].BaseSolidExtrude(depth=box_size, sketch=
#         mdb.models['Model-%d' %(q)].sketches['__profile__'])
#     del mdb.models['Model-%d' %(q)].sketches['__profile__']
#
#     num_incl = 0
#     x_coordinate = []
#     y_coordinate = []
#     z_coordinate = []
#
#     while (num_incl < max_incl):
#         random_x=random.uniform(3.3, 16.7)
#         random_y=random.uniform(3.3, 16.7)
#         random_z=random.uniform(3.3, 16.7)
#
#         isPointIntersecting = False
#         for j in range (0,len(x_coordinate)):
#
#
#             dis[j]=sqrt((random_x-x_coordinate[j])**2+(random_y-y_coordinate[j])**2+(random_z-z_coordinate[j])**2)
#
#
#             if dis[j] < (2.2*rad):
#
#                 isPointIntersecting = True
#                 break
#
#         if (isPointIntersecting == False):
#             x_coordinate.append(random_x)
#             y_coordinate.append(random_y)
#             z_coordinate.append(random_z)
#             num_incl = num_incl + 1
#
#     for i in range (num_incl):
#         partition(i,q)
#
#     # LET'S CREATE MATERIAL-1 (MATRIX POLYMER)
#     mdb.models['Model-%d' %(q)].Material(name='Matrix')
#     mdb.models['Model-%d' %(q)].materials['Matrix'].Elastic(table=
#         ((1e2, 0.47), ))
#
#     # LET'S CREATE MATERIAL-2 (ELASTIC INCLUSION)
#     mdb.models['Model-%d' %(q)].Material(name='Elastic')
#     mdb.models['Model-%d' %(q)].materials['Elastic'].Elastic(table=
#         ((1e3, 0.35), ))
#
#     # LET'S CREATE SECTIONS
#     mdb.models['Model-%d' %(q)].HomogeneousSolidSection(material='Matrix', name='Matrix',
#         thickness=None)
#     mdb.models['Model-%d' %(q)].HomogeneousSolidSection(material='Elastic', name='Inclusion',
#         thickness=None)
#
#     # LET'S ASSIGN SECTIONS
#     mdb.models['Model-%d' %(q)].parts['Part-1'].SectionAssignment(offset=0.0,
#         offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
#         cells=mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((0.1, 0.1, 0.1), ), )), sectionName='Matrix',
#         thicknessAssignment=FROM_SECTION)
#
#     for i in range (num_incl):
#         mdb.models['Model-%d' %(q)].parts['Part-1'].SectionAssignment(offset=0.0,
#             offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
#             cells=mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((x_coordinate[i], y_coordinate[i]-0.2*rad, z_coordinate[i]+0.2*rad), ), )),
#             sectionName='Inclusion', thicknessAssignment=FROM_SECTION)
#         mdb.models['Model-%d' %(q)].parts['Part-1'].SectionAssignment(offset=0.0,
#             offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
#             cells=mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((x_coordinate[i], y_coordinate[i]+0.2*rad, z_coordinate[i]+0.2*rad), ), )),
#             sectionName='Inclusion', thicknessAssignment=FROM_SECTION)
#
#     # LET'S CREATE INSTANCE
#     mdb.models['Model-%d' %(q)].rootAssembly.DatumCsysByDefault(CARTESIAN)
#     mdb.models['Model-%d' %(q)].rootAssembly.Instance(dependent=ON, name='Part-1-1',
#         part=mdb.models['Model-%d' %(q)].parts['Part-1'])
#
#     # LET'S CREATE STEP
#     mdb.models['Model-%d' %(q)].StaticStep(initialInc=0.01, maxInc=0.1, maxNumInc=10000,
#         minInc=1e-12, name='Step-1', previous='Initial')
#
#     # LET'S CREATE BOUNDARY CONDITIONS
#     mdb.models['Model-%d' %(q)].DisplacementBC(amplitude=UNSET, createStepName='Step-1',
#         distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
#         'BC-1', region=Region(
#         faces=mdb.models['Model-%d' %(q)].rootAssembly.instances['Part-1-1'].faces.findAt(
#         ((0.0, 1.0, 5.0), ), )), u1=0.0, u2=UNSET, u3=UNSET, ur1=UNSET,
#         ur2=UNSET, ur3=UNSET)
#
#     mdb.models['Model-%d' %(q)].DisplacementBC(amplitude=UNSET, createStepName='Step-1',
#         distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
#         'BC-2', region=Region(
#         faces=mdb.models['Model-%d' %(q)].rootAssembly.instances['Part-1-1'].faces.findAt(
#         ((1.0, 0.0, 5.0), ), )), u1=UNSET, u2=0.0, u3=UNSET, ur1=UNSET,
#         ur2=UNSET, ur3=UNSET)
#
#     mdb.models['Model-%d' %(q)].DisplacementBC(amplitude=UNSET, createStepName='Step-1',
#         distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
#         'BC-3', region=Region(
#         faces=mdb.models['Model-%d' %(q)].rootAssembly.instances['Part-1-1'].faces.findAt(
#         ((5.0, 5.0, 0.0), ), )), u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET,
#         ur2=UNSET, ur3=UNSET)
#
#     mdb.models['Model-%d' %(q)].DisplacementBC(amplitude=UNSET, createStepName='Step-1',
#         distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
#         'BC-4', region=Region(
#         faces=mdb.models['Model-%d' %(q)].rootAssembly.instances['Part-1-1'].faces.findAt(
#         ((20.0, 5.0, 5.0), ), )), u1=0.1, u2=UNSET, u3=UNSET, ur1=UNSET,
#         ur2=UNSET, ur3=UNSET)
#
#     # LET'S SEED THE PART
#     mdb.models['Model-%d' %(q)].parts['Part-1'].seedPart(deviationFactor=0.1,
#         minSizeFactor=0.1, size=1.0)
#
#     # LET'S SET ELEMENT TYPE
#     mdb.models['Model-%d' %(q)].parts['Part-1'].setElementType(elemTypes=(ElemType(
#         elemCode=C3D8, elemLibrary=STANDARD), ElemType(elemCode=C3D6,
#         elemLibrary=STANDARD), ElemType(elemCode=C3D4, elemLibrary=STANDARD)),
#         regions=(mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((0.1,0.1,0.1), ), ), ))
#
#     for i in range (num_incl):
#         mdb.models['Model-%d' %(q)].parts['Part-1'].setElementType(elemTypes=(ElemType(elemCode=C3D8, elemLibrary=STANDARD), ElemType(elemCode=C3D6,
#             elemLibrary=STANDARD), ElemType(elemCode=C3D4, elemLibrary=STANDARD)),
#             regions=(mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((x_coordinate[i],y_coordinate[i]-0.2*rad,z_coordinate[i]+0.2*rad), ), ((x_coordinate[i],y_coordinate[i]+0.2*rad,z_coordinate[i]+0.2*rad), ), ), ))
#
#     mdb.models['Model-%d' %(q)].parts['Part-1'].setMeshControls(elemShape=TET, regions=
#         mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((0.1, 0.1, 0.1), ), ), technique=FREE)
#
#     for i in range (num_incl):
#         mdb.models['Model-%d' %(q)].parts['Part-1'].setMeshControls(elemShape=TET, regions=
#     	    mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((x_coordinate[i],y_coordinate[i]-0.2*rad,z_coordinate[i]+0.2*rad), ), ((x_coordinate[i],y_coordinate[i]+0.2*rad,z_coordinate[i]+0.2*rad), ), ), technique=FREE)
#         mdb.models['Model-%d' %(q)].parts['Part-1'].setMeshControls(elemShape=TET, regions=
#     	    mdb.models['Model-%d' %(q)].parts['Part-1'].cells.findAt(((x_coordinate[i],y_coordinate[i]-1.05*rad,z_coordinate[i]), ), ((x_coordinate[i],y_coordinate[i]+1.05*rad,z_coordinate[i]), ), ), technique=FREE)
#
#     # LET'S GENERATE MESH
#     mdb.models['Model-%d' %(q)].parts['Part-1'].generateMesh()
#
#     #LET'S CREATE JOBS
#     mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,
#         explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,
#         memory=90, memoryUnits=PERCENTAGE, model='Model-%d' %(q), modelPrint=OFF,
#         multiprocessingMode=DEFAULT, name='Job-%d' %(q) , nodalOutputPrecision=SINGLE,
#         numCpus=1, queue=None, scratch='', type=ANALYSIS, userSubroutine='',
#         waitHours=0, waitMinutes=0)
#     mdb.jobs['Job-%d-%d' %(w,q)].writeInput()
#     mdb.jobs['Job-%d-%d' %(w,q) ].submit(consistencyChecking=OFF)
#     mdb.jobs['Job-%d-%d' %(w,q) ].waitForCompletion()
