# -*- coding: utf-8 -*-
"""RVE gen.ipynb
Desc: RVE generator

Written by Nicholas Finan

Interactive Colaboratory file is located at
    https://colab.research.google.com/drive/1qoSgqdF7cmvU1ds8DRMKH7SQmNas9nCD
"""

import logging
import numpy as np

"""Box object allows for the generation of RVE with randomly populated Spheres."""

class Box:
    def __init__(self, size):
        self.size = float(size)
        self.min = np.array([0.,0.,0.])
        self.max = np.array([self.size, self.size, self.size])
        logging.debug(f'I am bounded from {self.min} to {self.max}')

    def voxelize(self, split=10):
        self.voxSize = self.max * split
        X = int(self.max[0] * split)
        Y = int(self.max[1] * split)
        Z = int(self.max[2] * split)
        self.voxels = np.zeros(self.voxSize.astype(int))
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    xpos = x/split
                    ypos = y/split
                    zpos = z/split
                    pixel = self.Point(np.array([xpos, ypos, zpos]))
                    for sphere in self.spheres:
                        if pixel.collide(sphere):
                            self.voxels[x][y][z] = 1

    def voxelizeV2(self, split=10):
        self.voxSize = self.max * split
        X = int(self.max[0] * split)
        Y = int(self.max[1] * split)
        Z = int(self.max[2] * split)
        x = np.arange(X)
        y = np.arange(Y)
        z = np.arange(Z)
        for sphere in self.spheres:
            taken = ((x-sphere.pos[0])**2 + (y-sphere.pos[1])**2 + (z-sphere.pos[2])**2) < sphere.rad
            print(taken)
            self.voxels[self.voxels[0]] = 1


    def populateSpheresSequential(self, numSpheres, radiusIn, variance=0.):
        self.genMethod = "sequential"
        # rng = np.random.default_rng(2022)
        rng = np.random.default_rng()

        regenerations = []

        self.legalMin = self.min
        self.legalMax = self.max
        logging.debug(f'legal Box bounds: {self.legalMin}, {self.legalMax}')


        spheres = []
        for i in range(numSpheres):
            sphere_success = None
            subregen = 0
            while sphere_success is None:
                try:
                    radius = rng.normal(radiusIn, variance)
                    pos = rng.random((1, 3)) * (self.legalMax-self.legalMin) + self.legalMin
                    thisSphere = self.Sphere(pos[0], radius)
                    # check for collistion with wall
                    if thisSphere.collide_wall(self):
                        logging.debug(f"number of successful spheres before restart: {len(spheres)}")
                        logging.debug(f"WALL COLLISION DETECTED at {i}, sphere: {thisSphere.pos.round(2)}.\nREGENERATING SPHERES...")
                        raise Exception(f"WALL COLLISION DETECTED at {i}, sphere: {thisSphere.pos.round(2)}.\nREGENERATING SPHERES...")
                    # skip sphere-sphere collision detection if this is the first entry
                    if spheres:
                        for i, secondSphere in enumerate(spheres):
                            collision = thisSphere.collide(secondSphere) #detect collision with any other spheres
                            if collision:
                                logging.debug(f"number of successful spheres before restart: {len(spheres)}")
                                rcollision = thisSphere.rad + secondSphere.rad
                                logging.debug(f"COLLISION DETECTED at {i} between {thisSphere.pos.round(2)} and {secondSphere.pos.round(2)}, d = {thisSphere.recentd:.2f} < {rcollision.round(2)}.\nREGENERATING SPHERES...")
                                raise Exception(f"COLLISION DETECTED at {i} between {thisSphere.pos.round(2)} and {secondSphere.pos.round(2)}, d = {thisSphere.recentd:.2f} < {rcollision.round(2)}. REGENERATING SPHERES")
                            # logging.debug(f'collision = {collision}')
                    spheres.append(thisSphere)
                    sphere_success = "yes"
                except:
                    pass
                subregen += 1

            regenerations.append(subregen)

        logging.info(f'legal Box bounds: {self.legalMin}, {self.legalMax}')
        logging.info(f'sphere positions: {[ x.pos for x in spheres]}')
        logging.info(f'sphere radii: {[ x.rad for x in spheres]}')
        logging.info((f"regenerations needed to compute: {sum(regenerations)}"))
        logging.info((f"regenerations raw: {regenerations}"))
        self.regen = sum(regenerations)
        self.spheres = spheres
        self.getPositions(spheres)
        self.getRadii(spheres)
        logging.debug(f'Positions: {self.Positions}\nRadii: {self.Positions}')
        self.generateDf()

    def populateSpheres(self, numSpheres, radiusIn, variance=0.):
        self.genMethod = "all or nothing"
        # rng = np.random.default_rng(2022)
        rng = np.random.default_rng()
        result = None
        regenerations = 0
        while result is None:
            try:

                # radius = [ radiusIn for i in range(0,numSpheres)]

                # if variance:
                #     radius = rng.normal(radiusIn, variance, numSpheres)

                radius = rng.normal(radiusIn, variance, numSpheres)

                # self.legalMin = self.min + radiusIn
                # self.legalMax = self.max - radiusIn
                self.legalMin = self.min
                self.legalMax = self.max
                logging.debug(f'legal Box bounds: {self.legalMin}, {self.legalMax}')

                pos = rng.random((numSpheres, 3))
                posScaled = pos * (self.legalMax-self.legalMin) + self.legalMin

                spheres = []
                for i in range(numSpheres):
                    thisSphere = self.Sphere(posScaled[i], radius[i])
                    # skip sphere-sphere collision detection if this is the first entry
                    if spheres:
                        for i, secondSphere in enumerate(spheres):
                            collision = thisSphere.collide(secondSphere)  or thisSphere.collide_wall(self) #detect collision with any other spheres or the RVE bounds
                            if collision:
                                logging.debug(f"number of successful spheres before restart: {len(spheres)}")
                                rcollision = thisSphere.rad + secondSphere.rad
                                logging.debug(f"COLLISION DETECTED at {i} between {thisSphere.pos.round(2)} and {secondSphere.pos.round(2)}, d = {thisSphere.recentd:.2f} < {rcollision.round(2)}.\nREGENERATING SPHERES...")
                                raise Exception(f"COLLISION DETECTED at {i} between {thisSphere.pos.round(2)} and {secondSphere.pos.round(2)}, d = {thisSphere.recentd:.2f} < {rcollision.round(2)}. REGENERATING SPHERES")
                            # logging.debug(f'collision = {collision}')
                    spheres.append(thisSphere)

                result = "no collision, hooray"
                # Detect Collisions
            except:
                pass
            regenerations += 1
        logging.info(f'legal Box bounds: {self.legalMin}, {self.legalMax}')
        logging.info(f'sphere positions: {[ x.pos for x in spheres]}')
        logging.info(f'sphere radii: {[ x.rad for x in spheres]}')
        logging.info((f"regenerations needed to compute: {regenerations}"))
        self.regen = regenerations
        self.spheres = spheres
        self.getPositions(spheres)
        self.getRadii(spheres)
        logging.debug(f'Positions: {self.Positions}\nRadii: {self.Positions}')
        self.generateDf()

    def generateDf(self):
        import pandas as pd
        
        self.df = pd.DataFrame(np.concatenate([self.Positions,self.Radii[:, None]], axis=1), columns=['x', 'y', 'z', 'r'])

    def getRadii(self, spheres):
        radii = []
        for sphere in spheres:
            radii.append(sphere.rad)

        self.Radii = np.array(radii)

    def getPositions(self, spheres):
        pos = []
        for sphere in spheres:
            pos.append(sphere.pos)
        self.Positions = np.array(pos)

    # nested class Sphere
    class Sphere:
        def __init__(self, pos, rad):
            self.pos = pos
            self.rad = rad

        def collide(self, secondSphere):
            d = self.dist(secondSphere)
            self.recentd = d
            return d < self.rad + secondSphere.rad

        def dist(self, secondSphere):
            vecd = self.pos - secondSphere.pos
            return np.linalg.norm(vecd)

        def collide_wall(self, box):
            lowerBound = min(self.pos) < self.rad
            upperBound = min(box.max - self.pos) < self.rad
            return lowerBound or upperBound

    # nested subclass Point
    class Point(Sphere):
        def __init__(self, pos):
            super().__init__(pos, 0)

if __name__ == "__main__":
    from sys import argv
    import pickle
    if len(argv) == 6:
        this_program, RVE_size, n, r, variance, output = argv
        RVE = Box(float(RVE_size))
        RVE.populateSpheresSequential(int(n), float(r), float(variance))
        if output == "print":
            print(RVE.df)
        elif output == "pickle":
            l = RVE.df.values.tolist()
            with open("RVE.pkl", "wb") as fp:   #Pickling
                pickle.dump(l, fp, protocol=2)
    else: #default option
        this_program, RVE_size, n, r, variance = argv
        RVE = Box(float(RVE_size))
        RVE.populateSpheresSequential(int(n), float(r), float(variance))
        l = RVE.df.values.tolist()
        with open("RVE.pkl", "wb") as fp:   #Pickling
            pickle.dump(l, fp, protocol=2)
