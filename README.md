# microstructure-ve
A repository for FEA code developed by members of the Brinson Group at Duke University. Packages specialized for the analysis of polymer nanoparticle composites (PNCs).

Active Maintainers: [Richard Sheridan](richard.sheridan@duke.edu "Contact Richard")†, [Anqi (Claire) Lin](anqi.lin@duke.edu "Contact Claire")†, [Nicholas Finan](nicholas.finan@duke.edu "Contact Nicholas")† 

## Documentation
### 3D_rve_gen
contains *Box* object containing some important functions: *populateSpheres*, *populateSpheresSequential*, and *voxelize*

Box(size)

__size__: length of one edge of the RVE boundary (float)

#### Box.populateSpheres(numSpheres, radiusIn, variance=0.)
*fills RVE with inclusions of specified size and number. Positions are randomly selected via uniform dirtribution*

__numSpheres__: number of inclusions to generate (int)

__radiusIn__: target radius of inclusions (float)

__variance__: variance when allowing inclusion radii to vary according to normal distribution (float) [default is 0.]

#### Box.populateSpheresSequential(numSpheres, radiusIn, variance=0.)
*Faster than populateSpheres, and shown to have less confirmation bias when variance > 0.*

__numSpheres__: number of inclusions to generate (int)

__radiusIn__: target radius of inclusions (float)

__variance__: variance when allowing inclusion radii to vary according to normal distribution (float) [default is 0.]

#### Box.voxelize(split=10)
*Transforms a generated Box object into a binary 3D image. Outputs Box.voxels as a \[n\]\[n\]\[n\] array*


## Attributions
†Duke University, Brinson Group
