# microstructure-ve
A repository for FEA code developed by members of the Brinson Group at Duke University. Packages specialized for the analysis of polymer nanoparticle composites (PNCs).

Active Maintainers: [Richard Sheridan](richard.sheridan@duke.edu "Contact Richard")†, [Nicholas Finan](nicholas.finan@duke.edu "Contact Nicholas")† 

## Documentation
### rve_gen
contains *Box* object containing some important functions: *populateSpheres*, *populateSpheresSequential*, and *voxelize*
#### Box.populateSpheres(numSpheres, radiusIn, variance=0.)
*fills RVE with inclusions of specified size and number. Positions are randomly selected via uniform dirtribution*
numSpheres: number of inclusions to generate (int)
radiusIn: target radius of inclusions (float)
variance: variance when allowing inclusion radii to vary according to normal distribution (float) [default is 0.]
#### Box.populateSpheresSequential(numSpheres, radiusIn, variance=0.)
*Faster than populateSpheres, and shown to have less confirmation bias when variance > 0.*
numSpheres: number of inclusions to generate (int)
radiusIn: target radius of inclusions (float)
variance: variance when allowing inclusion radii to vary according to normal distribution (float) [default is 0.]
#### Box.voxelize(split=10)
*Transforms a generated Box object into a binary 3D image. Outputs Box.voxels as a \[n\]\[n\]\[n\] array*


## Attributions
†Duke University, Brinson Group
