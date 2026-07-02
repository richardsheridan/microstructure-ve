"""microstructure-ve: viscoelastic FEA of polymer nanoparticle composites.

The package is deliberately *flat-import-free*: it re-exports nothing here. Import the
solver-neutral spec dataclasses from their submodules and the backends from
``microstructure_ve.backends``::

    from microstructure_ve.core import GridNodes, GridElements, ElementSet
    from microstructure_ve.materials import Material
    from microstructure_ve.constitutive import Elastic, TabularViscoelastic
    from microstructure_ve.boundary import PeriodicBoundaryCondition, FixedBoundaryCondition
    from microstructure_ve.steps import Model, Simulation, Step, Dynamic, Heading
    from microstructure_ve.backends import write_inp   # ABAQUS .inp emission

Keeping ``__init__`` empty also keeps the top-level import numpy-only; the DOLFINx
backend pulls in dolfinx lazily, only when its ``run`` is actually called.
"""
