"""The ``Material`` container: a set of elements, a density, and a constitutive response.

A ``Material`` binds a constitutive ``response`` (from ``constitutive``) to an ``ElementSet``
and a density. It is a pure "has-a" container -- the physics (elastic constants,
``complex_modulus``, plasticity table) lives on ``response``; both backends read it there
(``mat.response.complex_modulus(freqs)`` for the FE backend, and the ABAQUS emitter dispatches
on ``type(mat.response)``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .constitutive import Elastic, Plastic, PronyViscoelastic, TabularViscoelastic
from .core import ElementSet


@dataclass
class Material:
    """A constitutive ``response`` applied to a set of elements at a given density.

    ``elset`` is the ElementSet it applies to, ``density`` is in kg/micron^3, and ``response``
    is one of the ``constitutive`` responses (``Elastic``, ``Plastic``, ``TabularViscoelastic``,
    ``PronyViscoelastic``) carrying the elastic constants and frequency/rate behavior::

        Material(elset, density=2.65e-15, response=Elastic(poisson=0.3, youngs=5.0))
    """

    elset: ElementSet
    density: float  # kg/micron^3
    response: Union[Elastic, Plastic, TabularViscoelastic, PronyViscoelastic]
