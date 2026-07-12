"""Image-processing and IO helpers.

High level functions representing important transformations or steps. SciPy is imported
lazily inside the functions that need it, so importing the package stays numpy-only.
"""
from __future__ import annotations

from typing import List

import numpy as np


def in_sorted(arr, val):
    """Determine if val is contained in arr, assuming arr is sorted.

    >>> import numpy as np
    >>> from microstructure_ve.utils import in_sorted
    >>> arr = np.array([1, 3, 5, 7])
    >>> bool(in_sorted(arr, 5))
    True
    >>> bool(in_sorted(arr, 4))
    False
    >>> bool(in_sorted(arr, 9))  # past the end
    False
    """
    index = np.searchsorted(arr, val)
    if index < len(arr):
        return val == arr[index]
    else:
        return False


def load_matlab_microstructure(matfile, var_name):
    """Load the microstructure in .mat file into a 2D boolean ndarray.
    @para: matfile --> the file name of the microstructure
           var_name --> the name of the variable in the .mat file
                        that contains the 2D microstructure 0-1 matrix.
    @return: 2D ndarray dtype=bool
    """
    from scipy.io import loadmat

    return loadmat(matfile, matlab_compatible=True)[var_name]


def assign_intph(microstructure: np.ndarray, num_layers_list: List[int]) -> np.ndarray:
    """Generate interphase layers around the particles.

    Microstructure must have at least one zero value.

    :rtype: numpy.ndarray
    :param microstructure: The microstructure array. Particles must be zero,
        matrix must be nonzero.
    :type microstructure: numpy.ndarray

    :param num_layers_list: The list of interphase thickness in pixels. The order of
        the layer values is based on the sorted distances in num_layers_list from
        the particles (near particles -> far from particles)
    :type num_layers_list: List(int)
    """
    from scipy.ndimage import distance_transform_edt

    dists = distance_transform_edt(microstructure)
    intph_img = (dists != 0).view("u1")
    for num_layers in sorted(num_layers_list):
        intph_img += dists > num_layers
    return intph_img


def periodic_assign_intph(
    microstructure: np.ndarray, num_layers_list: List[int]
) -> np.ndarray:
    """Generate interphase layers around the particles with periodic BC.

    Microstructure must have at least one zero value.

    :rtype: numpy.ndarray
    :param microstructure: The microstructure array. Particles must be zero,
        matrix must be nonzero.
    :type microstructure: numpy.ndarray

    :param num_layers_list: The list of interphase thickness in pixels. The order of
        the layer values is based on the sorted distances in num_layers_list from
        the particles (near particles -> far from particles)
    :type num_layers_list: List(int)
    """
    tiled = np.tile(microstructure, (3,) * microstructure.ndim)
    intph_tiled = assign_intph(tiled, num_layers_list)
    # trim tiling
    intph = intph_tiled[tuple(slice(dim, dim + dim) for dim in microstructure.shape)]
    # free intph's view on intph_tiled's memory
    intph = intph.copy()
    return intph


def refine_matl_img(matl_img, scale, refine):
    """Split each pixel into ``refine**ndim`` identical-material pixels, keeping the
    physical domain size by returning the correspondingly reduced scale.

    Returns ``(refined_img, scale / refine)`` -- always use both together: pairing the
    upsampled image with the original scale (or vice versa) describes a physically
    different, refine-times-larger domain, and nothing downstream can detect that.

    Refinement multiplies pixel counts, so pixel-unit image processing is order-
    sensitive: apply ``assign_intph``/``periodic_assign_intph`` BEFORE refining to keep
    layer thicknesses in original-pixel units (refining first makes a ``num_layers=n``
    interphase n/refine original pixels thin).

    >>> import numpy as np
    >>> from microstructure_ve.utils import refine_matl_img
    >>> img, scale = refine_matl_img(np.array([[0, 5], [7, 0]]), 1.0, 2)
    >>> img
    array([[0, 0, 5, 5],
           [0, 0, 5, 5],
           [7, 7, 0, 0],
           [7, 7, 0, 0]])
    >>> scale
    0.5
    """
    matl_img = np.asarray(matl_img)
    if refine != int(refine) or refine < 1:
        raise ValueError(f"refine must be a positive integer, got {refine!r}")
    if refine == 1:
        return matl_img, scale
    refine = int(refine)
    refined = np.kron(matl_img, np.ones((refine,) * matl_img.ndim, dtype=matl_img.dtype))
    return refined, scale / refine


def load_viscoelasticity(matrl_name):
    """load VE data from a text file according to ABAQUS requirements

    mainly the frequency array needs to be strictly increasing, but also having
    the storage/loss data in complex numbers helps our calculations.
    """
    freq, youngs_real, youngs_imag = np.loadtxt(matrl_name, unpack=True)
    youngs = np.empty_like(youngs_real, dtype=complex)
    youngs.real = youngs_real
    youngs.imag = youngs_imag
    sortind = np.argsort(freq)
    return freq[sortind], youngs[sortind]
