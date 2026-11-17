"""Image-processing and IO helpers.

High level functions representing important transformations or steps. SciPy is imported
lazily inside the functions that need it, so importing the package stays numpy-only.
"""
from __future__ import annotations

from typing import List

import numpy as np


def in_sorted(arr, val):
    """Determine if val is contained in arr, assuming arr is sorted.

    Parameters
    ----------
    arr : ndarray
        A 1D array sorted ascending.
    val : scalar
        The value to look for.

    Returns
    -------
    bool or numpy.bool_
        Whether ``val`` is an element of ``arr``.

    Examples
    --------
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
    """Load a microstructure variable from a MATLAB ``.mat`` file.

    Parameters
    ----------
    matfile : str or path-like
        The ``.mat`` file holding the microstructure.
    var_name : str
        The name of the variable that holds the 2D 0-1 microstructure matrix.

    Returns
    -------
    ndarray
        The variable as ``scipy.io.loadmat`` reads it, dtype as stored in the file.
    """
    from scipy.io import loadmat

    return loadmat(matfile, matlab_compatible=True)[var_name]


def assign_intph(microstructure: np.ndarray, num_layers_list: List[int]) -> np.ndarray:
    """Generate interphase layers around the particles.

    Parameters
    ----------
    microstructure : ndarray
        The microstructure array. Particles must be zero, matrix must be nonzero, and at least one
        pixel must be zero.
    num_layers_list : list of int
        The interphase thicknesses in pixels, as distances from the particles. They are sorted, so
        the order given does not matter.

    Returns
    -------
    ndarray of uint8
        0 on particles, 1 within the thinnest layer, rising by one past each thickness, so
        ``len(num_layers_list) + 1`` marks the matrix beyond every layer.
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

    Distances are measured across the periodic boundary: the image is tiled 3 times per axis,
    layered with ``assign_intph``, and the central tile is returned.

    Parameters
    ----------
    microstructure : ndarray
        As for ``assign_intph``.
    num_layers_list : list of int
        As for ``assign_intph``.

    Returns
    -------
    ndarray of uint8
        As for ``assign_intph``, shaped like ``microstructure``.
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

    Refinement multiplies pixel counts, so pixel-unit image processing is order-
    sensitive: apply ``assign_intph``/``periodic_assign_intph`` before refining to keep layer
    thicknesses in original-pixel units (refining first makes a ``num_layers=n`` interphase n/refine
    original pixels thin).

    Parameters
    ----------
    matl_img : array_like
        The material image.
    scale : float
        The node spacing of ``matl_img``.
    refine : int
        Pixels per original pixel along each axis; a positive integer.

    Returns
    -------
    refined_img : ndarray
        ``matl_img`` itself (as an array) when ``refine == 1``.
    refined_scale : float
        ``scale / refine``. Always use it together with ``refined_img``: pairing the upsampled image
        with the original scale (or vice versa) describes a physically different,
        refine-times-larger domain, and nothing downstream can detect that.

    Raises
    ------
    ValueError
        If ``refine`` is not a positive integer.

    Examples
    --------
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
    """Load a viscoelastic modulus table from a text file, sorted by frequency.

    ABAQUS needs the frequency table increasing; the storage and loss moduli are combined into one
    complex modulus.

    Parameters
    ----------
    matrl_name : str or path-like
        A whitespace-delimited file with columns frequency, storage modulus, loss modulus.

    Returns
    -------
    freq : ndarray
        The frequencies, ascending.
    youngs : ndarray of complex
        ``storage + 1j * loss`` at each of ``freq``.
    """
    freq, youngs_real, youngs_imag = np.loadtxt(matrl_name, unpack=True)
    youngs = np.empty_like(youngs_real, dtype=complex)
    youngs.real = youngs_real
    youngs.imag = youngs_imag
    sortind = np.argsort(freq)
    return freq[sortind], youngs[sortind]
