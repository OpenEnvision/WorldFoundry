import numpy as np
from scipy.ndimage import convolve
from skimage.filters.edges import HSOBEL_WEIGHTS


def sobel(image):
    # type: (numpy.ndarray) -> numpy.ndarray
    """
    Find edges using the Sobel approximation to the derivatives.

    Inspired by the [Octave implementation](https://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l196).
    """

    h1 = np.array(HSOBEL_WEIGHTS)
    h1 /= np.sum(abs(h1))  # normalize h1

    strength2 = np.square(convolve(image, h1.T))

    # Note: https://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l59
    thresh2 = 2 * np.sqrt(np.mean(strength2))

    strength2[strength2 <= thresh2] = 0
    return _simple_thinning(strength2)


def _simple_thinning(strength):
    # type: (numpy.ndarray) -> numpy.ndarray
    """
    Perform a very simple thinning.

    Inspired by the [Octave implementation](https://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l512).
    """
    num_rows, num_cols = strength.shape

    zero_column = np.zeros((num_rows, 1))
    zero_row = np.zeros((1, num_cols))

    x = (strength > np.c_[zero_column, strength[:, :-1]]) & (strength > np.c_[strength[:, 1:], zero_column])

    y = (strength > np.r_[zero_row, strength[:-1, :]]) & (strength > np.r_[strength[1:, :], zero_row])

    return x | y
