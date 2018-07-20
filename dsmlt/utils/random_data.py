import numpy as np
# import pandas as ps

from ..constants import INTEGERS, FLOATS, NUMERICS


__all__ = (
    'random_narray', 'random_size', 'random_dataframe', 'random_series',
)

# https://stackoverflow.com/questions/30053329/elegant-way-to-create-empty-pandas-dataframe-with-nan-of-type-float
# https://pandas.pydata.org/pandas-docs/stable/missing_data.html

# todo realize this functions


def random_narray(
        size: (list, tuple), dtype=int, p_missing: float=0,
        astype=None, low: (int, float)=0, high: (int, float)=1,
):
    """
    Generate random n-dimensional array with given type, size and
    percentage of corrupted data.

    Parameters:
    -----------
        size : int or tuple of ints, optional
            Output shape. If the given shape is, e.g., (m, n, k), then
            m * n * k samples are drawn. Default is None, in which case
            a single value is returned.
        dtype : dtype, optional
            Desired dtype of the result. Currently support numpy.int and
            numpy.float
        p_missing :
        low : float or int, optional
            Lower boundary of the output interval.
        high : float or int, optional
            Upper boundary of the output interval.

    Returns:
    --------
        out : ndarray
            size-shaped array of random generated numbers.
    """
    # prepare astype parameter
    if not astype:
        astype = dtype

    if astype not in NUMERICS:
        raise AttributeError(
            'Passed invalid value of `astype` - {}.'.format(astype)
        )

    # generate random data
    if dtype in INTEGERS:
        out = np.random.randint(
            low=low, high=high, size=size, dtype=dtype).astype(astype)
    elif dtype in FLOATS:
        out = np.random.uniform(low=low, high=high, size=size).astype(astype)
    else:
        raise AttributeError(
            'Passed invalid value of `dtype` - {}.'.format(dtype)
        )

    # corrupt data with p_missing probability
    if p_missing:
        mask = np.random.binomial(1, p_missing, size=out.shape) == 1
        if out.dtype.type in INTEGERS:
            out = out.astype(np.float)
        out[mask] = np.NAN

    return out


def random_size(n: int=None, low=0, high=100):
    """
    Generate random tuple of size parameter.

    Parameters:
    ----------
        n : int, optional
            Length of output size tuple. If not given it generated
            randomly in intervals between 1 and 10.
        low : int, optional
            Lower boundary of the length by one dimension.
        high : int, optional
            Upper boundary of the length by one dimension.

    Returns:
    -------
        size : tuple
            Randomly generated tuple.

    """
    def randomint():
        return np.random.randint(low, high)

    if n is None:
        n = np.random.randint(0, 10)

    return tuple(randomint() for _ in range(n))


def random_series(n: int = 1, dtype=int, n_missing: int=0):
    pass


def random_dataframe(col: int=1, row: int=1, dtype=int, n_missing: int=0):
    pass
