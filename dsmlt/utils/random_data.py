from itertools import islice, permutations
from math import ceil, log
from string import ascii_uppercase

import numpy as np
import pandas as pd

from ..constants import INTEGERS, FLOATS, NUMERICS


__all__ = (
    'random_narray', 'random_size', 'columns_names_generator',
    'random_dataframe', 'random_series',
)


def random_narray(
        size: (list, tuple), dtype=int, p_missing: float=0,
        astype=None, low: (int, float)=0, high: (int, float)=1,
):
    """
    Generate random n-dimensional array with given type, size and
    percentage of corrupted data.

    Parameters
    ----------
        size : int or tuple of ints, optional
            Output shape. If the given shape is, e.g., (m, n, k), then
            m * n * k samples are drawn. Default is None, in which case
            a single value is returned.
        dtype : dtype, optional
            Desired dtype of the result.
        p_missing : float, optional
            Probability of missing data.
        low : float or int, optional
            Lower boundary of the output interval.
        high : float or int, optional
            Upper boundary of the output interval.

    Returns
    -------
        out : ndarray
            size-shaped array of random generated numbers.

    Raises
    ------
        AttributeError
            If passed invalid type of `astype` value.
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

    # corrupt data with p_missing probability
    if p_missing:
        mask = np.random.binomial(1, p_missing, size=out.shape) == 1
        if out.dtype.type in INTEGERS:
            out = out.astype(np.float)
        out[mask] = np.NAN

    return out


def random_size(n: int=None, low=0, high=100):
    """Generate random tuple of size parameter.

    Parameters
    ----------
        n : int, optional
            Length of output size tuple. If not given it generated
            randomly in intervals between 1 and 10.
        low : int, optional
            Lower boundary of the length by one dimension.
        high : int, optional
            Upper boundary of the length by one dimension.

    Returns
    -------
        size : tuple
            Randomly generated tuple.
    """
    def randomint():
        return np.random.randint(low, high)

    if n is None:
        n = np.random.randint(0, 10)

    return tuple(randomint() for _ in range(n))


def columns_names_generator(n_names):
    """Generate sequence of columns names.

    Parameters
    ----------
        n_names : int
            Number of names in sequence that should be generated.

    Returns
    -------
        name_sequence : list of strings
            Generated sequence of names.

    """
    r_length = ceil(log(n_names, len(ascii_uppercase)))
    return [
        ''.join(_)
        for _ in islice(permutations(ascii_uppercase, r_length), n_names)
    ]


def random_series(n: int = 1, dtype=int, p_missing: float=0):
    """
    Generate random pandas Series with given length, type and
    percentage of corrupted data.

    Parameters
    ----------
        n : int, optional
            Length of Series.
        dtype : dtype, optional
            Desired dtype of the result.
        p_missing : float, optional
            Probability of missing data.

    Returns
    -------
        out : pandas.Series
    """
    return pd.Series(
        random_narray(size=n, dtype=dtype, p_missing=p_missing),
        dtype=dtype
    )


def random_dataframe(rows: int=1, cols: int=1, dtype=int, p_missing: float=0):
    """
    Generate random pandas DataFrame with given size of cols and rows,
    type and percentage of corrupted data.

    Parameters
    ----------
        rows : int, optional
            Number of rows
        cols : int, optional
            Number of columns
        dtype : dtype, optional
            Desired dtype of the result.
        p_missing : float, optional
            Probability of missing data.

    Returns
    -------
        out : pandas.DataFrame
    """
    size = (rows, cols, )
    return pd.DataFrame(
        random_narray(size=size, dtype=dtype, p_missing=p_missing),
        columns=columns_names_generator(cols)
    )
