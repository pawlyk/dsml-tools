"""
Helper function for detect missing values
"""
import numpy as np
import pandas as pd

__all__ = ('missing', 'missing_count', )


def missing(points):
    """
    Returns a boolean array with True if points have missing and False
    otherwise.

    Parameters:
    -----------
        points : numpy array, pandas Series, pandas DataFrame
            An numobservations by numdimensions array of observations.

    Returns:
    --------
        mask : numpy boolean array
            A numobservations-length boolean array.
    """
    if isinstance(points, (pd.DataFrame, pd.Series)):
        return points.isnull()
    elif isinstance(points, np.ndarray):
        return np.isnan(points)
    else:
        raise AttributeError(
            'Passed value `points` with invalid type - {}.'.format(
                type(points)
            )
        )


def missing_count(points):
    """
    Returns a count of missing values.

    Parameters:
    -----------
        points : numpy array, pandas Series, pandas DataFrame
            An numobservations by numdimensions array of observations.

    Returns:
    --------
        count : int
            A count of missing points.
    """
    return sum(missing(points))
