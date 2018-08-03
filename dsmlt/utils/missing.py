"""
Helper function for detect missing values
"""
import numpy as np
import pandas as pd

__all__ = ('missing', 'missing_count', 'single_missing', )


def single_missing(points,
                   single_missing_value: (int, float, str, None.__class__)):
    """
    Function that realize check on missing of given value.

    Parameters:
    -----------
        points : numpy array, pandas Series, pandas DataFrame
            An numobservations by numdimensions array of observations.
        missing_value : int, float, str
            A single value that we accept as indicator of missing value.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    Raises:
    -------
        AttributeError
            If passed invalid type of `points` value, e.g. all types
            except pandas Series/DataFrame or numpy ndarray.
            If passed invalid type of `missing_value` value.
    """
    if isinstance(points, (pd.DataFrame, pd.Series, np.ndarray)):
        if single_missing_value is np.NAN or single_missing_value is None:
            return pd.isnull(points)

        else:
            return points == single_missing_value

    else:
        raise AttributeError(
            'Passed value `points` with invalid type - {}.'.format(
                type(points)
            )
        )


def missing(points, missing_value: (int, float, str, list, tuple)=np.NAN):
    """
    Returns a boolean array with True if points have missing and False
    otherwise.

    Parameters:
    -----------
        points : numpy array, pandas Series, pandas DataFrame
            An numobservations by numdimensions array of observations.
        missing_value : int, float, str
                A single value that we accept as indicator of missing value.

    Returns:
    --------
        mask : numpy boolean array
            A numobservations-length boolean array.

    Raises:
    -------
        AttributeError
            If passed invalid type of `points` value, e.g. all types
            except pandas Series/DataFrame or numpy ndarray.
            If passed invalid type of `missing_value` value.
    """
    if isinstance(missing_value, (list, tuple)):
        result = single_missing(points, missing_value[0])
        for missing_value_item in missing_value[1:]:
            result_second = single_missing(points, missing_value_item)
            result |= result_second
        return result

    elif isinstance(missing_value, (int, float, str)) or \
            missing_value is np.NAN or missing_value is None:
        result = single_missing(points, missing_value)
        return result

    else:
        raise AttributeError(
            'Passed value `missing_value` with invalid type - {}.'.format(
                type(missing_value)
            )
        )


def missing_count(points,
                  missing_value: (int, float, str, list, tuple)=np.NAN):
    """
    Returns a count of missing values.

    Parameters:
    -----------
        points : numpy array, pandas Series, pandas DataFrame
            An numobservations by numdimensions array of observations.
        missing_value : int, float, str
            A single value that we accept as indicator of missing value.

    Returns:
    --------
        count : int
            A count of missing points.

    Raises:
    -------
        AttributeError
            If passed invalid type of `points` value        , e.g. all types
            except pandas Series/DataFrame or numpy ndarray.
            If passed invalid type of `missing_value` value.

    """
    if isinstance(points, (pd.DataFrame, pd.Series)):
        return missing(points, missing_value=missing_value).values.sum()

    elif isinstance(points, np.ndarray):
        return np.sum(missing(points, missing_value=missing_value))

    else:
        raise AttributeError(
            'Passed value `points` with invalid type - {}.'.format(
                type(points)
            )
        )
