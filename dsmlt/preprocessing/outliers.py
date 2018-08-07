"""
Helper function for detection outliers values
"""
import numpy as np
import pandas as pd


__all__ = (
    'mad_outlier', 'percentile_outlier',
    'outlier', 'outlier_count',
)

# TODO move part of this functionality into utils package.


def outlier_data_sanitize(points):
    """
    Sanitize points array - if there are possibility convert data into numbers
    its converted else return boolean array that contains False for every
    element.

    Parameters
    ----------
        points : An numobservations by numdimensions array of observations.

    Returns
    -------
        mask : A numobservations-length boolean array.
    """
    if isinstance(points, pd.Series):
        points = points.convert_objects(convert_numeric=True)
    if not (points.dtype.type == np.float_ or points.dtype.type == np.int_):
        return np.full(len(points), False, dtype=bool)

    return points


def mad_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise. Based on median-absolute-deviation (MAD) test.

    Parameters
    ----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns
    -------
        mask : A numobservations-length boolean array.

    References
    ---------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    points = outlier_data_sanitize(points)
    if np.issubsctype(points, bool):
        return points

    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def percentile_outlier(points, threshold=95):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise. Based on percentile-based outlier test.

    Parameters
    ----------
        points : An numobservations by numdimensions array of observations
        threshold : An threshold - percentile value.

    Returns
    -------
        mask : A numobservations-length boolean array.
    """
    points = outlier_data_sanitize(points)
    if np.issubsctype(points, bool):
        return points

    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(points, [diff, 100 - diff])

    return (points < minval) | (points > maxval)


def outlier(points, method='mad', **kwargs):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise. This function is wrapper on other specific function.

    Parameters
    ----------
        points : An numobservations by numdimensions array of observations
        method : method that used to calculate outliers
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers
        threshold : An threshold - percentile value
        drop_na : Drop all nan from array.

    Returns
    -------
        mask : A numobservations-length boolean array.
    """
    # todo fix this
    # drop_na = kwargs.get('drop_na', True)
    # if drop_na:
    #     points = points[~np.isnan(points)]

    if method == 'mad':
        mask = mad_outlier(points, **kwargs)
    elif method == 'percentile':
        mask = percentile_outlier(points, **kwargs)
    else:
        raise NotImplementedError(
            'Passed method `%s` not implemented yet.' % method
        )

    return mask


def outlier_count(points, method='mad', **kwargs):
    """Returns a count of outliers values.

    Parameters
    ----------
        points : An numobservations by numdimensions array of observations
        method : method that used to calculate outliers
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers
        threshold : An threshold - percentile value.

    Returns
    -------
        count : A count of outliers points.
    """
    outliers = outlier(points, method, **kwargs)
    if outliers is None:
        return 0

    return sum(outliers)
