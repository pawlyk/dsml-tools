"""
Helper function for detect missing values
"""


__all__ = ('missing', 'missing_count', )


def missing(points):
    """
    Returns a boolean array with True if points have missing and False
    otherwise.
    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations.
    Returns:
    --------
        mask : A numobservations-length boolean array.
    """
    return points.isnull()


def missing_count(points):
    """
    Returns a count of missing values.
    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations.
    Returns:
    --------
        count : A count of missing points.
    """
    return sum(missing(points))
