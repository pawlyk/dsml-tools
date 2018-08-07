from numpy import percentile


__all__ = ('trimean', )


def trimean(data):
    """Compute the trimean value of the data.

    Returns the trimean value of the array elements.

    Parameters
    ----------
        data : array_like
               Input array or object that can be converted to an array.

    Returns
    -------
        trimean : A calculated trimean value.

    .. _Trimean:
        https://www.wikiwand.com/en/Trimean
    """
    p_25, p_50, p_75 = percentile(data, [25, 50, 75, ], axis=0)

    return (p_25 + 2 * p_50 + p_75) / 4
