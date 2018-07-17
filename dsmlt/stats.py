from numpy import percentile


__all__ = ('trimean', )


def trimean(data):
    return (
        percentile(data, 25, axis=0) +
        2 * percentile(data, 50, axis=0) +
        percentile(data, 75, axis=0)
    ) / 4
