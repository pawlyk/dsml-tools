import types

from itertools import zip_longest
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import BaseEstimator


__all__ = ("plot_features_importance",)


def plot_features_importance(
    *estimators: (list, tuple, np.ndarray),
    features_names: Sequence[str] = None,
    method: str = None,
    reverse=False,
    threshold: float = None,
    ax=None,
    **kwargs,
) -> Sequence:
    """
    Build, plot and return features importance.
    Parameters:
    -----------
        estimators : list of estimators or list with importance
            coefficients.
        features_names : list of names of features in order represented
            in original data set.
        method : method of calculating total importance of estimators.
            If it is not given used numpy average.
        reverse : plot and return importance in ascent order.
        threshold : value, below which function discard importance.
        ax : matplotlib Axes, default None
            Target axes instance. If None, new figure and axes will be created.
        kwargs : Keyword arguments for plot.
    Returns:
    --------
        feature_importance : list - feature importance.
    """

    # check and get importance from estimators
    if isinstance(estimators, tuple) and all(
        isinstance(_, (tuple, list, np.ndarray, BaseEstimator))
        for _ in estimators
    ):
        importances = list()
        for estimator in estimators:
            if hasattr(estimator, "feature_importances_"):
                importances.append(estimator.feature_importances_)
            else:
                importances.append(estimator)
    else:
        raise AttributeError(
            "Invalid parameter type or it does not have "
            "`feature_importances_` parameter."
        )

    # select appropriate method, order and threshold
    if not method:
        method = np.average

    if not isinstance(method, types.FunctionType):
        raise AttributeError(
            "Invalid method `{}`. Should be valid called object.".format(
                str(method)
            )
        )

    reverse = not reverse

    # zip importance. Add features_names if given
    importances = list(zip_longest(*importances))
    if not features_names:
        features_names = range(len(importances))

    # perform feature importance operations
    importances = zip_longest(importances, features_names)
    importances = sorted(
        importances, key=lambda x: method(x[0]), reverse=reverse
    )

    # plot results
    title = kwargs.get("title")
    if not title:
        title = "Feature importance"
    xlabel = kwargs.get("xlabel")
    if not xlabel:
        xlabel = "F score"
    ylabel = kwargs.get("ylabel")
    if not ylabel:
        ylabel = "Features"
    grid = kwargs.get("grid")
    if not grid:
        grid = True
    legend = kwargs.get("legend")
    legends = [_.__class__.__name__ for _ in estimators]

    if not legend or (isinstance(legend, bool) and legend is False):
        legend = False
    elif isinstance(legend, bool) and legend:
        legend = True
    else:
        if isinstance(legend, (list, tuple)):
            if len(estimators) != len(legend):
                raise AttributeError(
                    "Invalid length of `legend` - {}, should be {}".format(
                        len(legend), len(estimators)
                    )
                )
            else:
                legends = legend
        else:
            raise AttributeError(
                "Invalid type of `legend` - {}".format(type(legend))
            )

    values, labels = zip(*importances)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    threshold = 0 if not threshold else threshold

    width = 1
    ylocs = np.arange(
        start=len(values) * (len(estimators) + width * 2) * width,
        stop=0,
        step=-(len(estimators) + width * 2) * width,
    )

    for item, label in zip(zip(*values), legends):
        ax.barh(ylocs, item, width, align="center", label=label)
        ylocs = ylocs + width

    ax.set(
        yticks=ylocs + len(estimators) * width,
        yticklabels=labels,
        ylim=[
            2 * width,
            len(values) * (len(estimators) + width * 2) * width
            + width * len(estimators),
        ],
    )

    # for x, y in zip(values, ylocs):
    #     ax.text(x + x * 0.1, y, '{:10.4f}'.format(x), va='center')

    ax.set_yticks(ylocs - (len(estimators) + width * 2) / 2)
    ax.set_yticklabels(labels)

    xlim = (
        min([item for sublist in values for item in sublist]),
        max([item for sublist in values for item in sublist]) * 1.1,
    )
    ax.set_xlim(xlim)

    ylim = (0, max(ylocs))
    ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    if legend:
        ax.legend()

    if threshold:
        ax.plot([threshold, threshold], [0, max(ylocs)], "k--")

    # return results
    if threshold:
        importances = [(v, n) for v, n in importances if method(v) > threshold]

    return [n for v, n in importances]
