import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

import numpy as np


__all__ = ('plot_distribution', )


def plot_distribution(data: (list, np.array, np.ndarray), bins_number=10,
                      outliers_index: (list, tuple, np.array) = None,
                      show_outliers: bool = False,
                      outliers_ticks: bool = False,
                      show_box: bool = True, showmeans: bool = True,
                      notch: bool = False, sym: bool = False):
    """
    Plot data distribution along with its box.
    Parameters:
    -----------
        data : A data array.
        bins_number : Number of bins for histogram.
        outliers_index : Data outliers index.
        show_outliers : Show outliers on distribution plot,
        outliers_ticks : Show outliers as ticks.
        show_box : Show or not box under distribution plot
        showmeans : Show mean value on box.
        notch : Show box as notch or regular.
        sym : Show outliers points on box.
    Returns:
    --------
    """

    # prepare settings and data
    if not outliers_index:
        outliers_index = []
    num_subplots = 1
    if show_box:
        num_subplots = 2

    if sym:
        sym = 'gD'
    else:
        sym = ''

    outliers_ticks = not outliers_ticks

    mu = np.median(data)
    sigma = np.std(data)

    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, sharex=True)
    fig.subplots_adjust(hspace=0)

    gs = gridspec.GridSpec(num_subplots, 1, height_ratios=[7, 2])

    # plot distribution
    ax1 = plt.subplot(gs[0])
    n, bins, patches = ax1.hist(data, bins_number, alpha=0.5)
    y = mlab.normpdf(bins, mu, sigma)
    ax1.plot(bins, y, 'b')
    if show_outliers and len(outliers_index):
        outliers = data[outliers_index]
        ax1.plot(outliers, np.zeros_like(outliers),
                 'rd', clip_on=False, alpha=0.5)
        ax1.set_xticks(outliers, minor=outliers_ticks)
    ax1.grid(True)
    # ax1.set_xticklabels([])

    # plot boxplot
    if show_box:
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.boxplot(data, notch=notch, sym=sym,
                    showmeans=showmeans, vert=False)
        ax2.grid(True)
        ax2.yaxis.set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

    fig.suptitle('Data distribution n={}'.format(len(data)), size=14)
