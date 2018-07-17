import numpy as np
import scipy as sp

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


__all__ = ('plot_feature_distribution', )


def plot_feature_distribution(data, num_bins=100):
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[7, 2])

    # calculate data for normal distribution plot
    ax0 = plt.subplot(gs[0])
    n, bins, patches = ax0.hist(data, num_bins, normed=1)
    mu = np.mean(data)
    sigma = np.std(data)
    y = mlab.normpdf(bins, mu, sigma)

    # calculate data for kde plot
    ax1 = plt.subplot(gs[1])
    data_min, data_max = np.min(data), np.max(data)
    data_grid = np.linspace(data_min, data_max, 1000)
    kde = sp.stats.gaussian_kde(data)
    pdf = kde.evaluate(data_grid)

    # plot data
    ax0.plot(bins, y, color='green', alpha=0.5, lw=3,
             label='Normal distribution')
    ax0.plot(data_grid, pdf, color='blue', alpha=0.5, lw=3, label='KDE')
    ax0.legend(loc='upper left')
    ax0.grid(True)
    ax1.boxplot(data, 0, 'rs', 0)
    ax1.grid(True)
    ax1.yaxis.set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
