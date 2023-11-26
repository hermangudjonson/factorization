"""
plotting support functions for examining pca module performance
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/herman/Dropbox/Peer/general/utils")
import plotting_utils as pu

from pca_practice import PytorchPCA

from typing import Tuple

# arrow helper
def _add_arrow(
    start: Tuple[float, float], 
    end: Tuple[float, float], 
    ax=None
):
    """Wrapper to add arrow on given plot
    """
    if not ax:
        ax = plt.gca()
    ax.annotate("", xy=end, xytext=start, 
                arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=0), annotation_clip=False)

def plot_arrow(
    fitted_model: PytorchPCA, 
    data: np.ndarray, 
    labels: np.ndarray, 
    normalize = False, 
    ax = None
):
    """Plot PC directions on original data space (only relevant for 2D data).

    Parameters
    ----------
    fitted_model: `PytorchPCA`
        PCA object that has been fit on data
    data: `numpy.ndarray`
        original data points
    labels: `numpy.ndarray`
        original data labels
    normalize
        if true, rescale PC vector length
    ax
        `matplotlib.axes.Axes` for plot, if None retrieve using `plt.gca()`
    """
    components = fitted_model.components[:, [0, 1]] # (K x P) ndarray
    mu = fitted_model.offset[[0, 1]]
    X = pd.DataFrame(data[:, [0, 1]])
    y = pd.Series(labels[:])

    norm_scale = 1.0
    if normalize:
        # rescale by average component length
        norm_scale = np.linalg.norm(components, axis=1).mean()
    
    # plot data points with group label coloring
    pu.factor_plot(X, y, matplotlib_cmap='Set1', s=10, axis_off=False)
    # arrows
    _add_arrow(mu, mu + (components[:, 0] / norm_scale))
    _add_arrow(mu, mu + (components[:, 1] / norm_scale))


def plot_loss(fitted_model: PytorchPCA, ax=None):
    """Plot loss per epoch
    """
    if not ax:
        ax = plt.gca()
    ax.plot(range(len(fitted_model.history['loss'])), np.log10(fitted_model.history['loss']))
    ax.set_xlabel('epoch', fontsize=16)
    ax.set_ylabel('log10(loss)', fontsize=16)