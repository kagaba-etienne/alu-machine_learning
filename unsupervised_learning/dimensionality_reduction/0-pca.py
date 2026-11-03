#!/usr/bin/env python3
"""Principal component analysis that keeps a fraction of variance."""
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a centered data set.

    Args:
        X: numpy.ndarray of shape (n, d) whose columns have mean 0.
        var: fraction of the variance the transformation should maintain.

    Returns:
        The weights matrix W of shape (d, nd).
    """
    _, s, vh = np.linalg.svd(X)
    cumulative = np.cumsum(s) / np.sum(s)
    r = np.argwhere(cumulative >= var)[0, 0]
    return vh[:r + 1].T
