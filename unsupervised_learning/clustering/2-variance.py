#!/usr/bin/env python3
"""Calculate the total intra-cluster variance of a dataset."""

import numpy as np


def variance(X, C):
    """Calculate the total intra-cluster variance for a data set.

    The total variance is the sum of squared distances from each data
    point to its nearest centroid.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        C: numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster.

    Returns:
        The total variance, or None on failure.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(C) is not np.ndarray or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1] or C.shape[0] < 1:
        return None

    sq_dist = ((X[:, None, :] - C) ** 2).sum(axis=2)
    return np.sum(np.min(sq_dist, axis=1))
