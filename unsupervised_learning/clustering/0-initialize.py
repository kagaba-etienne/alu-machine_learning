#!/usr/bin/env python3
"""Initialize cluster centroids for K-means."""

import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means.

    Centroids are drawn from a multivariate uniform distribution. The
    bounds along each dimension are the minimum and maximum of X.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: positive integer containing the number of clusters.

    Returns:
        numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(k) is not int or k <= 0:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    return np.random.uniform(low, high, (k, X.shape[1]))
