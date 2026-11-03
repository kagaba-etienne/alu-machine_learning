#!/usr/bin/env python3
"""Initialize variables for a Gaussian Mixture Model."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize priors, means, and covariances for a GMM.

    Priors are initialized evenly. Means are initialized with K-means.
    Covariance matrices are initialized as identity matrices.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        k: positive integer containing the number of clusters.

    Returns:
        pi: numpy.ndarray of shape (k,) containing the priors for each
            cluster.
        m: numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster.
        S: numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for each cluster.
        Or None, None, None on failure.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    n, d = X.shape
    pi = np.ones(k) / k
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    S = np.tile(np.identity(d), (k, 1, 1))
    return pi, m, S
