#!/usr/bin/env python3
"""Optimize the number of clusters for K-means by variance."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Test for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        kmin: positive integer containing the minimum number of
            clusters to check for (inclusive).
        kmax: positive integer containing the maximum number of
            clusters to check for (inclusive). None uses the number
            of data points.
        iterations: positive integer containing the maximum number of
            iterations for K-means.

    Returns:
        results: list containing the outputs of K-means for each
            cluster size.
        d_vars: list containing the difference in variance from the
            smallest cluster size for each cluster size.
        Or None, None on failure.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0 or kmax <= kmin:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    base = None
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        var = variance(X, C)
        results.append((C, clss))
        if base is None:
            base = var
        d_vars.append(base - var)
    return results, d_vars
