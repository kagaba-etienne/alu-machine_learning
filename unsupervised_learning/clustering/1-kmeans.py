#!/usr/bin/env python3
"""Perform K-means clustering on a dataset."""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means on a dataset.

    Cluster centroids are initialized from a multivariate uniform
    distribution over the range of X. If a cluster contains no data
    points during an update, its centroid is reinitialized.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: positive integer containing the number of clusters.
        iterations: positive integer containing the maximum number of
            iterations that should be performed.

    Returns:
        C: numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster.
        clss: numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to.
        Or None, None on failure.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low, high, (k, X.shape[1]))

    for _ in range(iterations):
        prev = C.copy()
        dist = ((X[:, None, :] - C) ** 2).sum(axis=2)
        clss = np.argmin(dist, axis=1)
        for j in range(k):
            mask = clss == j
            if np.any(mask):
                C[j] = np.mean(X[mask], axis=0)
            else:
                C[j] = np.random.uniform(low, high)
        if np.array_equal(C, prev):
            return C, clss

    dist = ((X[:, None, :] - C) ** 2).sum(axis=2)
    clss = np.argmin(dist, axis=1)
    return C, clss
