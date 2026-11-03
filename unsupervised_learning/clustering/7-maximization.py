#!/usr/bin/env python3
"""Maximization step in the EM algorithm of a GMM."""

import numpy as np


def maximization(X, g):
    """Calculate the maximization step in the EM algorithm of a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        g: numpy.ndarray of shape (k, n) containing the posterior
            probabilities of each data point in each cluster.

    Returns:
        pi: numpy.ndarray of shape (k,) containing the updated priors
            of each cluster.
        m: numpy.ndarray of shape (k, d) containing the updated centroid
            means of each cluster.
        S: numpy.ndarray of shape (k, d, d) containing the updated
            covariance matrices of each cluster.
        Or None, None, None on failure.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    if type(g) is not np.ndarray or g.ndim != 2:
        return None, None, None
    if g.shape[0] <= 0 or g.shape[1] != X.shape[0]:
        return None, None, None
    totals = np.sum(g, axis=0)
    if not np.all(np.isclose(totals, np.ones(g.shape[1]))):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    nk = np.sum(g, axis=1)
    pi = nk / n
    m = np.dot(g, X) / nk[:, np.newaxis]
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.dot(g[i] * diff.T, diff) / nk[i]
    return pi, m, S
