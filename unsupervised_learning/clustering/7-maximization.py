#!/usr/bin/env python3
"""Maximization step of the EM algorithm for a GMM."""

import numpy as np


def maximization(X, g):
    """Calculate the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        g: numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster.

    Returns:
        pi: numpy.ndarray of shape (k,) containing the updated priors
            for each cluster.
        m: numpy.ndarray of shape (k, d) containing the updated centroid
            means for each cluster.
        S: numpy.ndarray of shape (k, d, d) containing the updated
            covariance matrices for each cluster.
        Or None, None, None on failure.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    if type(g) is not np.ndarray or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape
    if n != n_g or k < 1:
        return None, None, None
    if not np.all(np.isclose(np.sum(g, axis=0), 1)):
        return None, None, None

    nk = np.sum(g, axis=1)
    pi = nk / n
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.matmul(g, X) / nk[:, None]
    S = np.zeros((k, d, d))
    for i in range(k):
        if nk[i] <= 0 or not np.isfinite(m[i]).all():
            m[i] = np.mean(X, axis=0)
            S[i] = np.identity(d)
            continue
        diff = X - m[i]
        weighted = g[i, :, None] * diff
        S[i] = np.matmul(weighted.T, diff) / nk[i]
        if np.linalg.det(S[i]) <= 0:
            S[i] = S[i] + 1e-6 * np.identity(d)
    return pi, m, S
