#!/usr/bin/env python3
"""Expectation step of the EM algorithm for a GMM."""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculate the expectation step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        pi: numpy.ndarray of shape (k,) containing the priors for each
            cluster.
        m: numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster.
        S: numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for each cluster.

    Returns:
        g: numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster.
        l: the total log likelihood.
        Or None, None on failure.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(pi) is not np.ndarray or pi.ndim != 1:
        return None, None
    if type(m) is not np.ndarray or m.ndim != 2:
        return None, None
    if type(S) is not np.ndarray or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]
    if k < 1 or m.shape != (k, d) or S.shape != (k, d, d):
        return None, None
    if np.any(pi < 0) or not np.isclose(np.sum(pi), 1):
        return None, None

    g = np.zeros((k, n))
    for i in range(k):
        density = pdf(X, m[i], S[i])
        if density is None:
            return None, None
        g[i] = pi[i] * density

    total = np.sum(g, axis=0)
    loglike = np.sum(np.log(total))
    g /= total
    return g, loglike
