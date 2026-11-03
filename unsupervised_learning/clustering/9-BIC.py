#!/usr/bin/env python3
"""Bayesian Information Criterion for a Gaussian Mixture Model."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using the BIC.

    BIC = p * ln(n) - 2 * l, where p is the number of parameters to
    learn, n is the number of data points, and l is the log likelihood.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        kmin: positive integer containing the minimum number of
            clusters to check for (inclusive).
        kmax: positive integer containing the maximum number of
            clusters to check for (inclusive). If None, the maximum
            number of clusters possible is used.
        iterations: positive integer containing the maximum number of
            iterations for the EM algorithm.
        tol: non-negative float containing the tolerance for the EM
            algorithm.
        verbose: boolean that determines if the EM algorithm should
            print information to the standard output.

    Returns:
        best_k: the best value for k based on its BIC.
        best_result: tuple containing pi, m, S for the best number of
            clusters.
        l: numpy.ndarray of shape (kmax - kmin + 1) containing the log
            likelihood for each cluster size tested.
        b: numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
            value for each cluster size tested.
        Or None, None, None, None on failure.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0:
        return None, None, None, None
    if kmin >= kmax or kmax > X.shape[0]:
        return None, None, None, None

    n, d = X.shape
    results = []
    loglikes = []
    bics = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None
        p = k * d + k * d * (d + 1) / 2 + (k - 1)
        results.append((pi, m, S))
        loglikes.append(ll)
        bics.append(p * np.log(n) - 2 * ll)

    loglikes = np.array(loglikes)
    bics = np.array(bics)
    best = int(np.argmin(bics))
    return kmin + best, results[best], loglikes, bics
