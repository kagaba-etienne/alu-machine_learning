#!/usr/bin/env python3
"""Expectation-maximization algorithm for a Gaussian Mixture Model."""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Perform expectation maximization for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        k: positive integer containing the number of clusters.
        iterations: positive integer containing the maximum number of
            iterations for the algorithm.
        tol: non-negative float containing the tolerance of the log
            likelihood, used to determine early stopping.
        verbose: boolean that determines if information about the
            algorithm should be printed.

    Returns:
        pi: numpy.ndarray of shape (k,) containing the priors for each
            cluster.
        m: numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster.
        S: numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for each cluster.
        g: numpy.ndarray of shape (k, n) containing the probabilities
            for each data point in each cluster.
        l: the log likelihood of the model.
        Or None, None, None, None, None on failure.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None
    g, loglike = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None

    msg = "Log Likelihood after {} iterations: {}"
    if verbose:
        print(msg.format(0, loglike.round(5)))

    for i in range(iterations):
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None
        g, new_like = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None
        step = i + 1
        if verbose and step % 10 == 0:
            print(msg.format(step, new_like.round(5)))
        if abs(new_like - loglike) <= tol:
            if verbose and step % 10 != 0:
                print(msg.format(step, new_like.round(5)))
            return pi, m, S, g, new_like
        loglike = new_like

    if verbose and iterations % 10 != 0:
        print(msg.format(iterations, loglike.round(5)))
    return pi, m, S, g, loglike
