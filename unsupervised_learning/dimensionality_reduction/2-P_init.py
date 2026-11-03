#!/usr/bin/env python3
"""Initialize the variables used to compute t-SNE P affinities."""
import numpy as np


def P_init(X, perplexity):
    """Initialize variables for the t-SNE P affinities.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        perplexity: perplexity of each Gaussian distribution.

    Returns:
        D: squared pairwise distances of shape (n, n), with a zero
            diagonal.
        P: affinities initialized to 0, of shape (n, n).
        betas: precision values initialized to 1, of shape (n, 1).
        H: Shannon entropy of perplexity, base 2.
    """
    n, _ = X.shape
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
