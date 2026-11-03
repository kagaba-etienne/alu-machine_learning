#!/usr/bin/env python3
"""Symmetric P affinities for t-SNE."""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Calculate the symmetric P affinities of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        tol: largest allowed gap between Shannon entropy and the
            entropy of perplexity.
        perplexity: perplexity of each Gaussian distribution.

    Returns:
        P: numpy.ndarray of shape (n, n) of symmetric affinities.
    """
    n, _ = X.shape
    D, P, betas, H = P_init(X, perplexity)
    for i in range(n):
        Di = np.delete(D[i], i)
        Hi, Pi = HP(Di, betas[i])
        Hdiff = Hi - H
        low = None
        high = None
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                low = betas[i].copy()
                if high is None:
                    betas[i] = betas[i] * 2.0
                else:
                    betas[i] = (betas[i] + high) / 2.0
            else:
                high = betas[i].copy()
                if low is None:
                    betas[i] = betas[i] / 2.0
                else:
                    betas[i] = (betas[i] + low) / 2.0
            Hi, Pi = HP(Di, betas[i])
            Hdiff = Hi - H
            tries += 1
        index = np.concatenate((np.r_[0:i], np.r_[i + 1:n]))
        P[i, index] = Pi
    P = (P + P.T) / (2 * n)
    return P
