#!/usr/bin/env python3
"""Calculates the posterior probability for various hypothetical
probabilities"""
import numpy as np
import math


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data.

    x:  number of patients that develop severe side effects
    n:  total number of patients observed
    P:  1D numpy.ndarray containing various hypothetical probabilities
    Pr: 1D numpy.ndarray containing the prior beliefs of P

    Returns: 1D numpy.ndarray containing the posterior probability of each
             probability in P given x and n, respectively
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")

    coeff = math.factorial(n) / (math.factorial(x) * math.factorial(n - x))
    lk = coeff * (P ** x) * ((1 - P) ** (n - x))
    inter = lk * Pr
    marg = np.sum(inter)
    return inter / marg
