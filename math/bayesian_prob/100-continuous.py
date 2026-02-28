#!/usr/bin/env python3
"""Calculates the posterior probability that p falls within a specific range"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability of developing
    severe side effects falls within a specific range given the data.

    Assumes a uniform prior over [0, 1]. With x successes in n trials and
    a uniform (Beta(1,1)) prior, the posterior is Beta(x+1, n-x+1).

    x:  number of patients that develop severe side effects
    n:  total number of patients observed
    p1: lower bound on the range
    p2: upper bound on the range

    Returns: posterior probability that p is within [p1, p2] given x and n
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Posterior is Beta(alpha, beta) where alpha = x+1, beta = n-x+1
    alpha = x + 1
    beta = n - x + 1

    # CDF of Beta distribution using regularized incomplete beta function btdtr
    cdf_p2 = special.btdtr(alpha, beta, p2)
    cdf_p1 = special.btdtr(alpha, beta, p1)

    return cdf_p2 - cdf_p1
