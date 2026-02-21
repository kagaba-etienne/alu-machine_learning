#!/usr/bin/env python3
"""Module for Poisson distribution."""


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson distribution.

        Args:
            data: list of data to estimate the distribution
            lambtha: expected number of occurrences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculate the PMF for a given number of successes.

        Args:
            k: number of successes

        Returns:
            PMF value for k, or 0 if k is out of range
        """
        k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        return (e ** (-self.lambtha) * self.lambtha ** k) / factorial

    def cdf(self, k):
        """Calculate the CDF for a given number of successes.

        Args:
            k: number of successes

        Returns:
            CDF value for k, or 0 if k is out of range
        """
        k = int(k)
        if k < 0:
            return 0
        return sum(self.pmf(i) for i in range(k + 1))
