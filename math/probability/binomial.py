#!/usr/bin/env python3
"""Module for Binomial distribution."""


class Binomial:
    """Represents a Binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize Binomial distribution.

        Args:
            data: list of data to estimate the distribution
            n: number of Bernoulli trials
            p: probability of a success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p_est = 1 - (variance / mean)
            self.n = round(mean / p_est)
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculate the PMF for a given number of successes.

        Args:
            k: number of successes

        Returns:
            PMF value for k, or 0 if k is out of range
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        coeff = 1
        for i in range(self.n - k):
            coeff *= (self.n - i)
            coeff //= (i + 1)
        return coeff * (self.p ** k) * ((1 - self.p) ** (self.n - k))

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
