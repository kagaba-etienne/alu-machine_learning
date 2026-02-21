#!/usr/bin/env python3
"""Module for Exponential distribution."""


class Exponential:
    """Represents an Exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Exponential distribution.

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
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """Calculate the PDF for a given time period.

        Args:
            x: time period

        Returns:
            PDF value for x, or 0 if x is out of range
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """Calculate the CDF for a given time period.

        Args:
            x: time period

        Returns:
            CDF value for x, or 0 if x is out of range
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return 1 - (e ** (-self.lambtha * x))
