#!/usr/bin/env python3
"""Module for Normal distribution."""


class Normal:
    """Represents a Normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal distribution.

        Args:
            data: list of data to estimate the distribution
            mean: mean of the distribution
            stddev: standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculate the z-score of a given x-value.

        Args:
            x: x-value

        Returns:
            z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x-value of a given z-score.

        Args:
            z: z-score

        Returns:
            x-value of z
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """Calculate the PDF for a given x-value.

        Args:
            x: x-value

        Returns:
            PDF value for x
        """
        pi = 3.1415926536
        e = 2.7182818285
        coeff = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coeff * (e ** exponent)

    def cdf(self, x):
        """Calculate the CDF for a given x-value.

        Args:
            x: x-value

        Returns:
            CDF value for x
        """
        pi = 3.1415926536
        val = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = (2 / pi ** 0.5) * (val - val ** 3 / 3 + val ** 5 / 10
                                 - val ** 7 / 42 + val ** 9 / 216)
        return 0.5 * (1 + erf)
