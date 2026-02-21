# Probability

This project implements probability distributions in Python without using any external libraries.

## Distributions

### Poisson (`poisson.py`)
Represents a Poisson distribution.
- `__init__(self, data=None, lambtha=1.)`: Initializes the distribution
- `pmf(self, k)`: Calculates the Probability Mass Function
- `cdf(self, k)`: Calculates the Cumulative Distribution Function

### Exponential (`exponential.py`)
Represents an Exponential distribution.
- `__init__(self, data=None, lambtha=1.)`: Initializes the distribution
- `pdf(self, x)`: Calculates the Probability Density Function
- `cdf(self, x)`: Calculates the Cumulative Distribution Function

### Normal (`normal.py`)
Represents a Normal distribution.
- `__init__(self, data=None, mean=0., stddev=1.)`: Initializes the distribution
- `z_score(self, x)`: Calculates the z-score of a given x-value
- `x_value(self, z)`: Calculates the x-value of a given z-score
- `pdf(self, x)`: Calculates the Probability Density Function
- `cdf(self, x)`: Calculates the Cumulative Distribution Function

### Binomial (`binomial.py`)
Represents a Binomial distribution.
- `__init__(self, data=None, n=1, p=0.5)`: Initializes the distribution
- `pmf(self, k)`: Calculates the Probability Mass Function
- `cdf(self, k)`: Calculates the Cumulative Distribution Function

## Mathematical Approximations Used

- π = 3.1415926536
- e = 2.7182818285
- erf(x) = (2/√π)(x - x³/3 + x⁵/10 - x⁷/42 + x⁹/216)
