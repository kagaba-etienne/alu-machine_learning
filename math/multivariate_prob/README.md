# Multivariate Probability

This module covers multivariate statistics and probability distributions, focusing on the Multivariate Normal (Gaussian) distribution.

## Files

| File | Description |
|------|-------------|
| `0-mean_cov.py` | Function to calculate the mean and covariance matrix of a data set |
| `1-correlation.py` | Function to calculate a correlation matrix from a covariance matrix |
| `multinormal.py` | Class representing a Multivariate Normal distribution |

---

## Tasks

### 0. Mean and Covariance

**File:** `0-mean_cov.py`

```python
def mean_cov(X):
```

Calculates the mean and covariance of a data set.

- `X`: `numpy.ndarray` of shape `(n, d)` — `n` data points, `d` dimensions
- Raises `TypeError` if `X` is not a 2D `numpy.ndarray`
- Raises `ValueError` if `n < 2`
- Returns:
  - `mean`: `numpy.ndarray` of shape `(1, d)`
  - `cov`: `numpy.ndarray` of shape `(d, d)`

> `numpy.cov` is not used.

---

### 1. Correlation

**File:** `1-correlation.py`

```python
def correlation(C):
```

Calculates a correlation matrix from a covariance matrix.

- `C`: `numpy.ndarray` of shape `(d, d)` — covariance matrix
- Raises `TypeError` if `C` is not a `numpy.ndarray`
- Raises `ValueError` if `C` is not a 2D square matrix
- Returns: `numpy.ndarray` of shape `(d, d)` — correlation matrix

The correlation between variables $i$ and $j$ is:

$$\rho_{ij} = \frac{C_{ij}}{\sqrt{C_{ii} \cdot C_{jj}}}$$

---

### 2. Initialize

**File:** `multinormal.py`

```python
class MultiNormal:
    def __init__(self, data):
```

Initializes a Multivariate Normal distribution from a data set.

- `data`: `numpy.ndarray` of shape `(d, n)` — `d` dimensions, `n` data points
- Raises `TypeError` if `data` is not a 2D `numpy.ndarray`
- Raises `ValueError` if `n < 2`
- Public instance variables:
  - `mean`: `numpy.ndarray` of shape `(d, 1)`
  - `cov`: `numpy.ndarray` of shape `(d, d)`

> `numpy.cov` is not used.

---

### 3. PDF

**File:** `multinormal.py`

```python
def pdf(self, x):
```

Calculates the Probability Density Function at a data point.

- `x`: `numpy.ndarray` of shape `(d, 1)`
- Raises `TypeError` if `x` is not a `numpy.ndarray`
- Raises `ValueError` if `x` does not have shape `(d, 1)`
- Returns: the PDF value (float)

The multivariate normal PDF is:

$$f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

where $\boldsymbol{\mu}$ is the mean vector and $\Sigma$ is the covariance matrix.

---

## Requirements

- Python 3.x
- `numpy`
