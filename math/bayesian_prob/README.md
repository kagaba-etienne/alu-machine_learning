# Bayesian Probability

This directory contains implementations of Bayesian probability concepts applied to a clinical drug trial scenario. The goal is to estimate the probability that a patient taking a drug will develop severe side effects, using observed trial data and Bayesian inference.

## Background

Given `n` patients who took a drug and `x` patients who developed severe side effects, we model `x` as following a **binomial distribution**. Bayesian inference allows us to update our beliefs about the underlying probability `p` as we observe data.

### Key Concepts

- **Likelihood** – $P(data \mid p)$: probability of observing the data given a hypothetical value of `p`
- **Prior** – $P(p)$: our initial belief about `p` before observing data
- **Intersection** – $P(data \cap p) = P(data \mid p) \cdot P(p)$
- **Marginal** – $P(data) = \sum_p P(data \cap p)$: total probability of the observed data
- **Posterior** – $P(p \mid data) = \frac{P(data \cap p)}{P(data)}$: updated belief about `p` after observing data

## Files

| File | Description |
|------|-------------|
| `0-likelihood.py` | Computes the binomial likelihood for each hypothetical probability in `P` |
| `1-intersection.py` | Computes the intersection (likelihood × prior) for each probability in `P` |
| `2-marginal.py` | Computes the marginal probability of the observed data |
| `3-posterior.py` | Computes the discrete posterior distribution over `P` |
| `100-continuous.py` | Computes the continuous posterior probability that `p` lies in a given range `[p1, p2]` using the Beta distribution |

## Usage

### 0. Likelihood

```python
import numpy as np
from likelihood import likelihood

P = np.linspace(0, 1, 11)
print(likelihood(26, 130, P))
```

### 1. Intersection

```python
import numpy as np
from intersection import intersection

P = np.linspace(0, 1, 11)
Pr = np.ones(11) / 11
print(intersection(26, 130, P, Pr))
```

### 2. Marginal Probability

```python
import numpy as np
from marginal import marginal

P = np.linspace(0, 1, 11)
Pr = np.ones(11) / 11
print(marginal(26, 130, P, Pr))
```

### 3. Posterior

```python
import numpy as np
from posterior import posterior

P = np.linspace(0, 1, 11)
Pr = np.ones(11) / 11
print(posterior(26, 130, P, Pr))
```

### 4. Continuous Posterior (Advanced)

```python
from continuous import posterior

print(posterior(26, 130, 0.17, 0.23))
# Output: 0.6098093274896035
```

With a uniform prior, the posterior distribution of `p` is a Beta distribution:

$$p \mid x, n \sim \text{Beta}(x + 1,\ n - x + 1)$$

The probability that `p` falls in `[p1, p2]` is then:

$$P(p_1 \leq p \leq p_2 \mid x, n) = F_{\text{Beta}}(p_2;\ x+1,\ n-x+1) - F_{\text{Beta}}(p_1;\ x+1,\ n-x+1)$$

## Dependencies

- `numpy`
- `scipy` (for `100-continuous.py` only)
