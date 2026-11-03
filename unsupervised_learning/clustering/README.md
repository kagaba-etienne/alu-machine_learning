# Clustering

Unsupervised clustering: K-means, Gaussian mixture models fit with
expectation-maximization, and hierarchical agglomerative clustering.

## Files

| File | Description |
|------|-------------|
| `0-initialize.py` | Initialize K-means centroids from a multivariate uniform distribution |
| `1-kmeans.py` | K-means clustering, reinitializing empty clusters |
| `2-variance.py` | Total intra-cluster variance |
| `3-optimum.py` | Variance drop across cluster sizes (elbow method) |
| `4-initialize.py` | Initialize GMM priors, means, and identity covariances |
| `5-pdf.py` | Multivariate Gaussian probability density function |
| `6-expectation.py` | EM expectation step (posterior responsibilities) |
| `7-maximization.py` | EM maximization step (priors, means, covariances) |
| `8-EM.py` | Full expectation-maximization for a GMM |
| `9-BIC.py` | Choose the number of GMM clusters with the BIC |
| `10-kmeans.py` | K-means with `sklearn.cluster` |
| `11-gmm.py` | Gaussian mixture with `sklearn.mixture` |
| `12-agglomerative.py` | Ward agglomerative clustering and a dendrogram |

## Tasks

### 0. Initialize K-means

`initialize(X, k)` draws `k` centroids with `numpy.random.uniform`. Along
each dimension, the lower bound is the minimum of `X` and the upper bound
is the maximum of `X`. Returns an array of shape `(k, d)`, or `None` on
failure.

### 1. K-means

`kmeans(X, k, iterations=1000)` runs Lloyd's algorithm. Centroids start
from the same uniform initialization as task 0. A centroid with no assigned
points is drawn again. The function returns the centroids and a label array
of shape `(n,)`, or `None, None` on failure.

### 2. Variance

`variance(X, C)` returns the total intra-cluster variance: the sum of
squared distances from each point to its nearest centroid.

### 3. Optimize k

`optimum_k(X, kmin=1, kmax=None, iterations=1000)` runs K-means for every
cluster count from `kmin` through `kmax`. It returns the K-means outputs
and the drop in variance relative to the smallest cluster count.

### 4. Initialize GMM

`initialize(X, k)` returns even priors, K-means means, and identity
covariance matrices.

### 5. PDF

`pdf(X, m, S)` evaluates the multivariate normal density. Every returned
density is at least `1e-300`.

### 6. Expectation

`expectation(X, pi, m, S)` returns posterior responsibilities of shape
`(k, n)` and the total log likelihood.

### 7. Maximization

`maximization(X, g)` updates the priors, means, and full covariance
matrices from those responsibilities.

### 8. EM

`expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False)`
fits a GMM. It stops when the log likelihood changes by at most `tol`.
With `verbose=True`, it prints the log likelihood at iteration 0, every 10
iterations, and on the final iteration.

### 9. BIC

`BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False)`
selects `k` by the Bayesian Information Criterion:

```
BIC = p * ln(n) - 2 * l
```

`p` counts means, full covariance entries, and free priors:
`k * d + k * d * (d + 1) / 2 + (k - 1)`. The smallest BIC wins. If `kmax`
is omitted, every cluster count up to the number of data points is tested.

### 10. Hello, sklearn!

`kmeans(X, k)` fits `sklearn.cluster.KMeans` and returns centroids and
labels.

### 11. GMM

`gmm(X, k)` fits `sklearn.mixture.GaussianMixture` and returns priors,
means, covariances, labels, and the BIC.

### 12. Agglomerative

`agglomerative(X, dist)` clusters with Ward linkage, draws a dendrogram
colored at cophenetic distance `dist`, and returns the cluster index of
each point.

## Requirements

- Python 3
- Tasks 0-9: `numpy` only, plus the explicitly imported previous tasks
- Task 10: `sklearn.cluster` only
- Task 11: `sklearn.mixture` only
- Task 12: `scipy.cluster.hierarchy` and `matplotlib.pyplot` only
- `pycodestyle`, module and function docstrings, executable files
