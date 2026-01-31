#!/usr/bin/env python3
"""
Calculate the definiteness of a square matrix.
"""


import numpy as np


def definiteness(matrix):
    """Determine the definiteness of a square matrix.

    Args:
        matrix (numpy.ndarray): A square matrix represented as a numpy array.
    Returns:
        str: 'Positive definite', 'Positive semi-definite',
             'Negative definite', 'Negative semi-definite', or 'Indefinite'.
        None: If the matrix is not square.
    """
    if (not isinstance(matrix, np.ndarray)):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    n = len(matrix)
    leading_principal_minors = []
    for i in range(1, n + 1):
        minor = matrix[:i, :i]
        leading_principal_minors.append(np.linalg.det(minor))

    pos_def = all(d > 0 for d in leading_principal_minors)
    pos_semi_def = all(d >= 0 for d in leading_principal_minors)
    neg_def = all(d < 0 if i % 2 == 0 else d > 0
                  for i, d in enumerate(leading_principal_minors))
    neg_semi_def = all(d <= 0 if i % 2 == 0 else d >= 0
                       for i, d in enumerate(leading_principal_minors))

    if pos_def:
        return 'Positive definite'
    elif pos_semi_def:
        return 'Positive semi-definite'
    elif neg_def:
        return 'Negative definite'
    elif neg_semi_def:
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
