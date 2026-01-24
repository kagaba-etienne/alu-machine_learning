#!/usr/bin/env python3
"""
Multiply two numpy.ndarrays using matrix multiplication.
"""
import numpy as np


def np_matmul(matrix_a, matrix_b):
    """
    Multiplies two numpy.ndarrays using matrix multiplication.

    Args:
        matrix_a (numpy.ndarray): The first input array.
        matrix_b (numpy.ndarray): The second input array.
    Returns:
        numpy.ndarray: The result of the matrix multiplication.
    """
    return np.matmul(matrix_a, matrix_b)
