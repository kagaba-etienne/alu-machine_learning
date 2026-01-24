#!/usr/bin/env python3
"""
Concatenate two numpy.ndarray.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy.ndarray.

    Args:
        mat1 (numpy.ndarray): The first array.
        mat2 (numpy.ndarray): The second array.
        axis (int): The axis to concatenate along.
    Returns:
        numpy.ndarray: The concatenated array.
    """
    return np.concatenate((mat1, mat2), axis=axis)
