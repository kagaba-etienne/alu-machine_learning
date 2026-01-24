#!/usr/bin/env python3
"""
Slice a numpy.ndarray along specified axes.
"""


def np_slice(matrix, axes={}):
    """
    Slices a numpy.ndarray along specified axes.

    Args:
        matrix (numpy.ndarray): The input array to be sliced.
        axes (dict): A dictionary where the key is an axis to
            slice along and the value is a tuple representing
            the slice to make along that axis

    Returns:
        numpy.ndarray: The sliced array.
    """
    slices = [slice(None)] * matrix.ndim
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)
    return matrix[tuple(slices)]
