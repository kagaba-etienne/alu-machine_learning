#!/usr/bin/env python3
"""
Module that provides a function to concatenate
two matrices along a specified axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two matrices along a specified axis.

    Args:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.
        axis (int): The axis to concatenate along (0 for rows, 1 for columns).
    Returns:
        list of list of int/float: The resulting concatenated matrix.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            result.append(mat1[i] + mat2[i])
        return result
    else:
        return None
