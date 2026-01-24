#!/usr/bin/env python3
"""
Add two matrices element-wise.
"""


def add_matrices(mat1, mat2):
    """
    Add two matrices element-wise.

    Args:
        mat1 (list of list of ints/floats): The first matrix (n-D matrix).
        mat2 (list of list of ints/floats): The second matrix (n-D matrix).
    Returns:
        list of list of ints/floats: The resulting matrix after addition.
    """
    matrix_shape = __import__('2-size_me_please').matrix_shape
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    if not isinstance(mat1, list):
        return mat1 + mat2
    result = []
    for i in range(len(mat1)):
        result.append(add_matrices(mat1[i], mat2[i]))

    return result
