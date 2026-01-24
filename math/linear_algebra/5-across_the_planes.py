#!/usr/bin/env python3
"""
Module that defines a function to line up two matrices.
"""


def add_matrices2D(mat1, mat2):
    """Line up two matrices by adding their corresponding elements.

    Args:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.
    Returns:
        list of list of int/float: The resulting matrix after lining up.
    """
    if len(mat1) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
        row_result = []
        for j in range(len(mat1[i])):
            row_result.append(mat1[i][j] + mat2[i][j])
        result.append(row_result)
    return result
