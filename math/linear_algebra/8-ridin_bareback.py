#!/usr/bin/env python3
"""
Defines a function to multiply two matrices.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two matrices and returns the result.

    Args:
        mat1 (list of list of floats/ints): The first matrix.
        mat2 (list of list of floats/ints): The second matrix.
    Returns:
        list of list of floats/ints: The resulting matrix after multiplication.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        row_result = []
        for j in range(len(mat2[0])):
            sum_product = 0
            for k in range(len(mat2)):
                sum_product += mat1[i][k] * mat2[k][j]
            row_result.append(sum_product)
        result.append(row_result)
    return result
