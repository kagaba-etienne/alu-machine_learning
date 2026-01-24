#!/usr/bin/env python3
"""
Docstring for math.linear_algebra.102-squashed_like_sardines
"""


def matrix_shape(matrix):
    """Get the shape of a given matrix.

    Args:
        matrix: A n-D list
        representing the matrix.
    Returns:
        tuple: A list of integers representing
        the shape of the matrix.
    """
    if not isinstance(matrix, list):
        return []
    dimensions = [len(matrix)]
    for row in matrix:
        if not isinstance(row, list):
            return dimensions
        elif isinstance(row, list):
            dimensions.extend(matrix_shape(row))
            return dimensions


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a specified axis.

    Args:
        mat1 (list): The first matrix.
        mat2 (list): The second matrix.
        axis (int): The axis to concatenate along (0 for rows, 1 for columns).
    Returns:
        list: The resulting concatenated matrix.
    """
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)

    if axis > len(shape1) - 1 or axis > len(shape2) - 1:
        return None
    if (shape1[:axis] != shape2[:axis]
            or shape1[axis + 1:] != shape2[axis + 1:]):
        return None

    # Base case: if axis is 0, concatenate at the current level
    if axis == 0:
        return mat1 + mat2

    # Recursive case: concatenate each element along axis - 1
    result = []
    for i in range(len(mat1)):
        result.append(cat_matrices(mat1[i], mat2[i], axis - 1))

    return result
