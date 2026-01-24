#!/usr/bin/env python3
"""
Module that defines a function to get the shape of a matrix.
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
