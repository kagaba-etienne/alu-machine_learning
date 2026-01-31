#!/usr/bin/env python3
"""
Calculate the determinant of a square matrix.
"""


def determinant(matrix):
    """Calculate the determinant of a square matrix.

    Args:
        matrix (list): A square matrix represented as a list of lists.
    Returns:
        float: The determinant of the matrix.
    """
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if len(matrix) == 2 and len(matrix[0]) == 2:
        return (matrix[0][0] * matrix[1][1]
                - matrix[0][1] * matrix[1][0])

    det = 0
    for c in range(len(matrix)):
        # Create minor matrix
        minor = [row[:c] + row[c + 1:] for row in matrix[1:]]
        # Recursive call for determinant
        det += ((-1) ** c) * matrix[0][c] * determinant(minor)
    return det
