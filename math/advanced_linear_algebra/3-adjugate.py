#!/usr/bin/env python3
"""
Calculate the adjugate matrix of a square matrix.
"""


def adjugate(matrix):
    """Calculate the adjugate matrix of a given square matrix.

    Args:
        matrix (list): A square matrix represented as a list of lists.
    Returns:
        list: The adjugate matrix.
    """
    cofactor_matrix = cofactor(matrix)
    # Transpose the cofactor matrix to get the adjugate
    adjugate_matrix = [
        [cofactor_matrix[j][i] for j in range(len(cofactor_matrix))]
        for i in range(len(cofactor_matrix))]
    return adjugate_matrix


def cofactor(matrix):
    """Calculate the cofactor matrix of a given square matrix.

    Args:
        matrix (list): A square matrix represented as a list of lists.
    Returns:
        list: The cofactor matrix.
    """
    minor_matrix = minor(matrix)
    size = len(minor_matrix)
    cofactor_matrix = []
    for i in range(size):
        cofactor_row = []
        for j in range(size):
            cofactor_value = ((-1) ** (i + j)) * minor_matrix[i][j]
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)
    return cofactor_matrix


def minor(matrix):
    """Calculate the minor matrix of a given square matrix.

    Args:
        matrix (list): A square matrix represented as a list of lists.
    Returns:
        list: The minor matrix.
    """
    if (len(matrix) == 0 or not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    size = len(matrix)
    minor_matrix = []
    for i in range(size):
        minor_row = []
        for j in range(size):
            # Create the minor matrix by excluding row i and column j
            minor = [row[:j] + row[j + 1:]
                     for k, row in enumerate(matrix) if k != i]
            minor_det = determinant(minor)
            minor_row.append(minor_det)
        minor_matrix.append(minor_row)
    return minor_matrix


def determinant(matrix):
    """Calculate the determinant of a square matrix.

    Args:
        matrix (list): A square matrix represented as a list of lists.
    Returns:
        float: The determinant of the matrix.
    """
    if len(matrix) == 0:
        return 1
    if (not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

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
