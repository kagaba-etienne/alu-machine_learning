#!/usr/bin/env python3

def matrix_transpose(matrix):
    """Transpose a given matrix.

    Args:
        matrix (list of list of int/float): A 2D list
        representing the matrix to be transposed.
    Returns:
        list of list of int/float: The transposed matrix.
    """
    transposed = []
    for i in range(len(matrix[0])):
        new_row = []
        for row in matrix:
            new_row.append(row[i])
        transposed.append(new_row)
    return transposed
