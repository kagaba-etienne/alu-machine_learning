#!/usr/bin/env python3
"""
Performs element-wise addition, subtraction, multiplication,
and division of two numpy.ndarrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication
    , and divisionof two numpy.ndarrays.

    Args:
        mat1 (numpy.ndarray): The first input array.
        mat2 (numpy.ndarray): The second input array.
    Returns:
        numpy.ndarray: A tuple containing the results of
        element-wise addition, subtraction, multiplication, and division.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
