#!/usr/bin/env python3
"""
Calculates the derivative of a polynomial
"""

def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial represented by a list of coefficients.

    Args:
        poly (list): A list of coefficients where the index is the power of x.

    Returns:
        list: A new list of coefficients representing the derivative.
        None: If poly is not a valid list.
    """
    # 1. Validation: Ensure poly is a list and is not empty
    if type(poly) is not list or len(poly) == 0:
        return None

    # 2. Base Case: If the polynomial is just a constant, its derivative is 0
    if len(poly) == 1:
        return [0]

    # 3. Calculation: Multiply each coefficient by its power (its index)
    # We start the range at 1 because the constant at index 0 drops out
    derivative = [poly[i] * i for i in range(1, len(poly))]

    # 4. Zero Edge Case: If the resulting derivative is effectively zero 
    # (e.g., poly was [5, 0, 0]), return a clean [0]
    if all(coeff == 0 for coeff in derivative):
        return [0]

    return derivative
