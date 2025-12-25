import numpy as np 

def identity(X):
    """
    Identity function is the basic activation function
    and returns the input as output.
    
    Parameters:
    X: int, -inf to +inf
    Returns:
    X: int, -inf to +inf
    """
    return X

def binary_step(X):
    """
    Binary Step function is a threshold-based activation function
    that outputs 0 for input less than 0 and 1 for input greater than or equal to 0.    
    
    Parameters:
    X: int, -inf to +inf
    Returns:
    int: 0 or 1
    """
    if X < 0:
        return 0
    else:
        return 1
    
def linear(X, constant=1):
    """
    Linear activation function returns the input multiplied by a constant factor.
    
    Parameters:
    X: int, -inf to +inf
    constant: int, default is 1
    Returns:
    int: -inf to +inf
    """
    return constant * X

def dlinear(constant=1):
    """
    Derivative of Linear activation function
    """
    return constant

def sigmoid(X):
    res = 1 / (1 + np.exp(-X))
    return res

    