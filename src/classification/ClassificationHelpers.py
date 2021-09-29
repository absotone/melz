import pandas as pd
import numpy as np

"""
Sigmoid Function
"""
def sigmoid(x):
    return 1.0/(1.0 + np.exp(x))

"""
Append a number to the start of every row in a matrix
"""
def appendNumberToEveryRow(array,number):
    return np.insert(array,0,number,axis=1)

"""
Reshaping a Matrix array to newDimensions
"""
def reshapeVector(array,newDimensions):
    return np.reshape(array,newDimensions)
