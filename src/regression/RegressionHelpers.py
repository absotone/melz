import numpy as np
import pandas as pd 


"""
Get RMSE error of Regression
"""
def getMeanSquaredError(given_data,weights,test_data):
    n = given_data.shape[0]

    # Performing the calculation
    errorValue = 0

    # Step 1 
    calc1 = np.dot(given_data,weights)

    # Step 2
    for i in range(n):
        errorValue += np.power(calc1[i]-test_data[i],2)/float(n)
    
    return errorValue

"""
Parse a Pandas DataFrame to a Numpy Array
"""
def dataFrameToNumpyArray(dataFrame):
    pass 

"""
Append a fixed number to the start of every row in a matrix
"""
def appendNumberToEveryRow(array,number):
    return np.insert(array,0,number,axis=1)

"""
Reshape a given array to newDimensions.
"""
def reshapeVector(array,newDimensions):
    return np.reshape(array,newDimensions)