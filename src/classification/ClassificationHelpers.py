from numpy.core.fromnumeric import var
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

"""
Get Error Metrics of the Model
"""
def getErrorMetrics(x,weights,y):
    predictedY = np.dot(x,weights)
    probabilityValues = sigmoid(predictedY)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(probabilityValues)):
        if probabilityValues[i] >= 0.5 and y[i] == 1:
            tp += 1
        elif probabilityValues[i] < 0.5 and y[i] == 0:
            tn += 1 
        elif probabilityValues[i] >= 0.5 and y[i] == 0:
            fp += 1 
        else:
            fn += 1 

    accuracyDict = {}
    accuracyDict["accuracy"] = (tp+tn)/(tp+tn+fp+fn)
    accuracyDict["tp"] = tp 
    accuracyDict["tn"] = tn 
    accuracyDict["fp"] = fp
    accuracyDict["fn"] = fn 
    return accuracyDict 

"""
Parse a Pandas DataFrame to a Numpy Array
"""
def dataFrameToNumpyArray(dataFrame):
    return dataFrame.to_numpy()

"""
Compute the Gaussian Probability
"""
def getGaussianLogVector(x,meanValue,varianceValue):

    # Step 1
    numFeatures = x.shape[1]

    # Step 2
    calc1 = np.log(varianceValue)

    # Step 3
    calc2  = -(numFeatures / 2.0) * np.log(2.0 * np.pi) - (calc1 / 2.0)

    # Step 4 
    calc3  = np.power(x - meanValue, 2) / (varianceValue)

    # Step 5
    return calc2 - 0.5 * np.sum(calc3,1)
    

"""
Get the mean over an array with column Calculation Index colIndex
"""
def getMeanOverArray(array, colIndex):
    return np.mean(array,axis = colIndex)

"""
Get the variance over an array with column Calculation Index colIndex
"""
def getVarianceOverArray(array, colIndex):
    return np.var(array, axis = colIndex)