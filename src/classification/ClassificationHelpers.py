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
def getGaussianValue():
    pass 