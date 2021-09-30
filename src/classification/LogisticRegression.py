import pandas as pd 
import numpy as np 
from ClassificationHelpers import appendNumberToEveryRow,reshapeVector, sigmoid, getErrorMetrics

"""
Implementation of the Logistic Regression Algorithm
"""
class LogisticRegression:

    """
    Constructor
    """
    def __init__(self,x,y):
        self._x = np.array(x,dtype='float') 
        self._y = np.array(y,dtype='float') 
        self._numRecords = self._x.shape[0]

    """
    Closed Form Solution
    """
    def getParametersGradientDescent(self,learningRate = 0.00001,numInterations = 1000,decay = 0):
        # Converting the input into suitable shapes
        x = appendNumberToEveryRow(self._x,1)
        y = reshapeVector(self._y,(self._numRecords,1))
        n = self._numRecords
        
        # Performing the computation
        numFeatures = x.shape[1]
        weights = np.zeros((numFeatures,1))

        for _ in range(numInterations):

            # Step 1
            calc1 = np.dot(x,weights)

            # Step 2
            calc2 = sigmoid(calc1)

            # Step 3 
            calc3 = np.dot(x.T, (calc2 - y))

            # Step 4 
            weights -= (-1.0/n) * learningRate * calc3 

            # Step 5 
            learningRate -= decay
        
        return weights
        

    """
    Get Solution for Logistic Regression using Newton's Method
    """
    def getNewtonMethodSolution(self):
        pass 

    def getAccuracy(self,weights):
        x = appendNumberToEveryRow(self._x,1)
        y = reshapeVector(self._y,(self._numRecords,1))
        return getErrorMetrics(x,weights,y)