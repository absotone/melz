import numpy as np
import pandas as pd 
from RegressionHelpers import appendNumberToEveryRow,reshapeVector, getMeanSquaredError,getMeanAbsoluteError

"""
Implementation of Multivariate Linear Regression
"""
class LinearRegression:
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
    def getParametersClosedForm(self):
        # Converting the input into suitable shapes
        x = appendNumberToEveryRow(self._x,1)
        y = reshapeVector(self._y,(self._numRecords,1))


        # Step 1
        calc1 = np.dot(x.T,x)
        calc2 = np.linalg.inv(calc1)

        # Step 2
        calc3 = np.dot(calc2,x.T)

        #Step 3
        weights = np.dot(calc3,y)

        return weights


    """
    Gradient Descent Solution
    """
    def getParametersGradientDescent(self,learningRate = 0.00001,numInterations = 1000,decay = 0):
        # Converting the input into suitable shapes
        x = appendNumberToEveryRow(self._x,1)
        y = reshapeVector(self._y,(self._numRecords,1))
        weights = np.zeros((self._x.shape[1]+1,1))

        # Performing the calculation
        n = self._numRecords

        for _ in range(numInterations):

            # Step 1 
            calc1 = np.dot(x,weights)

            # Step 2 
            calc2 = (2.0/n) * np.dot(x.T,calc1-y)

            # Step 3 
            weights = weights - learningRate*calc2 

            # Step 4 
            learningRate -= decay
        
        return weights
    
    """
    Newton's Method Solution
    """
    def getParametersNewtonMethod(self,numIterations):
        pass 

    
    """
    Get Combined Accuracy
    """
    def getAccuracyTraining(self, weights):
        # Converting the input into suitable shapes
        x = appendNumberToEveryRow(self._x, 1)
        y = self._y 

        rmseError = getMeanSquaredError(x,weights,y)
        mareError = getMeanAbsoluteError(x,weights,y)

        accuracyDict = {
            "MeanSquaredError" : rmseError,
            "MeanAbsoluteError" : mareError
        }

        return accuracyDict

    """
    Get Accuracy of Test Data
    """
    def getAccuracyTesting(self,x,y,weights):
        
        # Converting the input into suitable shapes
        x = np.array(x)
        y = np.array(y)

        x = appendNumberToEveryRow(x,1)
        y = reshapeVector(y,(x.shape[0],1))

        # Performing the calculation
        rmseError = getMeanSquaredError(x,weights,y)
        mareError = getMeanAbsoluteError(x,weights,y)

        accuracyDict = {
            "MeanSquaredError" : rmseError,
            "MeanAbsoluteError" : mareError
        }

        return accuracyDict





    
