import numpy as np
import pandas as pd 
from RegressionHelpers import appendNumberToEveryRow,reshapeVector

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
        self._x = appendNumberToEveryRow(self._x,1)
        self._y = reshapeVector(self._y,(self._numRecords,1))

        # Performing the Calculation
        x = self._x
        y = self._y

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
    def getParametersGradientDescent(self,learningRate = 0.000001,numInterations = 100,decay = 0):
        # Converting the input into suitable shapes
        self._x = appendNumberToEveryRow(self._x,1)
        weights = np.zeros((self._x.shape[1],1))

        # Performing the calculation
        x = self._x 
        y = self._y 
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

    
