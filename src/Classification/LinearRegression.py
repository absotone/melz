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
        self._numFeatures = self._y.shape[0]

    """
    Closed Form Solution
    """
    def getParametersClosedForm(self):
        # Converting the input into suitable shapes
        self._x = appendNumberToEveryRow(self._x,1)
        self._y = reshapeVector(self._y,(self._numFeatures,1))

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
    def getParametersGradientDescent(self,learningRate,numInterations,decay):
        pass 
    
    """
    Newton's Method Solution
    """
    def getParametersNewtonMethod(self,numIterations):
        pass 

    
