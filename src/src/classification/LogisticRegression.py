"""
Implementation of the Logistic Regression Algorithm
"""
from ClassificationHelpers import appendNumberToEveryRow, getMatrixInverse,reshapeVector,sigmoid,getErrorMetrics
import numpy as np
import pandas as pd 


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
    Gradient Descent Solution
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
            calc3 = np.dot(x.T, (y - calc2))

            # Step 4 
            weights += (1.0/n) * learningRate * calc3 

            # Step 5 
            learningRate -= decay
        
        return weights
        

    """
    Get Solution for Logistic Regression using Newton's Method
    """
    def getNewtonMethodSolution(self, numIterations = 1000):
        # Converting the input into suitable shapes
        x = appendNumberToEveryRow(self._x,1)
        y = reshapeVector(self._y,(self._numRecords,1))
        n = self._numRecords

        # Performing the computation
        numFeatures = x.shape[1]
        weights = np.zeros((numFeatures,1))

        for _ in range(numIterations):

          calc1 = np.dot(x,weights)

          calc2 = sigmoid(calc1)

          calc3 = (1.0/n) * np.dot(x.T, (calc2 - y))

          temp1 = calc2.reshape(n,)
          temp2 = np.diag(temp1)

          temp3 = (1-calc2).reshape(n,)
          temp4 = np.diag(temp3)

          calc4 = (1.0/n) * (x.T.dot(temp2)).dot(temp4).dot(x)

          weights -= getMatrixInverse(calc4).dot(calc3)
        
        return weights


    """
    Get Training Accuracy
    """
    def getAccuracy(self,weights):
        x = appendNumberToEveryRow(self._x,1)
        y = reshapeVector(self._y,(self._numRecords,1))
        return getErrorMetrics(x,weights,y)
    
    """
    Get Testing Accuracy
    """
    def getTestingAccuracy(self,x,y,weights):
        x = np.array(x)
        y = np.array(y)
        x = appendNumberToEveryRow(x,1)
        y = reshapeVector(y,(x.shape[0],1))
        return getErrorMetrics(x,weights,y)