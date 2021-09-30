import pandas as pd 
import numpy as np 
from ClassificationHelpers import getGaussianLogVector,getMeanOverArray,getVarianceOverArray

"""
Implementation of the Gaussian Naive Bayes Classifier
"""
class NaiveBayes:
    """
    Constructor
    """
    def __init__(self,x,y):
        self._x = np.array(x) 
        self._y = np.array(y) 

        self._numRecords = self._x.shape[0]
        self._numFeatures = self._x.shape[1]
        self._numClasses = len(np.unique(y))

        self._meanValues = {}
        self._varianceValues = {}
        self._likelihoodValues = {}


    """
    Get Predicted Class Labels 
    """
    def getClassLabels(self):
        x = self._x 
        y = self._y 
        n = self._numRecords
        m = self._numFeatures
        k = self._numClasses

        probabilityValues = np.zeros((n,k))

        for classLabel in range(k):
            classId = str(classLabel)

            # Step 1 
            calc1 = x[y == classLabel]

            # Step 2
            self._meanValues[classId] = getMeanOverArray(calc1,0)

            # Step 3 
            self._varianceValues[classId] = getVarianceOverArray(calc1,0)

            # Step 4 
            self._likelihoodValues[classId] = calc1.shape[0]/x.shape[0]

            # Step 5 
            calc2 = getGaussianLogVector(x,self._meanValues[classId],self._varianceValues[classId])

            # Step 6 
            calc3 = calc2 + np.log(self._likelihoodValues[classId])

            # Step 7 
            probabilityValues[:, classLabel] = calc3 
            



    """
    Get Accuracy
    """
    def getAccuracy(self):
        pass 