import pandas as pd 
import numpy as np 

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
        pass 

    """
    Get Accuracy
    """
    def getAccuracy(self):
        pass 