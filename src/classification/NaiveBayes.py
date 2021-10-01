import pandas as pd 
import numpy as np 
from ClassificationHelpers import getGaussianLogVector,getMeanOverArray,getVarianceOverArray,reshapeVector, getMaximumIndex

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
        self._y = reshapeVector(self._y,(self._x.shape[0],1))  

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
            calc1 = []
            for i in range(len(x)):
              if y[i] == classLabel:
                calc1.append(x[i])
            
            calc1 = np.array(calc1)
            print(calc1.shape)

            # Step 2
            self._meanValues[classId] = getMeanOverArray(calc1,0)

            # Step 3 
            self._varianceValues[classId] = getVarianceOverArray(calc1,0)

            # Step 4 
            self._likelihoodValues[classId] = calc1.shape[0]/x.shape[0]

            # Step 5 
            calc2 = getGaussianLogVector(x,self._meanValues[classId],self._varianceValues[classId])
            print(calc2.shape)

            # Step 6 
            calc3 = calc2 + np.log(self._likelihoodValues[classId])

            # Step 7 
            probabilityValues[:, classLabel] = calc3 

        return getMaximumIndex(probabilityValues,1)
            



    """
    Get Accuracy
    """
    def getAccuracy(self,labelValues):
        
        totalCorrect = 0
        y = self._y
        
        for i in range(len(y)):
            if labelValues[i] == y[i]:
                totalCorrect += 1 
        
        return totalCorrect/len(labelValues)