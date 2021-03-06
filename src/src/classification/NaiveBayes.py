from numpy.core.fromnumeric import mean
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

            # Generating the cut of the dataset where label = classLabel 
            calc1 = []
            for i in range(len(x)):
              if y[i] == classLabel:
                calc1.append(x[i])
            
            calc1 = np.array(calc1)

            # Calculating the mean values a priori
            self._meanValues[classId] = getMeanOverArray(calc1,0)

            # Calculating the variance values a priori 
            self._varianceValues[classId] = getVarianceOverArray(calc1,0)

            # Calculating the likelihood values a priori 
            self._likelihoodValues[classId] = calc1.shape[0]/x.shape[0]

            # Getting the Gaussian Approximation
            calc2 = getGaussianLogVector(x,self._meanValues[classId],self._varianceValues[classId])

            # Updating the Probability Values
            calc3 = calc2 + np.log(self._likelihoodValues[classId])

            probabilityValues[:, classLabel] = calc3 

        # Most Likely Outcome
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
    
    """
    Get Accuracy on Testing Data
    """
    def getAccuracyTesting(self,x,y):
        
        x = np.array(x)
        y = np.array(y)
        y = reshapeVector(y,(x.shape[0],1))
        
        n = x.shape[0]
        m = self._numFeatures
        k = self._numClasses

        probabilityValues = np.zeros((n,k))
        
        # Use the Training Calculations

        mean = self._meanValues
        variance = self._varianceValues
        likelihood = self._likelihoodValues

        for classId in range(k):
            classLabel = str(classId)

            # Get the apriori gaussian values 
            calc1 = getGaussianLogVector(x,mean[classLabel],variance[classLabel])

            # LogOdds Simplification
            calc2 = calc1 + np.log(likelihood[classLabel])

            # Step 3 
            probabilityValues[:, classId] = calc2 
        
        predictions = getMaximumIndex(probabilityValues,1)
        
        numCorrect = 0 
        for i in range(len(predictions)):
            if predictions[i] == y[i]:
                numCorrect += 1
        
        return numCorrect/len(predictions)