"""
Functions to generate different plots
"""

import matplotlib.pyplot as plt 
import seaborn as sns 

"""
Basic Scatter Plot
"""
def generateScatterPlot(x,y):
    plt.scatter(x,y)
    plt.show()


"""
Basic Box Plot
"""
def generateBoxPlot(x,figSize):
    plt.figure(figsize=figSize)
    sns.boxplot(x)
    plt.show()

"""
Basic Histogram
"""
def generateHistogram(x,y):
    plt.hist(x,y)
    plt.show()

"""
Basic Correlation Matrix
"""
def generateCorrelationMatrix(data):
    corr = data.corr()
    corr.style.background_gradient()
    return corr