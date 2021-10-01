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