"""
Helper functions to help in data manipulation
"""

"""
Pandas implementation of train_test_split
Note: Should be split into Features and Labels post split
"""

def getTrainingTestingData(data, trainingRatio = 0.75, seed = 42):

    # Step 1
    trainingData = data.sample(frac = trainingRatio, random_state = seed)

    # Step 2 
    trainingIndex = trainingData.index 

    # Step 3 
    testingData = data.drop(trainingIndex)

    return trainingData,testingData