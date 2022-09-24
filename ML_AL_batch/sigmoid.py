import numpy as np
# define sigmoid function
def sigmoid(x):
    # return 1.0 / (1 + np.exp(-x))
    row = len(x)
    column = len(x[0])
    result = np.zeros((row,column))
    result = np.array(result)
    for i in range(row):
        for j in range(column):
            if x[i,j] >= 0:
                result[i,j] = 1.0 / (1 + np.exp(-x[i,j]))
            else:
                result[i,j] = np.exp(x[i,j]) / (1 + np.exp(x[i,j]))
    return result