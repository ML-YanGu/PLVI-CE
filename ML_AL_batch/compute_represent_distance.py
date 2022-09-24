import numpy as np
import math
import pandas as pd
from compute_sigma import change_sigma
def rbf(x1,x2,rbf_sigma):
    diff = x1 - x2
    # distance = np.sqrt(sum(diff ** 2))
    # rbf_distance = math.tanh(sum(diff ** 2))
    rbf_distance = np.exp(sum(diff ** 2) / (-2.0 * rbf_sigma ** 2))
    return rbf_distance

def compute_represent_distance(unlabel_data,rbf_sigma):
    #Select the most representative sample in the unlabeled set

    unlabel_num = len(unlabel_data)
    represent_distance = np.zeros((unlabel_num,unlabel_num))
    represent_distance = np.array(represent_distance)
    for i in range(unlabel_num):
        for j in range(unlabel_num):
            represent_distance[i,j] = rbf(unlabel_data[i],unlabel_data[j],rbf_sigma)
    return represent_distance

#Reconcatenation delete Recombines RBF_distance row delete and column append
def change_represent_distance(represent_distance,Inconsis_index):
    # Iteratively delete the Inconsis_index row of unlabel_hammin, and the Inconsis_index column
    Inconsis_num = len(Inconsis_index)
    for i in range(Inconsis_num):
        index = Inconsis_index[i]
        represent_distance = np.delete(represent_distance, [index], axis=0)
        represent_distance = np.delete(represent_distance, [index], axis=1)
    return represent_distance





