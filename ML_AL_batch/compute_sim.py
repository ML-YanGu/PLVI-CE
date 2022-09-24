import numpy as np
import pandas as pd
from compute_sigma import change_sigma
#calculate the Euclidean distance
def calculate_educlide(x1,x2):
    diff = x1 - x2
    distance = np.sqrt(sum(diff ** 2))
    return distance

def sim(x1,x2):
    distance = calculate_educlide(x1,x2)
    sim = - distance
    return sim

def compute_unlabel_sim(unlabel_data,label_data):
    unlabel_num = len(unlabel_data)
    unlabel_sim = np.zeros((unlabel_num,unlabel_num))
    unlabel_sim = np.array(unlabel_sim)
    for i in range(unlabel_num):
        for j in range(unlabel_num):
            unlabel_sim[i,j] = sim(unlabel_data[i],unlabel_data[j])

    label_num = len(label_data)
    label_sim = np.zeros((unlabel_num,label_num))
    label_sim = np.array(label_sim)
    for i in range(unlabel_num):
        for j in range(label_num):
            label_sim[i,j] = calculate_educlide(unlabel_data[i],label_data[j])
    return unlabel_sim,label_sim

def change_unlabel_sim(unlabel_sim,label_sim,Inconsis_index):
    #unlabel_sim needs to delete the Inconsis_index row and the Inconsis_index column
    Inconsis_num = len(Inconsis_index)
    #First delete unlabel_sim
    for i in range(Inconsis_num):
        index = Inconsis_index[i]
        unlabel_sim = np.delete(unlabel_sim,[index],axis = 0)
        unlabel_sim = np.delete(unlabel_sim,[index],axis = 1)

    #Then delete label_sim, just delete the row and you're done
    for i in range(Inconsis_num):
        index = Inconsis_index[i]
        label_sim = np.delete(label_sim,[index],axis = 0)
    return unlabel_sim,label_sim






