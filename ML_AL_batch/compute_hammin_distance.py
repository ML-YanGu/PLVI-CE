import numpy as np
import pandas as pd
from compute_sigma import change_sigma
from sklearn.metrics import hamming_loss

def compute_hammin_distance(unlabel_target,label_target):
    unlabel_num = len(unlabel_target)
    label_num = len(label_target)
    hammin_distance = np.zeros((unlabel_num,label_num))
    hammin_distance = np.array(hammin_distance)
    for i in range(unlabel_num):
        for j in range(label_num):
            hammin_distance[i,j] = hamming_loss(unlabel_target[i],label_target[j])
    return hammin_distance

def change_hammin_distance(hammin_distance,Inconsis_index,unlabel_target,batch_num):
    unlabel_num = len(unlabel_target)
    for i in range(batch_num):
        #Remove the Inconsis_index line from hammin_distance
        index = int(Inconsis_index[i])
        unlabel_temp = unlabel_target[index]
        hammin_distance = np.delete(hammin_distance,[Inconsis_index[i]],axis = 0)
        unlabel_target = np.delete(unlabel_target,[Inconsis_index[i]],axis = 0)
        unlabel_num -= 1
        column_temp = np.zeros((unlabel_num,1))
        column_temp = np.array(column_temp)
        for j in range(unlabel_num):
            column_temp[j,0] = hamming_loss(unlabel_temp,unlabel_target[j])
        #By column append
        hammin_distance = np.column_stack((hammin_distance,column_temp))
    return hammin_distance



