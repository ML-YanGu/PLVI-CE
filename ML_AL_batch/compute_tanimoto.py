import numpy as np
import pandas as pd
from compute_sigma import change_sigma
def tanimoto_c(x1,x2):
    xy = np.dot(x1,np.transpose(x2))
    xx = np.dot(x1,np.transpose(x1))
    yy = np.dot(x2,np.transpose(x2))
    tanimoto = xy / (xx + yy - xy)
    return tanimoto

def compute_tanimoto(label_data,unlabel_data):
    label_num = len(label_data)
    unlabel_num = len(unlabel_data)
    tanimoto = np.zeros((unlabel_num,label_num))
    tanimoto = np.array(tanimoto)
    for i in range(unlabel_num):
        for j in range(label_num):
            tanimoto[i,j] = tanimoto_c(unlabel_data[i],label_data[j])
    return tanimoto

def change_tanimoto(tanimoto,Inconsis_index,unlabel_data,batch_num):
    unlabel_num = len(unlabel_data)
    for i in range(batch_num):
        #Delete the Inconsis_index row in tanimoto, Inconsis_index is in reverse order, delete in order
        index = Inconsis_index[i]  #The index of the sample to be deleted
        unlabel_temp = unlabel_data[index]
        tanimoto = np.delete(tanimoto,[Inconsis_index[i]],axis = 0)
        unlabel_data = np.delete(unlabel_data,[Inconsis_index[i]],axis = 0)
        unlabel_num -= 1
        #Add the samples removed from unlabel_data to label_data

        column_temp = np.zeros((unlabel_num,1))
        column_temp = np.array(column_temp)
        for j in range(unlabel_num):
            column_temp[j,0] = tanimoto_c(unlabel_temp,unlabel_data[j])
        tanimoto = np.column_stack((tanimoto,column_temp))
    return tanimoto



