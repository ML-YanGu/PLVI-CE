import numpy as np
import pandas as pd
from compute_sigma import change_sigma
def rbf(x1,x2,rbf_sigma):
    diff = x1 - x2
    rbf_distance = np.exp(sum(diff ** 2) / (-2.0 * rbf_sigma ** 2))
    return rbf_distance

# def compute_rbf_distance(label_data,unlabel_data,rbf_sigma):
#     matX = np.vstack((label_data, unlabel_data))
#     label_num = len(label_data)
#     unlabel_num = len(unlabel_data)
#     matx_num = label_num + unlabel_num
#     all_rbf = np.zeros((unlabel_num,matx_num))
#     all_rbf = np.array(all_rbf)
#     for i in range(unlabel_num):
#         for j in range(matx_num):
#             all_rbf[i,j] = rbf(unlabel_data[i],matX[j],rbf_sigma)
#     rbf_distance = all_rbf[:,0:label_num]
#     return rbf_distance,all_rbf
def compute_rbf_distance(label_data,unlabel_data,rbf_sigma):
    label_num = len(label_data)
    unlabel_num = len(unlabel_data)
    rbf_distance = np.zeros((unlabel_num,label_num))
    rbf_distance = np.array(rbf_distance)
    for i in range(unlabel_num):
        for j in range(label_num):
            rbf_distance[i,j] = rbf(unlabel_data[i],label_data[j],rbf_sigma)
    return rbf_distance

def compute_all_rbf(label_data,unlabel_data,rbf_sigma):
    matx = np.vstack((label_data, unlabel_data))
    unlabel_num = len(unlabel_data)
    matx_num = len(matx)
    all_rbf = np.zeros((unlabel_num,matx_num))
    all_rbf = np.array(all_rbf)
    for i in range(unlabel_num):
        for j in range(matx_num):
            all_rbf[i,j] = rbf(unlabel_data[i],matx[j],rbf_sigma)
    return all_rbf

#  Reconcatenation delete Recombines RBF_distance row delete and column append
def change_rbf(rbf_distance,Inconsis_index,label_data,unlabel_data,batch_num,rbf_sigma):
    unlabel_num = len(unlabel_data)

    for i in range(batch_num):
        #Delete the Inconsis_index line in rbf_distance, Inconsis_index is in reverse order, delete in order
        #Find the sample that was deleted from unlabel_data
        index = int(Inconsis_index[i])
        unlabel_temp = unlabel_data[index]
        rbf_distance = np.delete(rbf_distance,[Inconsis_index[i]],axis = 0)
        unlabel_data = np.delete(unlabel_data,[Inconsis_index[i]],axis = 0)
        unlabel_num -= 1
        #Add the samples deleted from unlabel_data to label_data, and update sigma
        label_data = np.row_stack((label_data,unlabel_temp))
        rbf_sigma = change_sigma(label_data, rbf_sigma)

        column_temp = np.zeros((unlabel_num,1))
        column_temp = np.array(column_temp)
        for j in range(unlabel_num):
            column_temp[j,0] = rbf(unlabel_temp,unlabel_data[j],rbf_sigma)
        #Column appending method. Append to rbf_distance
        rbf_distance = np.column_stack((rbf_distance,column_temp))
    return rbf_sigma,rbf_distance




