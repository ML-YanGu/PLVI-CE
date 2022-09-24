import numpy as np
from sklearn.metrics import hamming_loss


def compute_hammin(y1,y2):
    hammin_value = hamming_loss(y1,y2)
    return hammin_value

def compute_deff_distance(label_target,unlabel_output):
    label_num = len(label_target)
    unlabel_num = len(unlabel_output)
    class_num = len(unlabel_output[0])
    deff_distance = np.zeros((unlabel_num,label_num))
    deff_distance = np.array(deff_distance)
    for i in range(unlabel_num):
        for j in range(label_num):
            deff_distance[i,j] = compute_hammin(unlabel_output[i],label_target[j])
    return deff_distance

# Reconcatenate deletes the DEFF_distance reorganization
def change_deff(deff_distance,Insonsis_index,unlabel_output,batch_num):
    unlabel_num = len(deff_distance)
    for i in range(batch_num):
        #Delete the Inconsis_index line in DEFF_distance. Inconsis_index is in reverse order and is deleted in sequence
        index = Insonsis_index[i]
        candidate_temp = unlabel_output[index]
        deff_distance = np.delete(deff_distance,[index],axis = 0)
        unlabel_output = np.delete(unlabel_output,[index],axis = 0)
        unlabel_num -= 1
        # Add the label deleted from unlabel_output to label_target
        # label_target = np.row_stack((label_target,candidate_temp))
        column_temp = np.zeros((unlabel_num,1))
        column_temp = np.array(column_temp)
        for j in range(unlabel_num):
            column_temp[j,0] = compute_hammin(candidate_temp,unlabel_output[j])
        #Append to deff_distance as a column append
        deff_distance = np.column_stack((deff_distance,column_temp))
    return deff_distance
