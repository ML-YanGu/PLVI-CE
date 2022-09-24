import numpy as np
from sklearn.metrics import hamming_loss
def rbf(x1,x2,rbf_sigma):
    diff = x1 - x2
    rbf_distance = np.exp(sum(diff ** 2) / (-2.0 * rbf_sigma ** 2))
    return rbf_distance

def compute_unlabel_hammin(unlabel_output,sigmoid):
    unlabel_num = len(unlabel_output)
    unlabel_hammin = np.zeros((unlabel_num,unlabel_num))
    unlabel_hammin = np.array(unlabel_hammin)
    for i in range(unlabel_num):
        for j in range(unlabel_num):
            unlabel_hammin[i,j] = rbf(unlabel_output[i],unlabel_output[j],sigmoid)
    return unlabel_hammin

def change_unlabel_hammin(unlabel_hammin,Inconsis_index):
    #Iteratively delete the Inconsis_index row of unlabel_hammin, and the Inconsis_index column
    Inconsis_num = len(Inconsis_index)
    for i in range(Inconsis_num):
        index = Inconsis_index[i]
        unlabel_hammin = np.delete(unlabel_hammin,[index],axis = 0)
        unlabel_hammin = np.delete(unlabel_hammin,[index],axis = 1)
    return unlabel_hammin



