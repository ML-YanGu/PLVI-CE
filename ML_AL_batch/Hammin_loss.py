import numpy as np
def Hammin_loss(pre_Labels,test_targets):
    num_class,num_instance = np.mat(test_targets).shape
    temp = sum((pre_Labels != test_targets))
    miss_pairs = sum(temp)
    hammingLoss = miss_pairs/(num_class*num_instance)
    return hammingLoss
