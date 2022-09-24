# common label propagation method
import numpy as np
from sklearn.metrics import hamming_loss
def rbf(x1, x2, rbf_sigma):
    diff = x1 - x2
    rbf_distance = np.exp(sum(diff ** 2) / (-2.0 * rbf_sigma ** 2))
    return rbf_distance

# Calculate the correlation of each instance
def label_pro_only(label_data, unlabel_data, unlabel_output,label_target, batch_num, rbf_distance):
    # rbf_ditance
    unlabel_num = len(unlabel_data)
    label_num = len(label_data)
    class_num = len(label_target[0])
    k_neighbor = int(np.ceil(pow(label_num, 1 / 2)))
    #LC
    sum_temp = label_target.sum(axis=0)
    label_actual_sum = sum_temp.sum()
    average_cardinal = int(np.ceil(label_actual_sum / label_num))

    # use rbf_distance to obtain knn_distance
    knn_distance = np.zeros((unlabel_num, k_neighbor))
    knn_distance = np.array(knn_distance)
    knn_distance = knn_distance.astype(np.int64)
    distance_sort = np.argsort(rbf_distance, axis=1)
    for i in range(unlabel_num):
        distance_sort[i] = distance_sort[i][::-1]
    knn_distance = distance_sort[:, 0:k_neighbor]
    # Return the index of the labeled samples of the K-nearest neighbors corresponding to each unlabeled sample

    # label propagation
    unlabel_sum = np.zeros((unlabel_num, class_num))
    unlabel_sum = np.array(unlabel_sum)
    for p in range(class_num):
        # Calculate the score for each category for each instance
        for i in range(unlabel_num):
            for k in range(k_neighbor):
                unlabel_sum[i, p] += rbf_distance[i, knn_distance[i, k]] * label_target[knn_distance[i, k], p]
    output_target = np.zeros((unlabel_num, class_num))
    output_target = np.array(output_target)
    for i in range(unlabel_num):
        unlabel_index = np.argsort(unlabel_sum[i])
        unlabel_index = unlabel_index[::-1]
        unlabel_index = unlabel_index[0:average_cardinal]
        output_target[i, unlabel_index] = 1  # predicted label

    hammin_list = np.zeros(unlabel_num)
    hammin_list = np.array(hammin_list)
    for i in range(unlabel_num):
        hammin_list[i] = hamming_loss(output_target[i], unlabel_output[i])
    metric = hammin_list
    metric = np.argsort(metric)
    metric = metric[::-1]
    batch_list = metric[0:batch_num]
    return batch_list







