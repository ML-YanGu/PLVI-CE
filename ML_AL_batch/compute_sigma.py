#Initial calculation of the distance matrix for the labeled set data
import numpy as np
def calculate_sigma(label_data):
    label_num = len(label_data)
    label_dis_sum = 0
    num = (label_num * (label_num - 1)) / 2
    for i in range(label_num - 1):
        for j in range(i + 1,label_num):
            diff = label_data[i] - label_data[j]
            dis_for_one = np.sqrt(sum(diff ** 2))
            label_dis_sum += dis_for_one
    sigma = label_dis_sum / num
    return sigma


def change_sigma(label_data,sigma):
    label_num = len(label_data)
    num = ((label_num - 1) * (label_num - 2)) / 2
    label_sum = ((label_num - 1) * label_num) / 2
    label_dis_sum = sigma * num
    for i in range(label_num - 1):
        diff = label_data[label_num - 1] - label_data[i]
        dis_for_one = np.sqrt(sum(diff ** 2))
        label_dis_sum += dis_for_one
    sigma = label_dis_sum / label_sum
    return sigma

