import math
import numpy as np
def micro_f1(actual,predict):
    #actual: The true label matrix of the number of labels times the number of samples
    #predict:Predicted label matrix of the number of labels multiplied by the number of samples
    actual = np.transpose(actual)
    predict = np.transpose(predict)

    actual_row = len(actual)
    actual_column = len(actual[0])
    actual_num = actual_row * actual_column

    predict_row = len(predict)
    predict_column = len(predict[0])
    predict_num = predict_row * predict_column

    actual_temp = np.transpose(actual)
    predict_temp = np.transpose(predict)
    actual_trans = actual_temp.reshape(actual_num,1)
    predict_trans = predict_temp.reshape(predict_num,1)

    intSec = np.dot(np.transpose(actual_trans),predict_trans)
    non_zero_pre = actual.sum(axis= 1)
    dePre = non_zero_pre.sum()  #The number of positive in the real label
    non_zero_rec = predict.sum(axis = 1)
    deRec = non_zero_rec.sum()  #Predict the number of positives in the label
    if dePre != 0:
        precision = intSec / dePre
    else :
        precision = 0

    if deRec != 0:
        recall = intSec / deRec
    else:
        recall = 0

    # if recall.all() == 0 and precision.all() == 0:
    #     micro_f1 = 0
    # else:
    #     micro_f1 = (2 * precision * recall) / (precision + recall)
    if recall != 0 or precision != 0:
        micro_f1 = (2 * precision * recall) / (precision + recall)
    else:
        micro_f1 = 0

    return micro_f1