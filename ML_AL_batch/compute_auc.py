import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # 数据预处理
import pandas as pd
from draw_fig.draw_learningcurv_batch import draw

data = pd.read_csv('D:/matlab_project/AL-ML_result/measure/ml_al_os/label_corss/slashdot/reust3.csv')
data = data.values
data = np.array(data)
data_len = len(data) - 1
data = data[1:]
data = data[:,1:]
auc = np.zeros(10)
auc = np.array(auc)
for i in range(10):
    column_temp = data[:,i]
    auc[i] = (2 * np.sum(column_temp) - column_temp[0] - column_temp[data_len - 1]) / (2 * data_len)
print(auc)




