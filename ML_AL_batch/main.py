import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # 数据预处理

from draw_fig.draw_learningcurv_batch import draw
import warnings
warnings.filterwarnings("ignore")

dataset = sio.loadmat('D:/python_project/ML_AL_batch/dataset/matfile/CAL500.mat')
# train_data = dataset['train_data']
# test_data = dataset['test_data']
# train_target = dataset['train_target']
# test_target = dataset['test_target']
# train_data = train_data.astype(np.float64)
# test_data = test_data.astype(np.float64)
#
# scaler1 = MinMaxScaler()
# scaler1.fit(train_data)
# scaler1.data_max_
# train_data = scaler1.transform(train_data)
#
# scaler2 = MinMaxScaler()
# scaler2.fit(test_data)
# scaler2.data_max_
# test_data = scaler2.transform(test_data)  #实现数据的归一化
#
# train_target = np.transpose(train_target)
# test_target = np.transpose(test_target)
# train_target = train_target.astype(np.int64)
# test_target = test_target.astype(np.int64)
data = dataset['data']
data = data.astype(np.float64)
scaler = MinMaxScaler()
scaler.fit(data)
scaler.data_max_
data = scaler.transform(data)
target = dataset['target']
target = target.astype(np.float64)
target = np.transpose(target)
# len_num = len(target)
# sum_temp = target.sum(axis = 0)
# sum_all = sum_temp.sum()
# c = sum_all / len_num
# print(c)

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3,random_state=1)


draw1 = draw(train_data,train_target,test_data,test_target,60,10,10)
draw1.draw_fig()

