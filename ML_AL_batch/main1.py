import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # 数据预处理
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from draw_fig.draw_learningcurv_batch import draw

data = pd.read_csv('D:/python_project/ML_AL_batch/dataset/matfile/movielens.csv')
data = data.values
data = np.array(data)

data = data[1:]
#target
data_feature = data[:,0:943]
target = data[:,944:]
len_num = len(target)
sum_temp = target.sum(axis = 0)
sum_all = sum_temp.sum()
c = sum_all / len_num
print(c)


data_feature = data_feature.astype(np.float64)
scaler = MinMaxScaler()
scaler.fit(data_feature)
scaler.data_max_
data_feature = scaler.transform(data_feature)

train_data, test_data, train_target, test_target = train_test_split(data_feature, target, test_size=0.3,random_state=1)
train_target = train_target.astype(np.int64)

train_num = len(train_data)
added_num = int(0.6 * train_num)

draw1 = draw(train_data,train_target,test_data,test_target,60,10,20)
draw1.draw_fig()






