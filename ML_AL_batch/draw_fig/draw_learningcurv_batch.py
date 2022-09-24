import math
from sklearn.model_selection import train_test_split
import numpy as np
from Learning_batch.active_random_batch import active_random_batch
from Learning_batch.active_label_pro_batch import active_label_pro_batch
from Learning_batch.active_label_pro_jd import active_label_pro_jd
from Insure_iw_bias import Insure_iw_bias
from tqdm import tqdm

class draw(object):
    def __init__(self, train_data,train_target,test_data,test_target,hidden_num,C,batch_num):
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.test_target = test_target
        self.hidden_num = hidden_num
        self.num_feature = len(self.train_data[0])
        self.C = C
        self.batch_num = batch_num

    def draw_fig(self):
        unlabel_num = len(self.train_data) * 0.8
        query_num = int(0.6 * unlabel_num)
        query_num = int(self.batch_num * (query_num // self.batch_num))  # 将查询的数量变成10的倍数
        iter_num = query_num // self.batch_num
        print('iter_num',self.batch_num,iter_num)
        #
        f1measure_mic_random = np.zeros((10,iter_num + 1));f1measure_mac_random = np.zeros((10,iter_num + 1));hammin_random = np.zeros((10,iter_num + 1))

        f1measure_mic_labelpro = np.zeros((10,iter_num + 1));f1measure_mac_labelpro = np.zeros((10,iter_num + 1));hammin_labelpro = np.zeros((10,iter_num + 1))
        #
        f1_measure_mic_labelpro_jd = np.zeros((10,iter_num + 1));f1_measure_mac_labelpro_jd = np.zeros((10,iter_num + 1));hammin_labelpro_jd = np.zeros((10,iter_num + 1))
        epoach = 0
        for j in tqdm(range(10)):
            input_weight, hidden_init_bias = Insure_iw_bias(self.hidden_num, self.num_feature)
            label_data, unlabel_data, label_target, unlabel_target = train_test_split(self.train_data, self.train_target
                                                                     ,test_size=0.8,random_state=1)

            #label_pro
            active_label_pro = active_label_pro_batch(label_data, label_target, unlabel_data, unlabel_target,
                                                      self.test_data, self.test_target,self.hidden_num, self.C, query_num, input_weight,
                                                      hidden_init_bias, self.batch_num)
            test_mic_f1measure_labelpro,test_mac_f1measure_labelpro,test_hammin_labelpro = active_label_pro.train_active()
            print('label_pro')
            #

            # # random
            active_random = active_random_batch(label_data, label_target, unlabel_data, unlabel_target, self.test_data, self.test_target,
                                    self.hidden_num, self.C, query_num, input_weight, hidden_init_bias,self.batch_num)
            test_mic_f1measure_random,test_mac_f1measure_random,test_hammin_random = active_random.train_active()
            print('random')

            active_label_jd = active_label_pro_jd(label_data, label_target, unlabel_data, unlabel_target, self.test_data, self.test_target,
                                    self.hidden_num, self.C, query_num, input_weight, hidden_init_bias,self.batch_num)
            test_mic_f1measure_labelprojd,test_mac_f1measure_labelprojd,test_hammin_labelprojd = active_label_jd.train_active()


            for i in range(iter_num + 1):

                #
                f1measure_mic_random[j,i] = test_mic_f1measure_random[i]
                f1measure_mac_random[j,i] = test_mac_f1measure_random[i]
                hammin_random[j,i] = test_hammin_random[i]

                f1measure_mic_labelpro[j,i] = test_mic_f1measure_labelpro[i]
                f1measure_mac_labelpro[j,i] = test_mac_f1measure_labelpro[i]
                hammin_labelpro[j,i] = test_hammin_labelpro[i]

                f1_measure_mic_labelpro_jd[j,i] = test_mic_f1measure_labelprojd[i]
                f1_measure_mac_labelpro_jd[j,i] = test_mac_f1measure_labelprojd[i]
                hammin_labelpro_jd[j,i] = test_hammin_labelprojd[i]
        #
        f1measure_mic_random = f1measure_mic_random.mean(axis = 0)
        f1measure_mac_random = f1measure_mac_random.mean(axis = 0)
        hammin_random = hammin_random.mean(axis = 0)

        f1measure_mic_labelpro = f1measure_mic_labelpro.mean(axis = 0)
        f1measure_mac_labelpro = f1measure_mac_labelpro.mean(axis = 0)
        hammin_labelpro = hammin_labelpro.mean(axis = 0)

        f1_measure_mic_labelpro_jd = f1_measure_mic_labelpro_jd.mean(axis = 0)
        f1_measure_mac_labelpro_jd = f1_measure_mac_labelpro_jd.mean(axis = 0)
        hammin_labelpro_jd = hammin_labelpro_jd.mean(axis = 0)
        return f1measure_mic_random,f1measure_mac_random,hammin_random,f1measure_mic_labelpro,f1measure_mac_labelpro,hammin_labelpro





