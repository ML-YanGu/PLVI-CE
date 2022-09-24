import numpy as np
from ELM import LW_MLELM
from active_criterion.label_pro_jd import label_pro_jd
from compute_rbf_distance import compute_rbf_distance
from compute_rbf_distance import change_rbf
from compute_sigma import calculate_sigma

class active_label_pro_jd(object):
    def __init__(self, label_data,label_target,unlabel_data, unlabel_target,test_data,test_target,hidden_num,C,query_num,input_weight,hidden_init_bias,batch_num):
        # query_num refers to the number of active queries
        # Iw refers to a randomly specified weight from the input layer to the hidden layer
        # A bias is a randomly specified bias of a hidden layer
        self.label_data = np.array(label_data)
        self.unlabel_data = np.array(unlabel_data)
        self.test_data = np.array(test_data)
        self.label_target = np.array(label_target)
        self.unlabel_target = np.array(unlabel_target)
        self.test_target = np.array(test_target)
        self.num_feature = len(self.label_data[0])
        self.query_num = query_num
        self.hidden_num = hidden_num
        self.C = C
        self.input_weight = input_weight
        self.hidden_init_bias = hidden_init_bias
        self.batch_num = batch_num
        self.iter_num = self.query_num // self.batch_num
        self.test_mic_f1measure = np.zeros(self.iter_num + 1)
        self.test_mac_f1measure = np.zeros(self.iter_num + 1)
        self.test_hammin = np.zeros(self.iter_num + 1)

        #First train with the initial annotated set
        lw_elm = LW_MLELM.LW_MLELM(label_data, label_target, unlabel_data, unlabel_target, test_data, test_target, self.hidden_num,
                                   self.C, self.input_weight, self.hidden_init_bias)
        self.output_weight = lw_elm.train()
        self.label_probaility,self.unlabel_probaility,self.unlabel_output = lw_elm.output_unlabel(self.output_weight)
        f1_mic,f1_mac,hammin_loss = lw_elm.test(self.output_weight)
        self.test_mic_f1measure[0] = f1_mic
        self.test_mac_f1measure[0] = f1_mac
        self.test_hammin[0] = hammin_loss

        self.sigma = calculate_sigma(self.label_data)
        self.rbf_distance = compute_rbf_distance(self.label_data,self.unlabel_data,self.sigma)
    def train_active(self):
        count = 0
        count_loop = 0
        # for i in range(1,self.query_num + 1):
        while 1:
            count_loop += 1
            #Use a query strategy to pick the most valuable instance
            # rbf
            Inconsis_index = label_pro_jd(self.label_data, self.unlabel_data, self.unlabel_output,self.label_target, self.batch_num, self.rbf_distance,self.label_probaility,self.unlabel_probaility)
            # before update label_data and unlabel_data, to update rbf_distance
            Inconsis_index = np.sort(Inconsis_index)
            Inconsis_index = Inconsis_index[::-1]
            self.sigma,self.rbf_distance = change_rbf(self.rbf_distance,Inconsis_index,self.label_data,self.unlabel_data,self.batch_num,self.sigma)
            for i in range(self.batch_num):
            # add to a labeled set
                self.label_data = np.row_stack((self.label_data, self.unlabel_data[Inconsis_index[i]]))
                self.label_target = np.row_stack((self.label_target, self.unlabel_target[Inconsis_index[i]]))
                count += 1
            #Delete from the start of the large index
            for i in range(self.batch_num):

                self.unlabel_data = np.delete(self.unlabel_data, Inconsis_index[i], axis=0)
                self.unlabel_target = np.delete(self.unlabel_target, Inconsis_index[i], axis=0)
            # Retrain the classifier with the new labeled set
            lw_elm = LW_MLELM.LW_MLELM(self.label_data, self.label_target, self.unlabel_data, self.unlabel_target,
                                             self.test_data,self.test_target, self.hidden_num, self.C, self.input_weight,
                                             self.hidden_init_bias)
            self.output_weight = lw_elm.train()
            self.label_probaility,self.unlabel_probaility,self.unlabel_output = lw_elm.output_unlabel(self.output_weight)
            f1_mic,f1_mac,hammin_loss = lw_elm.test(self.output_weight)
            self.test_mic_f1measure[count_loop] = f1_mic
            self.test_mac_f1measure[count_loop] = f1_mac
            self.test_hammin[count_loop] = hammin_loss
            #The accuracy of the trained classifier was recorded
            if count == self.query_num:
                break
        return self.test_mic_f1measure,self.test_mac_f1measure,self.test_hammin

