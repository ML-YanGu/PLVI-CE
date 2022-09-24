import numpy as np
from sklearn.metrics import f1_score
from micro_f1 import micro_f1
from numpy.linalg import pinv
from sklearn.metrics import hamming_loss
from Hammin_loss import Hammin_loss
import cmath
from numpy.linalg import inv


class LW_MLELM(object):
    def __init__(self, label_data,label_target,unlabel_data, unlabel_target,test_data,test_target,hidden_num,C,input_weight,hidden_init_bias):
        #The number of data and labels that are passed in is the number of data as the number of abscissa
        self.label_data = np.array(label_data)
        self.unlabel_data = np.array(unlabel_data)
        self.test_data = np.array(test_data)
        self.label_target = np.array(label_target,np.float64)
        self.test_target = np.array(test_target)
        self.unlabel_target = np.array(unlabel_target)
        #Number of hidden layer nodes
        self.hidden_num = hidden_num
        self.C = C
        self.input_weight = input_weight
        self.hidden_init_bias = hidden_init_bias
        self.label_data_num = len(self.label_data)  # Number of labeled instances
        self.unlabel_data_num = len(self.unlabel_data) #Number of unlabeled instances
        self.test_data_num = len(self.test_data) #Number of testing instances
        self.num_feature = len(self.label_data[0])  # the number of feature
        self.class_num = len(self.label_target[0])  #the number of class

        #Converts the result set to 1 and -1
        self.label_target = self.label_target * 2 - 1
        self.unlabel_target = self.unlabel_target * 2 - 1
        self.test_target = self.test_target * 2 - 1

        #Update the weights for each category
        positive_T = np.zeros(self.class_num)
        for i in range(self.class_num):
            positive_num = 0
            for j in range(self.label_data_num):
                if self.label_target[j,i] == 1:
                    positive_num = positive_num + 1
            if positive_num == 0:
                positive_num = 1
            a = (self.label_data_num - positive_num) / positive_num
            positive_T[i] = np.sqrt(a)

        for p in range(self.class_num):
            for q in range(self.label_data_num):
                if self.label_target[q,p] == 1:
                    self.label_target[q,p] = positive_T[p]


    # define sigmoid funcation
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def train(self):
        tempH = np.dot(self.label_data,np.transpose(self.input_weight))
        hidden_bias = self.hidden_init_bias
        for i in range(self.label_data_num - 1):
            hidden_bias = np.row_stack((hidden_bias, self.hidden_init_bias))
        tempH = tempH + hidden_bias
        #Compute the output of the hidden layer
        label_H = self.sigmoid(tempH)
        H_trans = np.transpose(label_H)
        #The output weight matrix
        item = np.dot(H_trans,label_H)
        # first = inv(np.dot(label_H,H_trans))
        # output_weight = np.dot(np.dot(first,H_trans),self.label_target)
        label_taregt = self.label_target
        output_weight = np.dot(pinv(label_H),self.label_target)
        return output_weight

    def output_unlabel(self,output_weight):
        # Return the class probability of the unlabeled set, which is used to select
        # the most valuable sample points in the unlabeled set for active learning
        unlabel_tempH = np.dot(self.unlabel_data,np.transpose(self.input_weight))
        hidden_bias = self.hidden_init_bias
        #Augment hidden layer bias by line append
        for i in range(self.unlabel_data_num - 1):
            hidden_bias = np.row_stack((hidden_bias,self.hidden_init_bias))
        unlabel_tempH = unlabel_tempH + hidden_bias
        #Computes the hidden layer output of unlabeled sets
        unlabel_H = self.sigmoid(unlabel_tempH)
        unlabel_H_tarns = np.transpose(unlabel_H)
        #Calculate the true output value
        unlabel_output = np.dot(unlabel_H,output_weight)
        #Convert to class probability
        unlabel_probaility = self.sigmoid(unlabel_output)
        hidden_bias_label = self.hidden_init_bias
        label_tempH = np.dot(self.label_data,np.transpose(self.input_weight))
        for i in range(self.label_data_num - 1):
            hidden_bias_label = np.row_stack((hidden_bias_label,self.hidden_init_bias))
        label_tempH = label_tempH + hidden_bias_label
        #Hidden layer output of labeled set
        label_H = self.sigmoid(label_tempH)
        #Calculate the true output value
        label_output = np.dot(label_H,output_weight)
        #Convert to class probability
        label_probaility = self.sigmoid(label_output)

        class_num = len(unlabel_output)
        unlabel_num = len(unlabel_output[0])
        for i in range(class_num):
            for j in range(unlabel_num):
                if self.unlabel_target[i, j] == -1:
                    self.unlabel_target[i, j] = 0
                if unlabel_output[i, j] > 0:
                    unlabel_output[i, j] = 1
                else:
                    unlabel_output[i, j] = 0
        return label_probaility,unlabel_probaility,unlabel_output

    def test(self,output_weight):
        test_tempH = np.dot(self.test_data,np.transpose(self.input_weight))
        hidden_bias = self.hidden_init_bias
        #Augment hidden layer bias by line append
        for i in range(self.test_data_num - 1):
            hidden_bias = np.row_stack((hidden_bias,self.hidden_init_bias))
        test_tempH = test_tempH + hidden_bias
        #Computes the hidden layer output of the test set
        test_H = self.sigmoid(test_tempH)

        #Compute the true output of the test set
        test_output = np.dot(test_H,output_weight)
        class_num = len(test_output[0])
        test_num = len(test_output)
        for i in range(class_num):
            for j in range(test_num):
                if self.test_target[j,i] == -1:
                    self.test_target[j,i] = 0
                if test_output[j,i] > 0:
                    test_output[j,i] = 1
                else:
                    test_output[j,i] = 0
        f1_mic = f1_score(self.test_target, test_output, average='micro')
        f1_mac = f1_score(self.test_target, test_output, average='macro')
        hammin_loss = Hammin_loss(test_output, self.test_target)
        # f1_mic = micro_f1(self.test_target,test_output)
        return f1_mic,f1_mac,hammin_loss

