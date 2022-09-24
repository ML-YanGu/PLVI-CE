import numpy as np
from sklearn.metrics import f1_score
from numpy.linalg import pinv



class LW_ELM(object):
    def __init__(self, train_data,train_target,test_data,test_target,hidden_num):
        #The number of data and labels that are passed in is the number of data as the number of abscissa
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.test_target = test_target

        #Number of hidden layer nodes
        self.hidden_num = hidden_num
        self.train_num = len(self.train_data) #Number of training instances
        self.test_num = len(self.test_data) #Number of testing instances
        self.num_feature = len(self.train_data[0])  # the number of feature
        self.class_num = len(self.train_target[0])  #the number of class
        # Define the initial weight
        self.input_weight = np.random.uniform(-1, 1, (self.hidden_num, self.num_feature))
        # Define the initial bias
        self.hidden_init_bias = np.random.uniform(-1, 1, (1, self.hidden_num))

        category_list = np.zeros(self.class_num)  #Record the number of instances of each category
        category_list = np.array(category_list)
        category_list = self.train_target.sum(axis = 0)

        #Converts the result set to 1 and -1
        self.train_target = self.train_target * 2 - 1
        self.test_target = self.test_target * 2 - 1

        class_list = np.zeros(self.class_num)
        class_list = np.array(class_list)
        #Update the weights for each category
        for p in range(self.class_num):
            class_list[p] = np.sqrt(np.max(category_list) / category_list[p])

        for i in range(self.class_num):
            for j in range(self.train_num):
                if self.train_target[j,i] == 1:
                    self.train_target[j,i] = class_list[i]

    # define sigmoid funcation
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def train(self):
        tempH = np.dot(self.train_data,np.transpose(self.input_weight))
        # The bias vector is expanded to the bias matrix by the way of row following row
        hidden_bias = self.hidden_init_bias
        for i in range(self.train_num - 1):
            hidden_bias = np.row_stack((hidden_bias, self.hidden_init_bias))
        tempH = tempH + hidden_bias
        #Compute the output of the hidden layer
        label_H = self.sigmoid(tempH)
        H_trans = np.transpose(label_H)
        #The output weight matrix
        output_weight = np.dot(pinv(label_H),self.train_target)
        return output_weight

    def test(self,output_weight):
        test_tempH = np.dot(self.test_data,np.transpose(self.input_weight))
        hidden_bias = self.hidden_init_bias
        for i in range(self.test_num - 1):
            hidden_bias = np.row_stack((hidden_bias,self.hidden_init_bias))
        test_tempH = test_tempH + hidden_bias
        test_H = self.sigmoid(test_tempH)

        #计Compute the true output of the test set
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
        return test_output,f1_mic

