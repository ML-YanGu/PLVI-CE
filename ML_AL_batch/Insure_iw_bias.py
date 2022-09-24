
import numpy as np
def Insure_iw_bias(hidden_num,num_feature):
    #actual: The true label matrix of the number of labels times the number of samples
    #predict:Predicted label matrix of the number of labels multiplied by the number of samples
    # Random production weights (from -1, to 1, generated (num_feature row,num_hidden column))
    input_weight = np.random.uniform(-1, 1, (hidden_num,num_feature))
    # define bias
    hidden_init_bias = np.random.uniform(-1, 1, (1,hidden_num))
    return input_weight,hidden_init_bias