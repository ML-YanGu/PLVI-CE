import math
import numpy as np
import random
from sigmoid import sigmoid
#select the instance with the max margin
def random_sele(unlabel_data,batch_num):
    unlabel_num = len(unlabel_data)
    random_select = random.sample(range(0, unlabel_num), batch_num)
    return random_select



