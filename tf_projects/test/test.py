from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
sys.path.append(
    'F:/ml_fp_lytm/tf_projects/test/')

from sklearn.metrics import r2_score
sys.path.append('F:/ml_fp_lytm/tf_projects/')
from custom_estimator import my_dnn_regression_fn

current_path = os.path.dirname(os.path.abspath(__file__))
par_path = os.path.abspath(os.path.join(current_path, os.path.pardir))
print(current_path)
print(par_path)


aa = [1,2,3,4,5,6]
ser1 = pd.DataFrame(aa,columns=['aa'])['aa']
bb = [4,5,6,7,8,9,10]
ser2 = pd.DataFrame(bb,columns=['bb'])['bb']

df = pd.DataFrame(pd.concat([ser1,ser2],axis=1))

# print(df)
