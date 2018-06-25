import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow import data
import matplotlib.pyplot as plt

import os
current_path = os.path.dirname(os.path.abspath(__file__))   #lstm_test
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))    #TF_PROJECTS
par_path_2 = os.path.abspath(os.path.join(par_path_1, os.path.pardir))  
data_path = par_path_2 + '\\data\\'
model_path = current_path+'\\models\\orig\\'
print(10 * '-' + ' Current Path: {}'.format(current_path))
print(10 * '-' + ' Parent Path: {}'.format(par_path_1))
print(10 * '-' + ' Grandpa Path: {}'.format(par_path_2))
print(10 * '-' + ' Data Path: {}'.format(data_path))
print(10 * '-' + ' Model Path: {}'.format(model_path))

print(tf.__version__)

MODEL_NAME = 'orig'
SEQUENCE_LENGTH = 7
INPUT_SEQUENCE_LENGTH=6
OUTPUT_SEQUENCE_LENGTH=SEQUENCE_LENGTH-INPUT_SEQUENCE_LENGTH
TRAIN_DATA_SIZE = 4331
DEV_DATA_SIZE=541
TEST_DATA_SIZE=541
TRAIN_DATA_FILE=data_path+'lstm_unsample_orig_train.csv'
RESUME_TRAINING=False
MULTI_THREADING=True

