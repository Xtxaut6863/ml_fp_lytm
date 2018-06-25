# A simple test case for tf.contirb.learn.DNNRegressor
# Test flow prediction in the area LaoYu
# Data comes from the station TanMiao
# Use the high level tensorflow API

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics.regression import mean_absolute_error
from sklearn.metrics.regression import mean_squared_error
import math

tf.logging.set_verbosity(tf.logging.INFO)

# First, create one or more input functions.
# In this download data from
full_train_data = pd.read_excel(
    "F:/ml_fp_lytm/data/logtrans_day_train.xlsx").drop(
        "TM", axis=1)
full_dev_data = pd.read_excel(
    "F:/ml_fp_lytm/data/logtrans_day_dev.xlsx").drop(
        "TM", axis=1)
full_test_data = pd.read_excel(
    "F:/ml_fp_lytm/data/logtrans_day_test.xlsx").drop(
        "TM", axis=1)

# create the train,development and test set
train_inputs = full_train_data.drop("Y", axis=1)
train_outputs = full_train_data["Y"]
dev_inputs = full_dev_data.drop("Y", axis=1)
dev_outputs = full_dev_data["Y"]
test_inputs = full_test_data.drop("Y", axis=1)
test_outputs = full_test_data["Y"]

# get the size of trian,development and test dataset
train_len = train_outputs.size
dev_len = dev_outputs.size
test_len = test_outputs.size

# # input builders
# # create the input function for train
# def input_fn_train():
#     return train_inputs,train_outputs
# # create thr input function for evaluation
# def input_fn_eva():
#     return dev_inputs,dev_outputs
# # create the input function for finally predicting test
# def input_fn_pred_test():
#     return test_inputs



# second step: define the feature columns
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=6)]

# third step: instantiate an Estimator, specifying the feature columns and hyperparameters
model = tf.contrib.learn.DNNRegressor(
    feature_columns=feature_columns,
    hidden_units=[10, 10, 10],
    optimizer=tf.train.AdamOptimizer(0.01),
    model_dir=
    'F:/ml_fp_lytm/tf_projects/test/models/temp0'
)

# fourth step: call one or more method on Estimator object
model.fit(x=train_inputs, y=train_outputs, batch_size=200, steps=5000)

# evaluate the model by using development set
eval_result = model.evaluate(x=dev_inputs, y=dev_outputs, steps=5000)

# pedicting data use train set
train_pred = model.predict(x=train_inputs, as_iterable=False)
dev_pred = model.predict(x=dev_inputs, as_iterable=False)
test_pred = model.predict(x=test_inputs, as_iterable=False)

print(train_pred)

train_t = np.linspace(start=1, stop=train_len, num=train_len)
dev_t = np.linspace(start=1, stop=dev_len, num=dev_len)
test_t = np.linspace(start=1, stop=test_len, num=test_len)

# recovert the records and predictons to original data set
# transfer function: z=2.3*log(G+1), where G is the original data and z is the trab=nsformed data
# here G=10^z/2.3-1
# reconvert the train,dev and test data set
orig_train_outputs = np.power(10, train_outputs / 2.3) - 1
orig_dev_outputs = np.power(10, dev_outputs / 2.3) - 1
orig_test_outputs = np.power(10, test_outputs / 2.3) - 1
train_pred = np.array(list(train_pred))
dev_pred = np.array(list(dev_pred))
test_pred = np.array(list(test_pred))
# reconvert the train,dev and test predictions
orig_train_pred = np.power(10, train_pred / 2.3) - 1
orig_dev_pred = np.power(10, dev_pred / 2.3) - 1
orig_test_pred = np.power(10, test_pred / 2.3) - 1



# compare the train records and predictions
plt.figure(num=1, figsize=(19.2, 10.8))
plt.title('flow predicting based on DNN')
plt.xlabel('Time(d)')
plt.ylabel('flow(m^3/s)')
plt.plot(train_t, orig_train_outputs, '-', color='blue', label='records')
plt.plot(train_t, orig_train_pred, '--', color='red', label='predictions')
plt.savefig(
    "F:/ml_fp_lytm/tf_projects/test/models/images/train_pred_lines.png",
    format='png')
plt.show()

# use numpy polyfit to fit a line for train records and train predictions
z_train = np.polyfit(orig_train_outputs, orig_train_pred, 1)

# analyse the correlation between train records and train predictions
# fit a straight line for train records and train predictions
linear_fit = z_train[0] * orig_train_outputs + z_train[1]
ideal_fit = 1 * orig_train_outputs
plt.figure(num=1, figsize=(8, 6))
plt.title('scatters for training data')
plt.xlabel('train records')
plt.ylabel('train predictions')
plt.plot(
    orig_train_outputs,
    orig_train_pred,
    'o',
    color='blue',
    linewidth=1.0,
    label='train scatters')
plt.plot(orig_train_outputs, linear_fit, '--', color='red', label='Linear fit')
plt.plot(orig_train_outputs, ideal_fit, '-', color='black', label='Ideal fit')
plt.savefig(
    "F:/ml_fp_lytm/tf_projects/test/models/images/train_scatters.png",
    format='png')
plt.show()
