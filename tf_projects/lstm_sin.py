import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from lstm import lstm_model
from data_processing import generate_data
tf.logging.set_verbosity(tf.logging.INFO)
LOG_DIR = './ops_logs/sin'
TIMESTEPS = 3
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = None
TRAINING_STEPS = 10000
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 100
regressor = learn.SKCompat(
    learn.Estimator(
        model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
        model_dir=LOG_DIR))
X, y = generate_data(
    np.sin,
    np.linspace(0, 100, 10000, dtype=np.float32),
    TIMESTEPS,
    seperate=False)

# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'], every_n_steps=PRINT_STEPS, early_stopping_rounds=1000)
print(X['train'])
print(y['train'])

regressor.fit(
    X['train'],
    y['train'],
    monitors=[validation_monitor],
    batch_size=BATCH_SIZE,
    steps=TRAINING_STEPS)

predicted = regressor.predict(X['test'])
rmse = np.sqrt(((predicted - y['test'])**2).mean(axis=0))
score = mean_squared_error(predicted, y['test'])
print("MSE: %f" % score)

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
