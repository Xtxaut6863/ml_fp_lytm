from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path = os.path.abspath(os.path.join(current_path, os.path.pardir))

import tensorflow as tf
import numpy as np
import pandas as pd
sys.path.append(par_path+'/test/')
from load_data import load_normalized_data
from plot_util import plot_normreconvert_relation
from plot_util import plot_normreconvert_pred
from sklearn.metrics import r2_score
sys.path.append(par_path)
from custom_estimator import my_dnn_regression_fn


# Set the models path.
model_path = current_path+'/models/imf1/'
data_path = par_path+'/data/'


def main(argv):
    """ Builds, Trians, and evaluates the model. """
    assert len(argv) == 1
    # (x_train_dev, y_train_dev), (x_train, y_train), (x_dev, y_dev), (x_test, y_test), (series_max,series_min) = load_normalized_data("vmd_imf1.xlsx")

    # laod data from local disk.
    # Download the full original data set.
    # full_data_set = pd.read_excel('F:/ml_fp_lytm/data/vmd_imf1.xlsx').drop("TM", axis=1)
    # # Delete the rows with unkonws
    # full_data_set.dropna()
    # serise_max = full_data_set.max()
    # series_min = full_data_set.min()

    # Create parse_function
    def parser(example_proto):
        dicts = {
            'X1': tf.FixedLenFeature(shape=(), dtype=tf.float32),
            'X2': tf.FixedLenFeature(shape=(), dtype=tf.float32),
            'X3': tf.FixedLenFeature(shape=(), dtype=tf.float32),
            'Y': tf.FixedLenFeature(shape=(), dtype=tf.float32),
        }
        parsed_example = tf.parse_single_example(example_proto,dicts)
        return {
            'X1': parsed_example['X1'],
            'X2': parsed_example['X2'],
            'X3': parsed_example['X3']
        }, parsed_example['Y']



    def get_dataset_inp_fn(filenames, epochs=20):
        def dataset_input_fn():
            dataset = tf.data.TFRecordDataset(filenames)
            # Use `Dataset.map()` to build a pair of a feature dictionary and a label
            # tensor for each example.
            dataset = dataset.map(parser)
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(32)
            dataset = dataset.repeat(epochs)
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels
        return dataset_input_fn



    train_input_fn = get_dataset_inp_fn([data_path+'vmd_imf1_train.tfrecord'],epochs=20)
    dev_input_fn = get_dataset_inp_fn([data_path+'vmd_imf1_dev.tfrecord'],epochs=20)

    feature_columns = [
        tf.feature_column.numeric_column("X1"),
        # tf.feature_column.numeric_column("X2"),
        # tf.feature_column.numeric_column("X3"),
        # tf.feature_column.numeric_column("X4"),
        # tf.feature_column.numeric_column("X5"),
        # tf.feature_column.numeric_column("X6"),
        # tf.feature_column.numeric_column("X7"),
        # tf.feature_column.numeric_column("X8"),
        # tf.feature_column.numeric_column("X9"),
        # tf.feature_column.numeric_column("X10"),
        # tf.feature_column.numeric_column("X11"),
        # tf.feature_column.numeric_column("X12"),
        # tf.feature_column.numeric_column("X13"),
        # tf.feature_column.numeric_column("X14"),
        # tf.feature_column.numeric_column("X15"),
        # tf.feature_column.numeric_column("X16"),
        # tf.feature_column.numeric_column("X17"),
        # tf.feature_column.numeric_column("X18"),
        # tf.feature_column.numeric_column("X19"),
    ]

    my_check_point_config = tf.estimator.RunConfig(
        save_checkpoints_steps=50,
        keep_checkpoint_max=1000  #Retain the 50most recent checkpoints
    )

    for learning_rate in [0.01]:
        decay_steps = 1000  # Learning rate decay steps
        decay_rate = 0.98  # Learning rate decay rate
        hidden_units = [8]  #5:14
        batch_size = 256
        drop_rates = [0.0]
        # construct a hyperparameter str
        hparam_str = 'lr' + str(learning_rate) + '_ds' + str(
            decay_steps) + '_dr' + str(decay_rate) + '_hu' + str(
                hidden_units) + '_bs' + str(batch_size) + '_drop' + str(
                    drop_rates)

        model_dir = model_path + hparam_str

        # Build a custom Estimator, using the model_fn
        # 'params' is passed through the 'model_fn'
        model = tf.estimator.Estimator(
            model_fn=my_dnn_regression_fn,
            model_dir=model_dir,
            params={
                'feature_columns': feature_columns,
                'learning_rate': learning_rate,
                # without learning_rate decay
                'decay_steps': decay_steps,
                'decay_rate': decay_rate,
                'optimizer': tf.train.AdamOptimizer,
                'hidden_units': hidden_units,
                'drop_rates': drop_rates
            },
            config=my_check_point_config)

        STEPS = 100
        for i in range(20):
            # Train the model
            model.train(input_fn=train_input_fn, steps=STEPS)

            # Evaluate how the model performs on a data it has not yet seen.
            eval_result = model.evaluate(input_fn=dev_input_fn, steps=STEPS)

            # The evaluation returns a python dictionary. The 'average loss' key is
            # hold the Mean Square Error (MSE)
            average_loss = eval_result['mse']
            # Convert MSE to root mean square error (RMSE)
            print("\n" + 80 * "*")
            print("\nRMS error for the validation set: {:.8f}".format(
                average_loss))
            print()




if __name__ == '__main__':
    # The estimator periodically generates "INFO" logs; make this logs visable
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
