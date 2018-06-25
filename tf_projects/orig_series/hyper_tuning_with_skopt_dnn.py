from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import numpy as np
sys.path.append(
    'F:/ml_fp_lytm/tf_projects/test/')
from import_tanmiao import load_normalized_data
from plot_util import plot_normreconvert_relation
from plot_util import plot_normreconvert_pred
from sklearn.metrics import r2_score
sys.path.append('F:/ml_fp_lytm/tf_projects/')
from custom_estimator import my_dnn_regression_fn

from skopt import gp_minimize
from skopt.space import Real,Categorical,Integer
from skopt.utils import use_named_args

STEPS = 100

# Set the models path.
model_path = 'F:/ml_fp_lytm/tf_projects/orig_series/models/'


def main(argv):
    """ Builds, Trians, and evaluates the model. """
    assert len(argv) == 1

    # laod data from local disk.
    (x_train_dev,
     y_train_dev), (x_train, y_train), (x_dev, y_dev), (x_test, y_test), (
         series_mean, series_max,
         series_min) = load_normalized_data("orig_day_full_X.xlsx")

    # Build the training input_fn
    def input_train(features=x_train, labels=y_train, batch_size=512):
        """ An input function for training """
        # convert the input to a Dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shffling with a buffer larger than the data set ensures
        # that the examples are will mixed.
        dataset = dataset.shuffle(4000).batch(
            batch_size).repeat().make_one_shot_iterator().get_next()

        return dataset

    # Build the validation input_fn
    def input_dev(features=x_dev, labels=y_dev, batch_size=512):
        # Convert the input to a Dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffling
        dataset = dataset.shuffle(2000).batch(
            batch_size).make_one_shot_iterator().get_next()

        return dataset

    def input_test(features=x_test, labels=y_test, batch_size=512):
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffling
        dataset = dataset.shuffle(2000).batch(
            batch_size).make_one_shot_iterator().get_next()

        return dataset

    feature_columns = [
        tf.feature_column.numeric_column("X1"),
        tf.feature_column.numeric_column("X2"),
        tf.feature_column.numeric_column("X3"),
        tf.feature_column.numeric_column("X4"),
        tf.feature_column.numeric_column("X5"),
        tf.feature_column.numeric_column("X6"),
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

    log_dir = 'F:/ml_fp_lytm/tf_projects/orig_series/models/skopt_tuning_log/'

    space = [
        Real(1e-6, 1e-1, 'log-uniform',name='learning_rate'),
        # Integer(100, 1000, name='decay_steps'),
        # Real(0.95, 0.99, name='decay_rate'),
        # Categorical([1], [40], name='hidden_units'),
        # Real(0.1, 0.9, name='drop_rates'),
        # Categorical([True, False], name='skip_connections'),
        Categorical(feature_columns, name='feature_columns'),
        # Integer(1, 9, name='layers')
    ]

    @use_named_args(space)
    def score(**params):
        model_dir = os.path.join(log_dir,
        'lr_'+str(params.get('learning_rate'))
        # +'_ds'+str(params.get('decay_steps'))
        # +'_dr'+str(params.get('decay_rate'))
        # +'_hu'+str(params.get('hidden_units'))
        # +'_dpr'+str(params.get('drop_rates'))
        # +'_sk'+str(params.get('skip_connections'))
        # +'_lays'+str(params.get('layers'))
        )
        estimator = tf.estimator.Estimator(
            model_fn = my_dnn_regression_fn,
            model_dir=model_dir,
            params=params)
        trainspec = tf.estimator.TrainSpec(input_train)
        evalspec = tf.estimator.EvalSpec(input_dev)
        try:
            tf.estimator.train_and_evaluate(estimator,trainspec,evalspec)
            metrics = estimator.evaluate(input_test)
            return metrics['mse']
        except (tf.errors.ResourceExhaustedError,tf.train.NanLossDuringTrainingError):
            return 1e9

    gp_minimize(score,space)

if __name__ == '__main__':
    # The estimator periodically generates "INFO" logs; make this logs visable
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
