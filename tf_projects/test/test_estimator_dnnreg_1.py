# A simple test case for tf.estimator.DNNRegressor
# This is the offical.
# Test flow prediction in the area LaoYu
# Data comes from the station TanMiao
# Use the high level tensorflow API

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import import_tanmiao
import pandas as pd

import plot_util

STEPS = 3000

fig_savepath = "F:/ml_fp_lytm/tf_projects/test/models/images/"


def main(argv):
    """ Build, train, and evaluates the model. """
    assert len(argv) == 1
    (x_train, y_train), (x_dev, y_dev), (x_test,
                                         y_test) = import_tanmiao.load_data(logtrans=True)

    # build the training input_fn
    def input_train(features=x_train, labels=y_train, batch_size=128):
        """ An input function for training """
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        # Buffer size should be greater than the size of training samples.
        # repeat means epoch, .repeat(5) mean epoch=5
        # batch size is the mini-batch size.
        dataset = dataset.shuffle(4000).batch(
            batch_size).repeat().make_one_shot_iterator().get_next()

        # return the dataset
        return dataset

    # Build the validation input_fn
    def input_dev(featutres=x_dev, labels=y_dev, batch_size=128):
        dataset = tf.data.Dataset.from_tensor_slices((dict(featutres), labels))

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
        # tf.feature_column.numeric_column("X7")
    ]

    model_dir = "F:/ml_fp_lytm/tf_projects/test/models/temp1"

    # Build the estimator
    model = tf.estimator.DNNRegressor(
        hidden_units=[10, 10],
        feature_columns=feature_columns,
        model_dir=model_dir)
    # Train the model
    model.train(input_fn=input_train, steps=STEPS)

    # Evaluate how the model preforms on data it has not yet seen.
    eval_result = model.evaluate(input_fn=input_dev)

    # The evaluation returns a python dictionary. The 'average loss' key is
    # hold the Mean Square Error (MSE)
    average_loss = eval_result["average_loss"]

    # Convert MSE to root mean square error (RMSE)
    print("\n" + 80 * "*")
    print("\nRMS error for the validation set: ${:.2f}".format(
        average_loss**0.5))
    print()

    test_pred_input_fn = tf.estimator.inputs.pandas_input_fn(
        x_train, shuffle=False)

    test_pred_results = model.predict(input_fn=test_pred_input_fn)

    test_predicitons_list = []

    for dicts in list(test_pred_results):
        test_predicitons_list.append(dicts['predictions'][0])

    test_predictions = np.array(test_predicitons_list)
    plot_util.plot_pred(
        y_train,
        test_predictions,
        fig_savepath=fig_savepath + "test_pred_1.png")
    plot_util.plot_relation(
        y_train,
        test_predictions,
        fig_savepath=fig_savepath + "test_rela_1.png")


if __name__ == "__main__":
    # The estimator periodically generates "INFO" logs; make this logs visable
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
