# A simple case for custome tf.estimator.DNNRegressor
# Test flow prediction in the area LaoYu
# Data comes from the station TanMiao
# Use the high level tensorflow API
""" Regression using the DNNRegressor Estimator """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score

def my_dnn_regression_fn(features, labels, mode, params):
    """ A model function implementing DNN regression for a custom Estimator """

    # Extract the input intp  a dense layer, according to feature_columns.
    with tf.name_scope("feature_columns"):
        top = tf.feature_column.input_layer(features,
                                            params['feature_columns'])

    # Iterate over the "hidden_units" list of layer size, default is [20].
    for units in params.get('hidden_units', [20]):
        # Add a hidden layer, densely connected on top of previous layer.
        # Set dropout rate for each hidden layer, default is 0.0; no dropout
        for drop in params.get('drop_rates', [0.0]):
            top = tf.layers.dense(inputs=top, units=units, activation=tf.nn.relu)
            top = tf.layers.dropout(top, rate=drop)

    # Connect a liner output layer on top.
    output_layer = tf.layers.dense(inputs=top, units=1)

    output_layer = tf.cast(output_layer, tf.float64)
    # Reshape the output layer to a 1-deim Tensor to return predictions
    predictions = tf.squeeze(output_layer, 1)
    # predictions = tf.reshape(predictions, [-1, 1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        # In 'PREDICT' mode we only need to return predictions.
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions={"predictions": predictions})

    # calculate the loss using mean squared error
    average_loss = tf.losses.mean_squared_error(labels, predictions)

    # Pre-made estimators use the total_loss instead of the average,
    # so report total_loss for compatibility.
    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size) * average_loss

    # Create training op with exponentially decaying learning rate.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get("optimizer", tf.train.AdamOptimizer)
        learning_rate = tf.train.exponential_decay(
            # learning_rate=0.1,
            learning_rate=params.get('learning_rate', 0.1),
            global_step=tf.train.get_global_step(),
            # default: no learning_rate decay will be executed.
            decay_steps=params.get('decay_steps', 0),
            decay_rate=params.get('decay_rate', 1),
            staircase=True)
        tf.summary.scalar("learning_rate", learning_rate)
        optimizer = optimizer(learning_rate)
        # optimizer = optimizer(params.get("learning_rate", learning_rate))
        train_op = optimizer.minimize(
            loss=average_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op)

    # In the evaluation mode we will calculate evaluation metrics.
    assert mode == tf.estimator.ModeKeys.EVAL

    # Calculate mean squared error
    mse = tf.metrics.mean_squared_error(labels, predictions)

    # Add the mse to collection of evaluation metrics.
    eval_metrics = {"mse": mse}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        # Report sum of error for compatibility with pre-made estimators.
        loss=total_loss,
        eval_metric_ops=eval_metrics)
