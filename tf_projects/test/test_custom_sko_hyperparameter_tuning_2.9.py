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
import import_tanmiao
import plot_util
from sklearn.metrics import r2_score
import make_dir
from skopt.space import Real,Integer
from skopt import gp_minimize

STEPS = 1000
model_path = "F:/ml_fp_lytm/tf_projects/test/models/temp2.3/tmp/"



def my_dnn_regression_fn(features, labels, mode, params):
    """ A model function implementing DNN regression for a custom Estimator """

    # Extract the input intp  a dense layer, according to feature_columns.
    with tf.name_scope("feature_columns"):
        top = tf.feature_column.input_layer(features,
                                            params['feature_columns'])

    # Iterate over the "hidden_units" list of layer size, default is [20].
    for units in params.get('hidden_units', [20]):
        # Add a hidden layer, densely connected on top of previous layer.
        for drop in params.get('drop_rates', [0.5]):
            top = tf.layers.dense(inputs=top, units=units, activation=tf.nn.relu)
            top = tf.layers.dropout(top, rate=drop)

    # Connect a liner output layer on top.
    output_layer = tf.layers.dense(inputs=top, units=1)

    output_layer = tf.cast(output_layer, tf.float64)
    # Reshape the output layer to a 1-deim Tensor to return predictions
    predictions = tf.squeeze(output_layer, 1)

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
            learning_rate = params.get('learning_rate',0.1),
            global_step=tf.train.get_global_step(),
            decay_steps=params.get('decay_steps',200),
            decay_rate=params.get('decay_rate',1))
        tf.summary.scalar("learning_rate",learning_rate)
        optimizer = optimizer(learning_rate)
        # optimizer = optimizer(params.get("learning_rate", learning_rate))
        train_op = optimizer.minimize(
            loss=average_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op)

    # In the evaluation mode we will calculate evaluation metrics.
    assert mode == tf.estimator.ModeKeys.EVAL

    # Calculate root mean squared error
    rmse = tf.metrics.root_mean_squared_error(labels, predictions)

    # Add the rmse to collection of evaluation metrics.
    eval_metrics = {"rmse": rmse}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        # Report sum of error for compatibility with pre-made estimators.
        loss=total_loss,
        eval_metric_ops=eval_metrics)


def main(argv):
    """ Builds, Trians, and evaluates the model. """
    assert len(argv) == 1

    # laod data from local disk.
    (x_train,
     y_train), (x_dev, y_dev), (x_test,
                                y_test),(series_mean,series_max,series_min)= import_tanmiao.load_normalized_data(
                                    'orig_day_full_X.xlsx')

    # Build the training input_fn
    def input_train(features=x_train, labels=y_train, batch_size=128):
        """ An input function for training """
        # convert the input to a Dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shffling with a buffer larger than the data set ensures
        # that the examples are will mixed.
        dataset = dataset.shuffle(4000).batch(
            batch_size).repeat().make_one_shot_iterator().get_next()

        return dataset

    # Build the validation input_fn
    def input_dev(features=x_dev, labels=y_dev, batch_size=128):
        # Convert the input to a Dataset
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
        tf.feature_column.numeric_column("X6")
        # tf.feature_column.numeric_column("X7")
    ]

    # Build a custom Estimator, using the model_fn
    # 'params' is passed through the 'model_fn'
    model = tf.estimator.Estimator(
        model_fn=my_dnn_regression_fn,
        model_dir=model_path,
        params={
            'feature_columns': feature_columns,
            'learning_rate': 0.1,
            'decay_steps':500,
            'decay_rate':0.98,
            'optimizer': tf.train.AdamOptimizer,
            'hidden_units': [20, 20, 20, 20]
        })


    train_pred_input_fn = tf.estimator.inputs.pandas_input_fn(
        x_train, shuffle=False)

    space=[
        Real(10**-5,10**0,'log-uniform',name='learning_rate'),
        Real(0.1,0.8,'log-uniform',name='drop_rates')
    ]

    def objective(**paramss):
        model = tf.estimator.Estimator(
            model_fn=my_dnn_regression_fn,
            model_dir=model_path,
            params={
                'feature_columns': feature_columns,
                'learning_rate': paramss.get('learning_rate'),
                'decay_steps': 1,
                'decay_rate': 1,
                'optimizer': tf.train.AdamOptimizer,
                'hidden_units': [20, 20, 20, 20],
                'drop_rates': paramss.get('drop_rates')
            })

        # Train the model
        model.train(input_fn=input_train, steps=STEPS)
        pred = model.predict(input_fn=train_pred_input_fn)
        train_predictions = np.array(list(p['predictions'] for p in pred))
        train_predictions = train_predictions.reshape(np.array(y_train).shape)
        y_train = np.array(y_train)
        return np.sum(np.abs(y_train - train_predictions))

    res_gp = gp_minimize(objective,space,n_calls=50,random_state=0)

    print("minmun MAE=%.4f"%res_gp.fun)



if __name__ == '__main__':
    # The estimator periodically generates "INFO" logs; make this logs visable
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
