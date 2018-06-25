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

STEPS = 1000
model_path = "F:/ml_fp_lytm/tf_projects/test/models/temp2.4/"


def my_dnn_regression_fn(features, labels, mode, params):
    """ 
    A model function implementing DNN regression for a custom Estimator.
    Each layer having a dropout probility of 0.1
    """

    # Extract the input intp  a dense layer, according to feature_columns.
    top = tf.feature_column.input_layer(features, params['feature_columns'])

    # Iterate over the "hidden_units" list of layer size, default is [20].
    for units in params.get('hidden_units', [20]):
        # Add a hidden layer, densely connected on top of previous layer.
        for drop in params.get('drop_rates',[0.5]):
            top = tf.layers.dense(inputs=top, units=units, activation=tf.nn.relu)
            top = tf.layers.dropout(top,rate=0.1)

    # Connect a liner output layer on top.
    output_layer = tf.layers.dense(inputs=top, units=1)

    output_layer = tf.cast(output_layer,tf.float64)
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

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get("optimizer", tf.train.AdamOptimizer)
        optimizer = optimizer(params.get("learning_rate", None))
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
     y_train), (x_dev, y_dev), (x_test,y_test) = import_tanmiao.load_data(logtrans=True)

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
        tf.feature_column.numeric_column("X6"),
        tf.feature_column.numeric_column("X7")
    ]

    # Build a custom Estimator, using the model_fn
    # 'params' is passed through the 'model_fn'
    model = tf.estimator.Estimator(
        model_fn=my_dnn_regression_fn,
        model_dir=model_path,
        params={
            'feature_columns': feature_columns,
            'learning_rate': 0.001,
            'optimizer': tf.train.AdamOptimizer,
            'hidden_units': [20, 20, 20, 20],
            'drop_rates':[0.5,0.5,0.5,0.5]
        })


    # Train the model
    model.train(input_fn=input_train, steps=STEPS)

    # Evaluate how the model performs on a data it has not yet seen.
    eval_result = model.evaluate(input_fn=input_dev, steps=STEPS)

    # The evaluation returns a python dictionary. The 'average loss' key is
    # hold the Mean Square Error (MSE)
    average_loss = eval_result['rmse']
    # train_average_loss = train_eval_result['rmse']

    # Convert MSE to root mean square error (RMSE)
    print("\n" + 80 * "*")
    print("\nRMS error for the validation set: {:.2f}".format(
    average_loss))
    # print("\nRMS error for the validation set: {:.2f}".format(
    #     train_average_loss))
    print()

    # test the model's performance based on test set
    test_pred_input_fn = tf.estimator.inputs.pandas_input_fn(
        x_test, shuffle=False)

    # The result of 'PREDICT' mode is a python generator
    test_pred_results = model.predict(input_fn=test_pred_input_fn)

    # print(list(test_pred_results))

    # Convert generator to numpy array
    test_predictions_list = []
    for dictionary in list(test_pred_results):
        # print(dictionary['predictions'])
        test_predictions_list.append(dictionary['predictions'])
    test_predictions = np.array(test_predictions_list)

    r2 = r2_score(y_test,test_predictions)
    print('r2_score = '+str(r2))

    # print(test_predictions)

    # plot the predicted line
    plot_util.plot_pred(
        y_test,
        test_predictions,
        fig_savepath=model_path + "test_pred_2.png")

    # plot the relationship between the records and predcitions
    plot_util.plot_relation(
        y_test,
        test_predictions,
        fig_savepath=model_path + "test_rela_2.png")


if __name__ == '__main__':
    # The estimator periodically generates "INFO" logs; make this logs visable
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
