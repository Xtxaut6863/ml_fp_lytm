import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1, os.path.pardir))
sys.path.append(par_path_1)
from custom_estimator import my_dnn_regression_fn
sys.path.append(par_path_1 + '\\test\\')
from load_data import load_normalized_data
from dump_data import dump_train_dev_test_to_excel
from plot_utils import plot_normreconvert_relation
from plot_utils import plot_normreconvert_pred
from plot_utils import plot_rela_pred
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from metrics import NSEC
import tensorflow as tf
import numpy as np
import pandas as pd


def main(argv):
    """ Predict based on the trained model and specfic checkpoints. """
    assert len(argv) == 1

    # laod data from local disk.
    (x_train_dev, y_train_dev), (x_train, y_train), (x_dev, y_dev), (
        x_test, y_test), (series_max, series_min) = load_normalized_data(
            # "orig_day_full_X.xlsx", # Do original prediction
            "vmd_imf10.xlsx", # Do vmd IMFs prediction
             seed=123)

    # create feature colums
    feature_columns = [
        tf.feature_column.numeric_column("X1"),
        tf.feature_column.numeric_column("X2"),
        tf.feature_column.numeric_column("X3"),
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

    # recovery the model, and set the dropout rate to 0.0
    model_path = current_path + '/models/imf10/'

    # current_model = 'DNNRegressor_Hidden_Units[7, 13]'  # orig
    # current_model = 'DNNRegressor_Hidden_Units[5, 8]'  # imf1
    # current_model = 'DNNRegressor_Hidden_Units[3]'  # imf2
    # current_model = 'DNNRegressor_Hidden_Units[9, 12]'  # imf3
    # current_model = 'DNNRegressor_Hidden_Units[6, 10]'  # imf4
    # current_model = 'DNNRegressor_Hidden_Units[5, 11]'  # imf5
    # current_model = 'DNNRegressor_Hidden_Units[4, 7]'  # imf6
    # current_model = 'DNNRegressor_Hidden_Units[11, 12]'  # imf7
    # current_model = 'DNNRegressor_Hidden_Units[4]'  # imf8
    # current_model = 'DNNRegressor_Hidden_Units[13, 12]'  # imf9
    current_model = 'DNNRegressor_Hidden_Units[3]'  # imf10
    model_dir = model_path + current_model + '/'

    # model = tf.estimator.Estimator(
    #     model_fn=my_dnn_regression_fn,
    #     model_dir=model_dir,
    #     params={
    #         'feature_columns': feature_columns,
    #         # NOTE: Set the hidden units for predictions
    #         'hidden_units': [7],
    #         'drop_rates': [0.0]
    #     },
    # )

    model = tf.estimator.DNNRegressor(
        # hidden_units=[7, 13], # orig
        # hidden_units=[5, 8], # imf1
        # hidden_units=[3], # imf2
        # hidden_units=[9,12], # imf3
        # hidden_units=[6,10], # imf4
        # hidden_units=[5,11], # imf5
        # hidden_units=[4], # imf6
        # hidden_units=[11,12], # imf7
        # hidden_units=[13,12], # imf9
        hidden_units=[3], # imf10
        feature_columns=feature_columns,
        model_dir=model_dir,
    )

    train_pred_input_fn = tf.estimator.inputs.pandas_input_fn(
        x_train, shuffle=False)
    dev_pred_input_fn = tf.estimator.inputs.pandas_input_fn(
        x_dev, shuffle=False)
    test_pred_input_fn = tf.estimator.inputs.pandas_input_fn(
        x_test, shuffle=False)

    # Use the specific file to predict
    # checkpoint_path = model_dir + 'model.ckpt-7200' #orig
    # checkpoint_path = model_dir + 'model.ckpt-29600' #imf1
    # checkpoint_path = model_dir + 'model.ckpt-50600' #imf2
    # checkpoint_path = model_dir + 'model.ckpt-74900' #imf3
    # checkpoint_path = model_dir + 'model.ckpt-30000' #imf4
    # checkpoint_path = model_dir + 'model.ckpt-73100' #imf5
    # checkpoint_path = model_dir + 'model.ckpt-81700' #imf6
    # checkpoint_path = model_dir + 'model.ckpt-13200' #imf7
    # checkpoint_path = model_dir + 'model.ckpt-32800' #imf8
    # checkpoint_path = model_dir + 'model.ckpt-11700' #imf9
    checkpoint_path = model_dir + 'model.ckpt-45600' #imf10

    # predict the training set by specfic checkpoint
    train_pred_results = model.predict(
        input_fn=train_pred_input_fn, checkpoint_path=checkpoint_path)
    # predict the developing set
    dev_pred_results = model.predict(
        input_fn=dev_pred_input_fn, checkpoint_path=checkpoint_path)
    # predict the testing set.
    test_pred_results = model.predict(
        input_fn=test_pred_input_fn, checkpoint_path=checkpoint_path)

    # Convert generator to numpy array
    train_predictions = np.array(
        list(p['predictions'] for p in train_pred_results))
    dev_predictions = np.array(
        list(p['predictions'] for p in dev_pred_results))
    test_predictions = np.array(
        list(p['predictions'] for p in test_pred_results))

    # reshape the prediction to y shape.
    train_predictions = train_predictions.reshape(np.array(y_train).shape)
    dev_predictions = dev_predictions.reshape(np.array(y_dev).shape)
    test_predictions = test_predictions.reshape(np.array(y_test).shape)

    # Renormalize the records and predictions
    y_train = np.multiply(
        y_train + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
    train_predictions = np.multiply(train_predictions + 1, series_max["Y"] -
                                    series_min["Y"]) / 2 + series_min["Y"]
    y_dev = np.multiply(
        y_dev + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
    dev_predictions = np.multiply(dev_predictions + 1, series_max["Y"] -
                                  series_min["Y"]) / 2 + series_min["Y"]
    y_test = np.multiply(
        y_test + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
    test_predictions = np.multiply(test_predictions + 1, series_max["Y"] -
                                   series_min["Y"]) / 2 + series_min["Y"]

    # compute R square
    r2_train = r2_score(y_train, train_predictions)
    r2_dev = r2_score(y_dev, dev_predictions)
    r2_test = r2_score(y_test, test_predictions)

    # compute MSE
    mse_train = mean_squared_error(y_train, train_predictions)
    mse_dev = mean_squared_error(y_dev, dev_predictions)
    mse_test = mean_squared_error(y_test, test_predictions)

    # compute MAE
    mae_train = mean_absolute_error(y_train, train_predictions)
    mae_dev = mean_absolute_error(y_dev, dev_predictions)
    mae_test = mean_absolute_error(y_test, test_predictions)

    # compute MAPE
    mape_train = np.true_divide(
        np.sum(np.abs(np.true_divide(
            (y_train - train_predictions), y_train))), y_train.size) * 100
    mape_dev = np.true_divide(
        np.sum(np.abs(np.true_divide(
            (y_dev - dev_predictions), y_dev))), y_dev.size) * 100
    mape_test = np.true_divide(
        np.sum(np.abs(np.true_divide(
            (y_test - test_predictions), y_test))), y_test.size) * 100

    #
    print('r2_score_train = {:.10f}'.format(r2_train))
    print('r2_score_dev = {:.10f}'.format(r2_dev))

    dump_train_dev_test_to_excel(
        path=model_path + current_model + '.xlsx',
        y_train=y_train,
        train_pred=train_predictions,
        r2_train=r2_train,
        mse_train=mse_train,
        mae_train=mae_train,
        mape_train=mape_train,
        y_dev=y_dev,
        dev_pred=dev_predictions,
        r2_dev=r2_dev,
        mse_dev=mse_dev,
        mae_dev=mae_dev,
        mape_dev=mape_dev,
        y_test=y_test,
        test_pred=test_predictions,
        r2_test=r2_test,
        mse_test=mse_test,
        mae_test=mae_test,
        mape_test=mape_test)

    plot_rela_pred(
        y_train,
        train_predictions,
        series_max,
        series_min,
        fig_savepath=model_path + current_model + '_train_pred.tif')

    plot_rela_pred(
        y_dev,
        dev_predictions,
        series_max,
        series_min,
        fig_savepath=model_path + current_model + "_dev_pred.tif")

    plot_rela_pred(
        y_test,
        test_predictions,
        series_max,
        series_min,
        fig_savepath=model_path + current_model + "_test_pred.tif")



if __name__ == '__main__':
    # The estimator periodically generates "INFO" logs; make this logs visable
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
