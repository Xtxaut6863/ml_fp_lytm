from hyper_params_tuning import my_dnn_regression_fn
import sys
sys.path.append(
    'F:/ml_fp_lytm/tf_projects/test/')
from import_tanmiao import load_normalized_data
from dump_data import dump_train_dev_test_to_excel
from plot_util import plot_normreconvert_relation
from plot_util import plot_normreconvert_pred
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from metrics import NSEC

import tensorflow as tf
import numpy as np


def main(argv):
    """ Predict based on the trained model and specfic checkpoints. """
    assert len(argv) == 1

    # laod data from local disk.
    (x_train_dev, y_train_dev), (x_train, y_train), (x_dev, y_dev), (
        x_test, y_test), (series_mean, series_max,
                          series_min) = load_normalized_data(
                              "orig_day_full_imf2.xlsx", seed=123)

    # create feature colums
    feature_columns = [
        tf.feature_column.numeric_column("X1"),
        tf.feature_column.numeric_column("X2"),
        tf.feature_column.numeric_column("X3"),
        tf.feature_column.numeric_column("X4"),
        tf.feature_column.numeric_column("X5"),
        tf.feature_column.numeric_column("X6"),
        tf.feature_column.numeric_column("X7"),
        tf.feature_column.numeric_column("X8"),
        tf.feature_column.numeric_column("X9"),
        tf.feature_column.numeric_column("X10"),
        tf.feature_column.numeric_column("X11"),
        tf.feature_column.numeric_column("X12"),
        tf.feature_column.numeric_column("X13"),
        tf.feature_column.numeric_column("X14"),
        tf.feature_column.numeric_column("X15"),
        tf.feature_column.numeric_column("X16"),
        tf.feature_column.numeric_column("X17"),
        tf.feature_column.numeric_column("X18"),
        tf.feature_column.numeric_column("X19"),
    ]

    # recovery the model, and set the dropout rate to 0.0
    model_path = 'F:/ml_fp_lytm/tf_projects/imf2_series/models/'
    current_model = 'lr0.01_ds1000_dr0.98_hu[5]_bs512_drop[0.0]'
    model_dir = model_path + current_model + '/'
    model = tf.estimator.Estimator(
        model_fn=my_dnn_regression_fn,
        model_dir=model_dir,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [5],
            'drop_rates': [0.0]
        },
    )

    train_pred_input_fn = tf.estimator.inputs.pandas_input_fn(
        x_train, shuffle=False)
    dev_pred_input_fn = tf.estimator.inputs.pandas_input_fn(
        x_dev, shuffle=False)
    test_pred_input_fn = tf.estimator.inputs.pandas_input_fn(
        x_test, shuffle=False)

    # Use the specific file to predict
    checkpoint_path = model_dir + 'model.ckpt-23100'

    # predict the training set
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
    y_train = np.multiply(y_train,
                          series_max["Y"] - series_min["Y"]) + series_mean["Y"]
    train_predictions = np.multiply(train_predictions, series_max["Y"] -
                                    series_min["Y"]) + series_mean["Y"]
    y_dev = np.multiply(y_dev,
                        series_max["Y"] - series_min["Y"]) + series_mean["Y"]
    dev_predictions = np.multiply(
        dev_predictions, series_max["Y"] - series_min["Y"]) + series_mean["Y"]
    y_test = np.multiply(y_test,
                         series_max["Y"] - series_min["Y"]) + series_mean["Y"]
    test_predictions = np.multiply(
        test_predictions, series_max["Y"] - series_min["Y"]) + series_mean["Y"]


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

    #
    print('r2_score_train = {:.10f}'.format(r2_train))
    print('r2_score_dev = {:.10f}'.format(r2_dev))

    dump_train_dev_test_to_excel(
        path=
        'F:/ml_fp_lytm/tf_projects/imf2_series/models/'
        + current_model + '.xlsx',
        y_train=y_train,
        train_pred=train_predictions,
        r2_train=r2_train,
        mse_train=mse_train,
        mae_train=mae_train,

        y_dev=y_dev,
        dev_pred=dev_predictions,
        r2_dev=r2_dev,
        mse_dev=mse_dev,
        mae_dev=mae_dev,

        y_test=y_test,
        test_pred=test_predictions,
        r2_test=r2_test,
        mse_test=mse_test,
        mae_test=mae_test,
    )

    # print(test_predictions)

    # plot the predicted line
    plot_normreconvert_pred(
        y_train,
        train_predictions,
        series_mean,
        series_max,
        series_min,
        fig_savepath=model_path + current_model + '_train_pred.png')

    plot_normreconvert_relation(
        y_train,
        train_predictions,
        series_mean,
        series_max,
        series_min,
        fig_savepath=model_path + current_model + "_train_rela.png")

    plot_normreconvert_pred(
        y_dev,
        dev_predictions,
        series_mean,
        series_max,
        series_min,
        fig_savepath=model_path + current_model + "_dev_pred.png")

    # plot the relationship between the records and predcitions
    plot_normreconvert_relation(
        y_dev,
        dev_predictions,
        series_mean,
        series_max,
        series_min,
        fig_savepath=model_path + current_model + "_dev_rela.png")

    plot_normreconvert_pred(
        y_test,
        test_predictions,
        series_mean,
        series_max,
        series_min,
        fig_savepath=model_path + current_model + "_test_pred.png")

    # plot the relationship between the records and predcitions
    plot_normreconvert_relation(
        y_test,
        test_predictions,
        series_mean,
        series_max,
        series_min,
        fig_savepath=model_path + current_model + "_test_rela.png")


if __name__ == '__main__':
    # The estimator periodically generates "INFO" logs; make this logs visable
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
