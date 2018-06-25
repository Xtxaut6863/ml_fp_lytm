import matplotlib.pyplot as plt
from sklearn.svm import SVR
from load_data import load_normalized_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import numpy as np
from plot_utils import plot_normreconvert_relation
from plot_utils import plot_normreconvert_pred
from plot_utils import plot_rela_pred
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1, os.path.pardir))
print(10 * '-' + ' Current Path: {}'.format(current_path))
print(10 * '-' + ' Parent Path: {}'.format(par_path_1))
print(10 * '-' + ' Grandpa Path: {}'.format(par_path_2))
sys.path.append(par_path_1 + '\\test\\')
from dump_data import dump_train_dev_test_to_excel

if __name__ == '__main__':

    model_path = current_path + '\\models\\imf10\\'
    data_file = 'vmd_imf10.xlsx'
    print(10*'-'+' model path: {}'.format(model_path))
    print(10*'-'+' data file: {}'.format(data_file))
    # load data
    (x_train_dev, y_train_dev), (x_train, y_train), (x_dev, y_dev), (
        x_test, y_test), (series_max, series_min) = load_normalized_data(
            # "orig_day_full_X.xlsx",
            #  "vmd_imf4.xlsx",
            data_file,
             seed=123)



    """ # Hyper-params for Original
    reg = SVR(
        C=1.68926756, epsilon=0.00002334, gamma=0.93010959)  #Best Score=0.0036 """

    """ # Hyper-params for IMF1
    reg = SVR(
        C=23.96795408, epsilon=0.00000141,
        gamma=0.33998293)  #Best Score=0.0007 """

    """ # Hyper-params for IMF2
    reg = SVR(
        C=23.96795408, epsilon=0.00000141,
        gamma=0.33998293)  #Best Score=0.0010 """

    """ # Hyper-params for IMF3
    reg = SVR(
        C=24.81272376,
        epsilon=0.00066688,
        gamma=0.09719872)  #Best Score=0.0009 """

    """ # Hyper-params for IMF4
    reg = SVR(
        C=23.93107423, epsilon=0.00030246,
        gamma=0.03321255)  #Best Score=0.0009 """

    """ # Hyper-params for IMF5
    reg = SVR(
        C=24.11239783, epsilon=0.00040077,
        gamma=0.10832599)  #Best Score=0.0009 """

    """ # Hyper-params for IMF6
    reg = SVR(
        C=23.93107423, epsilon=0.00030246,
        gamma=0.03321255)  #Best Score=0.0009 """

    """ # Hyper-params for IMF7
    reg = SVR(
        C=25.00000000, epsilon=0.00000000,
        gamma=0.29591667)  #Best Score=0.0013 """

    """ # Hyper-params for IMF8
    reg = SVR(
        C=20.01284460, epsilon=0.00217122,
        gamma=0.20267005)  #Best Score=0.0018 """

    """ # Hyper-params for IMF9
    reg = SVR(
        C=24.01172672, epsilon=0.00158315,
        gamma=0.05230437)  #Best Score=0.0009 """

    # Hyper-params for IMF10
    reg = SVR(
        C=24.65713102, epsilon=0.00029633,
        gamma=0.04384308)  #Best Score=0.0006

    train_predictions = reg.fit(x_train, y_train).predict(x_train)
    dev_predictions = reg.fit(x_dev, y_dev).predict(x_dev)
    test_predictions = reg.fit(x_test, y_test).predict(x_test)

    # Renormalized the records and predictions
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

    dump_train_dev_test_to_excel(
        path=model_path + 'SVR.xlsx',
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

    # print(test_predictions)

    # plot the predicted line
    plot_rela_pred(
        y_train,
        train_predictions,
        series_max,
        series_min,
        fig_savepath=model_path + 'SVR_train_pred.png')

    # plot_normreconvert_relation(
    #     y_train,
    #     train_predictions,
    #     series_max,
    #     series_min,
    #     fig_savepath=model_path + "SVR_train_rela.png")

    plot_rela_pred(
        y_dev,
        dev_predictions,
        series_max,
        series_min,
        fig_savepath=model_path + "SVR_dev_pred.png")

    # plot the relationship between the records and predcitions
    # plot_normreconvert_relation(
    #     y_dev,
    #     dev_predictions,
    #     series_max,
    #     series_min,
    #     fig_savepath=model_path + "SVR__dev_rela.png")

    plot_rela_pred(
        y_test,
        test_predictions,
        series_max,
        series_min,
        fig_savepath=model_path + "SVR_test_pred.png")

    # plot the relationship between the records and predcitions
    # plot_normreconvert_relation(
    #     y_test,
    #     test_predictions,
    #     series_max,
    #     series_min,
    #     fig_savepath=model_path + "SVR_test_rela.png")
