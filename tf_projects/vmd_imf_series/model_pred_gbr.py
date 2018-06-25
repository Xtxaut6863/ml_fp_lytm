import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from load_data import load_normalized_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import numpy as np
from plot_utils import plot_normreconvert_relation
from plot_utils import plot_normreconvert_pred
from plot_utils import plot_rela_pred
import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1, os.path.pardir))
print(10 * '-' + ' Current Path: {}'.format(current_path))
print(10 * '-' + ' Parent Path: {}'.format(par_path_1))
print(10 * '-' + ' Grandpa Path: {}'.format(par_path_2))
import sys
sys.path.append(par_path_1+'\\test\\')
from dump_data import dump_train_dev_test_to_excel

if __name__ == '__main__':

    model_path = current_path+'\\models\\imf10\\'
    # data_file = 'orig_day_full_X.xlsx'
    data_file = 'vmd_imf10.xlsx'
    print(10 * '-' + ' model path: {}'.format(model_path))
    print(10 * '-' + ' data file: {}'.format(data_file))


    # load data
    (x_train_dev,
     y_train_dev), (x_train, y_train), (x_dev, y_dev), (x_test, y_test), (
         series_max, series_min) = load_normalized_data(
         data_file,
         seed=123)

    """ # Hyperparameters for orig
    GBR = GradientBoostingRegressor(
        learning_rate=0.564013,
        max_depth=1,
        max_features=6,
        min_samples_split=199,
        min_samples_leaf=8) """

    """ # Hyperparameters for vmd_imf1
    GBR = GradientBoostingRegressor(
        learning_rate=0.268431,
        max_depth=10,
        max_features=3,
        min_samples_split=100,
        min_samples_leaf=1
        ) """

    """ # Hyperparameters for vmd_imf2
    GBR = GradientBoostingRegressor(
        learning_rate=0.259072,
        max_depth=15,
        max_features=3,
        min_samples_split=19,
        min_samples_leaf=2) """

    """ # Hyperparameters for vmd_imf3
    GBR = GradientBoostingRegressor(
        learning_rate=0.214276,
        max_depth=20,
        max_features=8,
        min_samples_split=200,
        min_samples_leaf=2) """

    """ # Hyperparameters for vmd_imf4
    GBR = GradientBoostingRegressor(
        learning_rate=0.741337,
        max_depth=5,
        max_features=10,
        min_samples_split=6,
        min_samples_leaf=1) """

    """ # Hyperparameters for vmd_imf5
    GBR = GradientBoostingRegressor(
        learning_rate=0.741337,
        max_depth=5,
        max_features=8,
        min_samples_split=6,
        min_samples_leaf=1) """

    """ # Hyperparameters for vmd_imf6
    GBR = GradientBoostingRegressor(
        learning_rate=0.772053,
        max_depth=7,
        max_features=6,
        min_samples_split=186,
        min_samples_leaf=2) """

    """ # Hyperparameters for vmd_imf7
    GBR = GradientBoostingRegressor(
        learning_rate=0.741337,
        max_depth=10,
        max_features=4,
        min_samples_split=6,
        min_samples_leaf=1) """

    """ # Hyperparameters for vmd_imf8
    GBR = GradientBoostingRegressor(
        learning_rate=1.000,
        max_depth=50,
        max_features=8,
        min_samples_split=200,
        min_samples_leaf=1) """

    """ # Hyperparameters for vmd_imf9
    GBR = GradientBoostingRegressor(
        learning_rate=0.855922,
        max_depth=45,
        max_features=8,
        min_samples_split=5,
        min_samples_leaf=5) """

    # Hyperparameters for vmd_imf10
    GBR = GradientBoostingRegressor(
        learning_rate=0.741337,
        max_depth=5,
        max_features=3,
        min_samples_split=4,
        min_samples_leaf=1)

    """ # Hyperparameters for ensemble
    GBR = GradientBoostingRegressor(
        learning_rate=0.166467,
        max_depth=15,
        max_features=10,
        min_samples_split=85,
        min_samples_leaf=63) """

    # Do prediction with the opyimal model
    train_predictions = GBR.fit(x_train,y_train).predict(x_train)
    dev_predictions = GBR.fit(x_dev,y_dev).predict(x_dev)
    test_predictions = GBR.fit(x_test,y_test).predict(x_test)

    # Renormalized the records and predictions
    y_train = np.multiply(y_train + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
    train_predictions = np.multiply(train_predictions + 1, series_max["Y"] -series_min["Y"]) / 2 + series_min["Y"]
    y_dev = np.multiply(y_dev + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
    dev_predictions = np.multiply(dev_predictions + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
    y_test = np.multiply( y_test + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
    test_predictions = np.multiply(test_predictions + 1, series_max["Y"] -series_min["Y"]) / 2 + series_min["Y"]

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
        path=model_path + 'GBR.xlsx',
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
        fig_savepath=model_path + 'GBR_train_pred.png')

    # plot_normreconvert_relation(
    #     y_train,
    #     train_predictions,
    #     series_max,
    #     series_min,
    #     fig_savepath=model_path  + "GBR_train_rela.png")

    plot_rela_pred(
        y_dev,
        dev_predictions,
        series_max,
        series_min,
        fig_savepath=model_path + "GBR_dev_pred.png")

    # plot the relationship between the records and predcitions
    # plot_normreconvert_relation(
    #     y_dev,
    #     dev_predictions,
    #     series_max,
    #     series_min,
    #     fig_savepath=model_path  + "GBR__dev_rela.png")

    plot_rela_pred(
        y_test,
        test_predictions,
        series_max,
        series_min,
        fig_savepath=model_path + "GBR_test_pred.png")

    # plot the relationship between the records and predcitions
    # plot_normreconvert_relation(
    #     y_test,
    #     test_predictions,
    #     series_max,
    #     series_min,
    #     fig_savepath=model_path  + "GBR_test_rela.png")