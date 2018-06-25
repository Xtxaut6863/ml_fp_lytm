import scipy.io as sio
import pandas as pd
import numpy as np
import os
import math
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1, os.path.pardir))
data_path = par_path_2 + '\\data\\'
# print(10 * '-' + ' Current Path: {}'.format(current_path))
# print(10 * '-' + ' Parent Path: {}'.format(par_path_1))
# print(10 * '-' + ' Grandpa Path: {}'.format(par_path_2))


def load_normalized_data(current_path,
                         y_name="Y",
                         seed=None,
                         train_dev_fraction=0.9,
                         train_fraction=0.888888888889):
    """ 
    Get daily flow of TanMiao station located in LaoYu.
    Normalize these data by the Equation Q_i_normalized = (Q_i-Q_mean)/(Q_max-Q_min)

    Args:
        Current_path: data path.
        y_name: the column to return as label.
        seed: the random seed to use when shufflling the data. 'None'
        generate a unique shuffle every run.
        train_dev_fraction: The fraction of training and developing set to the full data set.
        train_fraction: The fraction of training set to training and developing set.
    Return:
        (x_train, y_train), (x_dev, y_dev), (x_test, y_test),(series_mean,serise_max,series_min)


    """
    # Download the full original data set.
    data = pd.read_excel(data_path + current_path)
    if (data.keys().contains('TM')):
        full_data_set = data.drop("TM", axis=1)
    else:
        full_data_set = data

    # Delete the rows with unkonws
    full_data_set.dropna()


    series_max = full_data_set.max()
    series_min = full_data_set.min()

    full_norm_set = 2*(full_data_set - series_min) / (series_max - series_min)-1
    # print(type(full_norm_set))
    # writer = pd.ExcelWriter(par_path_1+'\\full_norm_set.xlsx')
    # full_norm_set.to_excel(writer, sheet_name='Sheet1')
    # serise_max.to_excel(writer, sheet_name='Sheet2')
    # series_min.to_excel(writer, sheet_name='Sheet3')
    # Get the length of this series
    series_len = len(full_norm_set)


    # Get the length of training and developing set
    # train_dev_len = int(math.ceil(series_len * train_dev_fraction))

    # Get the training and developing set
    # train_dev_set = full_norm_set[0:train_dev_len]
    train_dev_set = full_norm_set[0:(series_len-541)]

    y_train_dev = train_dev_set['Y']
    x_train_dev = train_dev_set.drop('Y', axis=1)

    # Get the test set
    test_set = full_norm_set[(series_len-541):series_len]

    # Shuffle the data
    np.random.seed(seed)

    # split the data into train/developing subsets
    x_train = train_dev_set.sample(frac=train_fraction, random_state=seed)
    x_dev = train_dev_set.drop(x_train.index)

    # Extract the label from the features dataframe
    y_train = x_train.pop(y_name)
    y_dev = x_dev.pop(y_name)

    # print(test_set)
    x_test = test_set
    y_test = x_test.pop(y_name)
    print(10 * '-' + ' Series length: {}'.format(full_norm_set['X1'].size))
    print(10 * '-' + ' Trian length: {}'.format(y_train.size))
    print(10 * '-' + ' Dev length: {}'.format(y_dev.size))
    print(10 * '-' + ' Test length: {}'.format(y_test.size))



    return (x_train_dev,
            y_train_dev), (x_train, y_train), (x_dev,
                                               y_dev), (x_test,
                                                        y_test), (series_max,
                                                                  series_min)




(x_train_dev, y_train_dev), (x_train, y_train), (x_dev,y_dev), (x_test,y_test), (serise_max,series_min)=load_normalized_data('VMD_IMFS.xlsx')
x_dev_dict = x_dev.to_dict(orient='list')
y_train = pd.Series.to_frame(y_train,name='Y')
print(serise_max)
print(series_min)
# print(x_dev_dict)