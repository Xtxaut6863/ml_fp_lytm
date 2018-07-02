import pandas as pd
import numpy as np

import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path,os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1,os.path.pardir))
data_path = par_path_1+'\\data\\'

print(10*'-'+' Current Path: {}'.format(current_path))
print(10*'-'+' Parent Path: {}'.format(par_path_1))
print(10*'-'+' Grandpa Path: {}'.format(par_path_2))
print(10 * '-' + ' Data Path: {}'.format(data_path))


def gen_data(
    path,
    columns,
    lags,
    train_csv,
    dev_csv,
    test_csv,
    test_len=None,
    train_frac=0.8888888888888889,
    seed=None,
    sampling=True):
    """ 
    Generate learning data for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -path: The sourcve data file path for generate the learning data.
        -columns: the columns name for read the source data by pandas.
        -lags: The lags for autoregression.
        -train_csv: The CSV file to restore the training set in data path.
        -dev_csv: The CSV file to restore the developing set in the data path.
        -test_csv: The CSV file to restore the testing set in the data path. 
        -test_len: The length of Test set.
        -train_frac: The fraction of training set of training and developing set.
        -seed: The seed for sampling.
        -sampling: Decide wether or not sampling.
    return:
        (series_min,series_max): The min and max value of each series
    """
    dataframe = pd.read_excel(path)[columns]
    dataframe.dropna()
    nparr = np.array(dataframe)
    full_data_set = pd.DataFrame()
    for i in range(lags):
        x = pd.DataFrame(nparr[i:dataframe.size - (lags - i)],columns=['X' + str(i + 1)])['X' + str(i + 1)]
        full_data_set = pd.DataFrame(pd.concat([full_data_set, x], axis=1))

    label = pd.DataFrame(nparr[lags:], columns=['Y'])['Y']
    full_data_set = pd.DataFrame(pd.concat([full_data_set, label], axis=1))
    # Get the max and min value of each series
    serise_max = full_data_set.max()
    series_min = full_data_set.min()

    print(series_min)
    print(serise_max)
    # NOrmalize each series to the range between -1 and 1
    full_norm_set = 2 * (full_data_set - series_min) / (serise_max - series_min) - 1

    # full_norm_set = full_data_set
    # Get the length of this series
    series_len = len(full_norm_set)

    # Get the training and developing set
    train_dev_set = full_norm_set[0:(series_len-test_len)]

    # Get the testing set.
    test_df = full_norm_set[(series_len-test_len):series_len]

    if sampling:
        # sampling
        np.random.seed(seed)
        train_df = train_dev_set.sample(frac=train_frac,random_state=seed)
        dev_df = train_dev_set.drop(train_df.index)
    else:
        train_df = full_norm_set[0:(series_len - test_len - test_len)]
        dev_df = full_norm_set[(series_len - test_len - test_len):(series_len - test_len)]

    assert (train_df['X1'].size + dev_df['X1'].size +test_df['X1'].size) == series_len

    train_df.to_csv(data_path + train_csv)
    dev_df.to_csv(data_path + dev_csv)
    test_df.to_csv(data_path + test_csv)
    return (series_min,serise_max)




if __name__ == '__main__':
    gen_data(
        path=par_path_1+'\\Test.xlsx',
        columns='Test',
        lags=3,
        train_csv='Test_train.csv',
        dev_csv='Test_dev.csv',
        test_csv='Test_test.csv',
        test_len=2,
        sampling=False
        )