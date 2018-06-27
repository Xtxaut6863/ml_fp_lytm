import pandas as pd
import numpy as np
import tensorflow as tf
import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1, os.path.pardir))
data_path = par_path_1+'\\data\\'
print(10 * '-' + ' Current Path: {}'.format(current_path))
print(10 * '-' + ' Parent Path: {}'.format(par_path_1))
print(10 * '-' + ' Grandpa Path: {}'.format(par_path_2))
print(10 * '-' + ' Data Path: {}'.format(data_path))

def csv_gen_sample(
    source_file,
    train_file,
    dev_file,
    test_file,
    seed=None,
    sample_frac=0.888888888888888889,
    test_size=541):
    # Download the full original data set.
    # full_data_set = pd.read_excel(par_path_1+'\\data\\vmd_imf10.xlsx').drop("TM", axis=1)
    full_data_set = pd.read_excel(data_path + source_file).drop("TM", axis=1)
    # Delete the rows with unkonws
    full_data_set.dropna()
    serise_max = full_data_set.max()
    series_min = full_data_set.min()
    full_norm_set = 2 * (full_data_set - series_min) / (serise_max - series_min) - 1
    # Get the length of this series
    series_len = len(full_norm_set)

    # Get the training and developing set
    # train_dev_set = full_norm_set[0:train_dev_len]
    train_dev_set = full_norm_set[0:(series_len - test_size)]

    # Get the test set
    # Shuffle the data
    np.random.seed(seed)
    # split the data into train/developing subsets
    x_train = train_dev_set.sample(frac=sample_frac, random_state=seed)
    x_dev = train_dev_set.drop(x_train.index)
    train_df = train_dev_set.sample(frac=sample_frac, random_state=seed)
    dev_df = train_dev_set.drop(train_df.index)
    test_df = full_norm_set[(series_len - test_size):series_len]
    assert (train_df['X1'].size + dev_df['X1'].size +test_df['X1'].size) == series_len
    train_df.to_csv(data_path+train_file)
    dev_df.to_csv(data_path+dev_file)
    test_df.to_csv(data_path+test_file)


def csv_gen_unsample(source_file,
                   train_file,
                   dev_file,
                   test_file,
                   dev_test_size=541):
    # Download the full original data set.
    # full_data_set = pd.read_excel(par_path_1+'\\data\\vmd_imf10.xlsx').drop("TM", axis=1)
    full_data_set = pd.read_excel(data_path + source_file).drop("TM", axis=1)
    # Delete the rows with unkonws
    full_data_set.dropna()
    serise_max = full_data_set.max()
    series_min = full_data_set.min()
    full_norm_set = 2 * (full_data_set - series_min) / (serise_max - series_min) - 1
    # Get the length of this series
    series_len = len(full_norm_set)

    # Get the training and developing set
    # train_dev_set = full_norm_set[0:train_dev_len]
    train_df = full_norm_set[0:(series_len-dev_test_size-dev_test_size)]
    dev_df = full_norm_set[(series_len-dev_test_size-dev_test_size):(series_len-dev_test_size)]
    test_df = full_norm_set[(series_len-dev_test_size):series_len]
    assert (train_df['X1'].size+dev_df['X1'].size+test_df['X1'].size)==series_len
    train_df.to_csv(data_path + train_file)
    dev_df.to_csv(data_path + dev_file)
    test_df.to_csv(data_path + test_file)


def main():

    """ csv_gen_sample(
        'orig_day_full_X.xlsx',
        'sample_orig_train.csv',
        'sample_orig_dev.csv',
        'unsample_orig_test.csv') """

    csv_gen_unsample(
        'orig_day_full_X.xlsx',
        'lstm_unsample_orig_train.csv',
        'lstm_umsample_orig_dev.csv',
        'lstm_unsample_orig_test.csv')




if __name__ == '__main__':
    main()