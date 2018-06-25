""" A dataset loader for flow data of TanMiao station, which located in LaoYu """
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import collections
data_path = "F:/ml_fp_lytm/data/"

defaults = collections.OrderedDict(
    [("Y", [0.0]), ("X1", [0.0]), ("X2", [0.0]), ("X3", [0.0]), ("X4", [0.0]),
     ("X5", [0.0]), ("X6", [0.0]), ("X7", [0.0])])  # pyformat: disable


def dataset(y_name="Y", train_fraction=0.7):
    """ 
    Load the day flow data set of TanMiao station in LaoYu
    as a (train,dev) pair of 'Dataset' 
    Each dataset generates (features,label) pairs.

    Args:
        y_name: The name of column to be use as the label.
        train_fraction: A float, the fraction of data to use for training.
        the reminder will be used for validation
    Returns:
        A (train,dev) pair of 'Dataset'
    """

    # Define how the lines of the file should be parsed
    def decode_line(line):
        """ convert a csv line into a (feature_dict,label) pair. """
        # Decode the line to a tuple of items based the types of csv_header.values()
        items = tf.decode_csv(line, list(defaults.values()))

        # Convert the key and items to a dict.
        pairs = zip(defaults.keys(), items)
        feature_dict = dict(pairs)

        # Remove the label from the feature_dict
        label = feature_dict.pop(y_name)

        return feature_dict, label

    def in_training_set(line):
        """ Returns a boolean tensir, true if the line is in the training set. """
        # If you randomly split the dataset you won't get the same split in both
        # seeeions if you stop and restart training later. Also a simple
        # random split won't work with a dataset that too big to '.cache()' as
        # we were ding here
        num_buckets = 1000000
        bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
        # use the hash bucket id as a random number that's deterministic per example
        return bucket_id < int(train_fraction * num_buckets)

    def in_test_set(line):
        """ Returns a boolean tensor, true if the line is in the developing set """
        # Item not in the training set are in the developing set.
        # This line must use '~' instead of 'not' because 'not' only works on python
        # boolean but we are dealing with symbolic tensors.
        return ~in_training_set(line)

    base_dataset = (
        tf.data
        # Get the line from the file
        .TextLineDataset(data_path + "logtrans_day_train_dev.csv"))

    train =(base_dataset
        # Take only the training set lines
        .filter(in_training_set)
        # Decode each line into a (feature_dict,lable) pair
        .map(decode_line)
        # Cache data so you only decode the file once
        .cache()
    )

    # Do the same for test-set
    test = (base_dataset.filter(in_test_set).cache().map(decode_line))

    return train, test


def raw_dataframe(logtrans):
    """ Load the tanmiao data as pandas DataFrame """
    # Download and cache the data
    # path = _get_tanmiao()

    # load it into a pandas dataframe
    if logtrans:
        df_train_dev = pd.read_excel(
            data_path + "logtrans_day_train_dev.xlsx").drop(
                "TM", axis=1)
        df_test = pd.read_excel(data_path + "logtrans_day_test.xlsx").drop(
            "TM", axis=1)
    else:
        df_train_dev = pd.read_excel(
            data_path + "orig_day_train_dev.xlsx").drop(
                "TM", axis=1)
        df_test = pd.read_excel(data_path + "orig_day_test.xlsx").drop(
            "TM", axis=1)

    return df_train_dev, df_test


def load_data(y_name="Y", train_fraction=0.7, seed=None, logtrans=False):
    """ Get the day flow of tanmiao station
    Args:
        y_name: the column to return as label.
        train_fraction: the fraction of the dataset to used for training.
        seed: the random seed to use when shufflling the data. 'None'
        generate a unique shuffle every run

    Returns:
        a pair of pairs where the first pair is the training data
        and the second is the developing data for evaluating different models
     """

    #  load the raw data columns
    data_train_dev, data_test = raw_dataframe(logtrans)

    # delect rows with unknows
    data_train_dev = data_train_dev.dropna()
    data_test = data_test.dropna()

    # Shuffle the data
    np.random.seed(seed)

    # split the data into train/developing subsets
    x_train = data_train_dev.sample(frac=train_fraction, random_state=seed)
    x_dev = data_train_dev.drop(x_train.index)

    # Extract the label from the features dataframe
    y_train = x_train.pop(y_name)
    y_dev = x_dev.pop(y_name)

    x_test = data_test
    y_test = x_test.pop(y_name)

    return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)


def load_normalized_data(current_path,
                         y_name="Y",
                         seed=None,
                         train_dev_fraction=0.9,
                         train_fraction=0.89):
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
    full_data_set = pd.read_excel(data_path + current_path).drop("TM", axis=1)

    # Delete the rows with unkonws
    full_data_set.dropna()

    series_mean = full_data_set.mean()
    serise_max = full_data_set.max()
    series_min = full_data_set.min()

    full_norm_set = (full_data_set - series_mean) / (serise_max - series_min)

    # Get the length of this series
    series_len = len(full_norm_set)

    # Get the length of training and developing set
    train_dev_len = int(round(series_len * train_dev_fraction))

    # Get the training and developing set
    train_dev_set = full_norm_set[0:train_dev_len]

    y_train_dev = train_dev_set['Y']
    x_train_dev = train_dev_set.drop('Y', axis=1)

    # Get the test set
    test_set = full_norm_set[train_dev_len:series_len]

    # Shuffle the data
    np.random.seed(seed)

    # split the data into train/developing subsets
    x_train = train_dev_set.sample(frac=train_fraction, random_state=seed)
    x_dev = train_dev_set.drop(x_train.index)

    # Extract the label from the features dataframe
    y_train = x_train.pop(y_name)
    y_dev = x_dev.pop(y_name)

    x_test = test_set
    y_test = x_test.pop(y_name)

    return (x_train_dev,
            y_train_dev), (x_train, y_train), (x_dev,
                                               y_dev), (x_test,
                                                        y_test), (series_mean,
                                                                  serise_max,
                                                                  series_min)


def load_logtransformed_data(train_dev_path,
                             test_path,
                             y_name="Y",
                             train_fraction=0.7,
                             seed=None):
    """ Get the day flow of tanmiao station
    Args:
        y_name: the column to return as label.
        train_fraction: the fraction of the dataset to used for training.
        seed: the random seed to use when shufflling the data. 'None'
        generate a unique shuffle every run

    Returns:
        a pair of pairs where the first pair is the training data
        and the second is the developing data for evaluating different models
     """

    #  load the raw data columns
    data_train_dev = pd.read_excel(data_path + train_dev_path).drop(
        "TM", axis=1)
    data_test = pd.read_excel(data_path + test_path).drop("TM", axis=1)

    # delect rows with unknows
    data_train_dev = data_train_dev.dropna()
    data_test = data_test.dropna()

    # Shuffle the data
    np.random.seed(seed)

    # split the data into train/developing subsets
    x_train = data_train_dev.sample(frac=train_fraction, random_state=seed)
    x_dev = data_train_dev.drop(x_train.index)

    # Extract the label from the features dataframe
    y_train = x_train.pop(y_name)
    y_dev = x_dev.pop(y_name)

    x_test = data_test
    y_test = x_test.pop(y_name)

    return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)
