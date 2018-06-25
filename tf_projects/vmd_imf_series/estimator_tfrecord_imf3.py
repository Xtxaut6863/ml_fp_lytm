from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1,os.path.pardir))
data_path = par_path_2 + '\\data\\'
model_path = current_path+'\\models\\imf3\\' #===========Change here==========
print('---------Current Path: {}'.format(current_path))
print('---------Parent Path: {}'.format(par_path_1))
print('---------Grandpa Path: {}'.format(par_path_2))
print('---------Data path: {}'.format(data_path))
print('---------Model Path: {}'.format(model_path))
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

TRAIN_DATA_FILES_PATTERN = par_path_2 + '\\data\\sample_vmdimf3_train.tfrecords'
VALID_DATA_FILES_PATTERN = par_path_2 + '\\data\\sample_vmdimf3_dev.tfrecords'
TEST_DATA_FILES_PATTERN = par_path_2 + '\\data\\unsample_vmdimf3_test.tfrecords'

RESUME_TRAINING = False
PROCESS_FEATURES = False
EXTEND_FEATURE_COLUMNS = False
MULTI_THREADING = True

# 1. Define dataset metadata
#   tf.example feature name and defaults
#   Numeric feature names
#   Target feature name
#   Unused features
HEADER = [
    'key',
    'Y',
    'X1',
    'X2',
    'X3',
    'X4',
    'X5',
    'X6',
    'X7',
    'X8',
    # 'X9',
    # 'X10'
]
HEADER_DEFAULTS = [
    [0],
    [0.0],
    [0.0],  #X1
    [0.0],  #X2
    [0.0],  #X3
    [0.0],  #X4
    [0.0],  #X5
    [0.0],  #X6
    [0.0],  #X7
    [0.0],  #X8
    # [0.0],  #X9
    # [0.0]  #X10
]
NUMERIC_FEATURE_NAMES = [
    'X1',
    'X2',
    'X3',
    'X4',
    'X5',
    'X6',
    'X7',
    'X8',
    # 'X9',
    # 'X10'
]

CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {}
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys())
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
TARGET_NAME = 'Y'
UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME})

print("Header: {}".format(HEADER))
print("Numeric Features: {}".format(NUMERIC_FEATURE_NAMES))
print("Categorical Features: {}".format(CATEGORICAL_FEATURE_NAMES))
print("Target: {}".format(TARGET_NAME))
print("Unused Features: {}".format(UNUSED_FEATURE_NAMES))

# 2.DEfine Data Input Function
#   Input.tfrecords files name pattern
#   Use TF Dataset APIs to read and process the data
#   Parse tf.examples to feature tensors
#   Apply feature processing
#   Return (feature, target) tensors
def parse_tf_example(example_proto):

    feature_spec = {}

    for feature_name in NUMERIC_FEATURE_NAMES:
        feature_spec[feature_name] = tf.FixedLenFeature(
            shape=(1), dtype=tf.float32)

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        feature_spec[feature_name] = tf.FixedLenFeature(
            shape=(1), dtype=tf.string)

    feature_spec[TARGET_NAME] = tf.FixedLenFeature(shape=(1), dtype=tf.float32)

    parsed_features = tf.parse_example(
        serialized=example_proto, features=feature_spec)

    target = parsed_features.pop(TARGET_NAME)

    return parsed_features, target

def process_features(features):

    # example of clipping
    features['x'] = tf.clip_by_value(
        features['x'], clip_value_min=-3, clip_value_max=3)
    features['y'] = tf.clip_by_value(
        features['y'], clip_value_min=-3, clip_value_max=3)

    # example of polynomial expansion
    features["x_2"] = tf.square(features['x'])
    features["y_2"] = tf.square(features['y'])

    # example of nonlinearity
    features["xy"] = features['x'] * features['y']

    # example of custom logic
    features['dist_xy'] = tf.sqrt(
        tf.squared_difference(features['x'], features['y']))
    features["sin_x"] = tf.sin(features['x'])
    features["cos_y"] = tf.sin(features['y'])

    return features

def tfrecods_input_fn(files_name_pattern,
                      mode=tf.estimator.ModeKeys.EVAL,
                      num_epochs=None,
                      batch_size=256):
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    print("")
    print("* data input_fn:")
    print("================")
    print("Input file(s): {}".format(files_name_pattern))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")

    file_names = tf.matching_files(files_name_pattern)
    dataset = tf.data.TFRecordDataset(filenames=file_names)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example))

    if PROCESS_FEATURES:
        dataset = dataset.map(
            lambda features, target: (process_features(features), target))

    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    features, target = iterator.get_next()
    return features, target


features, target = tfrecods_input_fn(files_name_pattern="")
print("Feature read from TFRecords: {}".format(list(features.keys())))
print("Target read from TFRecords: {}".format(target))

def extend_feature_columns(feature_columns, hparams):

    num_buckets = hparams.num_buckets
    embedding_size = hparams.embedding_size

    buckets = np.linspace(-3, 3, num_buckets).tolist()

    alpha_X_beta = tf.feature_column.crossed_column(
        [feature_columns['alpha'], feature_columns['beta']], 4)

    x_bucketized = tf.feature_column.bucketized_column(
        feature_columns['x'], boundaries=buckets)

    y_bucketized = tf.feature_column.bucketized_column(
        feature_columns['y'], boundaries=buckets)

    x_bucketized_X_y_bucketized = tf.feature_column.crossed_column(
        [x_bucketized, y_bucketized], num_buckets**2)

    x_bucketized_X_y_bucketized_embedded = tf.feature_column.embedding_column(
        x_bucketized_X_y_bucketized, dimension=embedding_size)

    feature_columns['alpha_X_beta'] = alpha_X_beta
    feature_columns[
        'x_bucketized_X_y_bucketized'] = x_bucketized_X_y_bucketized
    feature_columns[
        'x_bucketized_X_y_bucketized_embedded'] = x_bucketized_X_y_bucketized_embedded

    return feature_columns

def get_feature_columns(hparams):

    CONSTRUCTED_NUMERIC_FEATURES_NAMES = [
        # 'x_2', 'y_2', 'xy', 'dist_xy', 'sin_x', 'cos_y'
    ]
    all_numeric_feature_names = NUMERIC_FEATURE_NAMES.copy()

    if PROCESS_FEATURES:
        all_numeric_feature_names += CONSTRUCTED_NUMERIC_FEATURES_NAMES

    numeric_columns = {
        feature_name: tf.feature_column.numeric_column(feature_name)
        for feature_name in all_numeric_feature_names
    }

    categorical_column_with_vocabulary = \
        {item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
         for item in CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}

    feature_columns = {}

    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    if categorical_column_with_vocabulary is not None:
        feature_columns.update(categorical_column_with_vocabulary)

    if EXTEND_FEATURE_COLUMNS:
        feature_columns = extend_feature_columns(feature_columns, hparams)

    return feature_columns

feature_columns = get_feature_columns(tf.contrib.training.HParams(num_buckets=5, embedding_size=3))
print("Feature Columns: {}".format(feature_columns))

TRAIN_SIZE = 4329
NUM_EPOCHS = 1000
BATCH_SIZE = 512
EVAL_AFTER_SEC = 15
TOTAL_STEPS = (TRAIN_SIZE / BATCH_SIZE) * NUM_EPOCHS
hparams = tf.contrib.training.HParams(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    # hidden_units=[16, 12, 8],
    num_buckets=6,
    embedding_size=3,
    max_steps=TOTAL_STEPS,
    dropout_prob=0.001)

def get_wide_deep_columns():

    feature_columns = list(get_feature_columns(hparams).values())

    dense_columns = list(
        filter(lambda column: isinstance(column, feature_column._NumericColumn) |
                              isinstance(column, feature_column._EmbeddingColumn),
               feature_columns
        )
    )

    categorical_columns = list(
        filter(lambda column: isinstance(column, feature_column._VocabularyListCategoricalColumn) |
                              isinstance(column, feature_column._BucketizedColumn),
                   feature_columns)
    )

    sparse_columns = list(
        filter(lambda column: isinstance(column,feature_column._HashedCategoricalColumn) |
                              isinstance(column, feature_column._CrossedColumn),
               feature_columns)
    )

    indicator_columns = list(
        map(lambda column: tf.feature_column.indicator_column(column),
            categorical_columns))

    deep_feature_columns = dense_columns + indicator_columns
    wide_feature_columns = categorical_columns + sparse_columns

    return wide_feature_columns, deep_feature_columns

def main(argv):
    """ Build, train, and evaluates the model. """
    assert len(argv) == 1
    # feature_columns = get_feature_columns()
    # print("Feature Columns: {}".format(feature_columns))

    wide_feature_columns, deep_feature_columns = get_wide_deep_columns()
    TRAIN_SIZE = 4329
    VALID_SIZE = 541
    TEST_SIZE = 541
    train_input_fn = lambda: tfrecods_input_fn(files_name_pattern= TRAIN_DATA_FILES_PATTERN,
                                          mode= tf.estimator.ModeKeys.EVAL,
                                          batch_size= TRAIN_SIZE)

    valid_input_fn = lambda: tfrecods_input_fn(files_name_pattern= VALID_DATA_FILES_PATTERN,
                                          mode= tf.estimator.ModeKeys.EVAL,
                                          batch_size= VALID_SIZE)

    test_input_fn = lambda: tfrecods_input_fn(files_name_pattern= TEST_DATA_FILES_PATTERN,
                                          mode= tf.estimator.ModeKeys.EVAL,
                                          batch_size= TEST_SIZE)


    my_check_point_config = tf.estimator.RunConfig(
        # save_checkpoints_steps=50,
        keep_checkpoint_max=1000  #Retain the 50most recent checkpoints
    )
    for learning_rate in [0.01]:
        hidden_units = [7,12]  #4:13
        hparam_str = 'DNNRegressor_Hidden_Units'+str(hidden_units)

        # Set the models path.
        model_dir = model_path + hparam_str

        model = tf.estimator.DNNRegressor(
            hidden_units=hidden_units,
            feature_columns=deep_feature_columns,
            model_dir=model_dir,
            config=my_check_point_config
            )

        STEPS = 100
        for i in range(100):
            # Train the model
            model.train(input_fn=train_input_fn, steps=STEPS)
            # Evaluate how the model performs on a data it has not yet seen.
            eval_result = model.evaluate(input_fn=valid_input_fn, steps=STEPS)

            tf.reset_default_graph()
            # The evaluation returns a python dictionary. The 'average loss' key is
            # hold the Mean Square Error (MSE)
            # average_loss = eval_result['mse']
            # average_loss = eval_result['average_loss']
            # Convert MSE to root mean square error (RMSE)
            # print("\n" + 80 * "*")
            # print("\nMS error for the validation set: {:.8f}".format( average_loss))
            # print()


if __name__ == "__main__":
    # The estimator periodically generates "INFO" logs; make this logs visable
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)