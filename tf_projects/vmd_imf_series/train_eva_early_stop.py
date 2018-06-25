import tensorflow as tf
import numpy as np
from tensorflow import data
import shutil
import math
from datetime import datetime
from tensorflow.python.feature_column import feature_column
print(tf.__version__)
import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1, os.path.pardir))
data_path = par_path_2 + '\\data\\'
model_path = current_path + '\\models\\imf1\\'
print('---------Current Path: {}'.format(current_path))
print('---------Parent Path: {}'.format(par_path_1))
print('---------Grandpa Path: {}'.format(par_path_2))
print('---------Data path: {}'.format(data_path))
print('---------Model Path: {}'.format(model_path))
""" 
    Steps to use the TF estimator (Train_And_Evaluate) APIs
        1. Define dataset metadata
        2. Define data input function to read data from .tfrecords files + feature processing
        3. Create TF feature columns based on metadata + extended feature columns
        4. Define an estimator (LinearCombinedDNNRegressor) with the required feature columns (Wide/deep) & parameters
        5. Run an experiment using estimator train_and_evaluate function to train, evaluate, and export the model
        6. Evaluate the model using testing data
        7. Perform prediction & serving the exported model 
"""



TRAIN_DATA_FILES_PATTERN = par_path_2 + '\\data\\sample_vmdimf1_train.tfrecords'
VALID_DATA_FILES_PATTERN = par_path_2 + '\\data\\sample_vmdimf1_dev.tfrecords'
TEST_DATA_FILES_PATTERN = par_path_2 + '\\data\\unsample_vmdimf1_test.tfrecords'

RESUME_TRAINING = False
PROCESS_FEATURES = False
EXTEND_FEATURE_COLUMNS = False
MULTI_THREADING = True
"""
    1. Define dataset metadata
        tf.example feature names and defaults
        Numeric and categorical feature names
        Target feature name
        Unused features
"""
HEADER = [
    'key',
    'Y',
    'X1',
    'X2',
    'X3',
]
HEADER_DEFAULTS = [[0], [0.0], [0.0], [0.0], [0.0]]
NUMERIC_FEATURE_NAMES = ['X1', 'X2', 'X3']
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {}
CATEGORICAL_FEATURE_NAMES = list(
    CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys())
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
TARGET_NAME = 'Y'
UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME})

print("Header: {}".format(HEADER))
print("Numeric Features: {}".format(NUMERIC_FEATURE_NAMES))
print("Categorical Features: {}".format(CATEGORICAL_FEATURE_NAMES))
print("Target: {}".format(TARGET_NAME))
print("Unused Features: {}".format(UNUSED_FEATURE_NAMES))
"""
    2. Define Data Input Function
        Input .tfrecords files name pattern
        Use TF Dataset APIs to read and process the data
        Parse tf.examples to feature tensors
        Apply feature processing
        Return (feature,target) tensors
"""


# A. Parsing and preprocessing logic
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


# b. Data pipeline input function
def tfrecods_input_fn(files_name_pattern,
                      mode=tf.estimator.ModeKeys.EVAL,
                      num_epochs=None,
                      batch_size=200):

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
    dataset = data.TFRecordDataset(filenames=file_names)

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


# Define Feature Columns
# The input numeric columns are assumed to be normalized (or have the same scale).
# Otherwise, a normalizer_fn, along with the normlisation params (mean,stdv,or min , max)
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


feature_columns = get_feature_columns(
    tf.contrib.training.HParams(num_buckets=5, embedding_size=3))
print("Feature Columns: {}".format(feature_columns))


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


feature_columns = get_feature_columns(
    tf.contrib.training.HParams(num_buckets=5, embedding_size=3))
print("Feature Columns: {}".format(feature_columns))


# 4.Define an Estimator Creation Function
# a. Get wide and deep feature columns
#       dense columns = numeric columns + embedding columns
#       categorical columns = vocabolary list columns + bucketized columns
#       sparse columns =  hashed categorical columns + crossed columns
#       deep columns = dense columns+indicator columns
#       wide columns =  categorical columns + sparse columns
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


# b. Define the DNNLinearCombinedRegressor (replaced with DNN regressor)
def create_estimator(run_config, hparams, print_desc=False):

    wide_feature_columns, deep_feature_columns = get_wide_deep_columns()

    # estimator = tf.estimator.DNNLinearCombinedRegressor(
    #     dnn_feature_columns=deep_feature_columns,
    #     linear_feature_columns=wide_feature_columns,
    #     dnn_hidden_units=hparams.hidden_units,
    #     dnn_optimizer=tf.train.AdamOptimizer(),
    #     dnn_activation_fn=tf.nn.elu,
    #     dnn_dropout=hparams.dropout_prob,
    #     config=run_config)

    estimator = tf.estimator.DNNRegressor(
        feature_columns=deep_feature_columns,
        hidden_units=hparams.hidden_units,
        optimizer=tf.train.AdamOptimizer(),
        activation_fn=tf.nn.elu,
        dropout=hparams.dropout_prob,
        config=run_config)

    if print_desc:
        print("")
        print("*Estimator Type:")
        print("================")
        print(type(estimator))
        print("")
        print("*deep columns:")
        print("==============")
        print(deep_feature_columns)
        print("")
        print("wide columns:")
        print("=============")
        print(wide_feature_columns)
        print("")

    return estimator


TRAIN_SIZE = 4332
NUM_EPOCHS = 1000
BATCH_SIZE = 500
EVAL_AFTER_SEC = 15
TOTAL_STEPS = (TRAIN_SIZE / BATCH_SIZE) * NUM_EPOCHS

hparams = tf.contrib.training.HParams(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    hidden_units=[3],#3:12
    num_buckets=6,
    embedding_size=3,
    max_steps=TOTAL_STEPS,
    dropout_prob=0.001)
MODEL_NAME = 'numEpochs' + str(hparams.num_epochs) + '_batchSize' + str(
    hparams.batch_size) + '_hiddenUnits' + str(
        hparams.hidden_units) + '_maxSteps' + str(
            hparams.max_steps) + '_dropProb' + str(hparams.dropout_prob)
model_dir = current_path + '\\models\\imf1\\{}'.format(MODEL_NAME)

run_config = tf.estimator.RunConfig(
    tf_random_seed=19830610, model_dir=model_dir)

print(hparams)
print("Model Directory:", run_config.model_dir)
print("")
print("Dataset Size:", TRAIN_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Steps per Epoch:", TRAIN_SIZE / BATCH_SIZE)
print("Total Steps:", TOTAL_STEPS)
print("That is 1 evaluation step after each", EVAL_AFTER_SEC,
      " training seconds")


# b. Define Serving Function
def csv_serving_input_fn():

    SERVING_HEADER = ['X1', 'X2', 'X3']
    SERVING_HEADER_DEFAULTS = [[0.0], [0.0], [0.0]]

    rows_string_tensor = tf.placeholder(
        dtype=tf.string, shape=[None], name='csv_rows')

    receiver_tensor = {'csv_rows': rows_string_tensor}

    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(
        row_columns, record_defaults=SERVING_HEADER_DEFAULTS)
    features = dict(zip(SERVING_HEADER, columns))

    if PROCESS_FEATURES:
        features = process_features(features)

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


# c. Define an Early Stopping Monitor(Hook)
class EarlyStoppingHook(tf.train.SessionRunHook):
    def __init__(self, early_stopping_rounds=1):
        self._best_loss = None
        self._early_stopping_rounds = early_stopping_rounds
        self._counter = 0

        print("")
        print("*** Early Stopping Hook: - Created")
        print("*** Early Stopping Hook:: Early Stopping Rounds: {}".format(
            self._early_stopping_rounds))
        print("")

    def before_run(self, run_context):

        graph = run_context.session.graph

        #         tensor_name = "dnn/head/weighted_loss/Sum:0" #works!!
        #         loss_tensor = graph.get_tensor_by_name(tensor_name)

        loss_tensor = graph.get_collection(tf.GraphKeys.LOSSES)[1]
        return tf.train.SessionRunArgs(loss_tensor)

    def after_run(self, run_context, run_values):

        last_loss = run_values.results

        print("")
        print("************************")
        print("** Evaluation Monitor - Early Stopping **")
        print("-----------------------------------------")
        print("Early Stopping Hook: Current loss: {}".format(str(last_loss)))
        print("Early Stopping Hook: Best loss: {}".format(
            str(self._best_loss)))

        if self._best_loss is None:
            self._best_loss = last_loss

        elif last_loss > self._best_loss:

            self._counter += 1
            print("Early Stopping Hook: No improvment! Counter: {}".format(
                self._counter))

            if self._counter == self._early_stopping_rounds:

                run_context.request_stop()
                print("Early Stopping Hook: Stop Requested: {}".format(
                    run_context.stop_requested))
        else:

            self._best_loss = last_loss
            self._counter = 0

        print("************************")
        print("")


# d. Define TrainSpec and EvaluSpace
train_spec = tf.estimator.TrainSpec(
    input_fn = lambda: tfrecods_input_fn(
        TRAIN_DATA_FILES_PATTERN,
        mode = tf.estimator.ModeKeys.TRAIN,
        num_epochs=hparams.num_epochs,
        batch_size=hparams.batch_size
    ),
    max_steps=hparams.max_steps,
    hooks=None
)

eval_spec = tf.estimator.EvalSpec(
    input_fn = lambda: tfrecods_input_fn(
        VALID_DATA_FILES_PATTERN,
        mode=tf.estimator.ModeKeys.EVAL,
        num_epochs=1,
        batch_size=hparams.batch_size
    ),
    exporters=[tf.estimator.LatestExporter(
        name="estimate",  # the name of the folder in which the model will be exported to under export
        serving_input_receiver_fn=csv_serving_input_fn,
        exports_to_keep=1,
        as_text=True)],
    steps=None,
    #hooks=[EarlyStoppingHook(15)],
    throttle_secs = EVAL_AFTER_SEC # evalute after each 15 training seconds!
)

# e. Run Experiment via train_and_evaluate
if not RESUME_TRAINING:
    print("Removing previous artifacts...")
    shutil.rmtree(model_dir, ignore_errors=True)
else:
    print("Resuming training...")

tf.logging.set_verbosity(tf.logging.INFO)

time_start = datetime.utcnow()
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................")

estimator = create_estimator(run_config, hparams, True)

tf.estimator.train_and_evaluate(
    estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

time_end = datetime.utcnow()
print(".......................................")
print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
print("")
time_elapsed = time_end - time_start
print("Experiment elapsed time: {} seconds".format(
    time_elapsed.total_seconds()))

# 6. Evaluate the Model
TRAIN_SIZE = 4332
VALID_SIZE = 542
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

estimator = create_estimator(run_config, hparams)

train_results = estimator.evaluate(input_fn=train_input_fn, steps=1)
train_rmse = round(math.sqrt(train_results["average_loss"]), 5)
print()
print(
    "############################################################################################"
)
print("# Train RMSE: {} - {}".format(train_rmse, train_results))
print(
    "############################################################################################"
)

valid_results = estimator.evaluate(input_fn=valid_input_fn, steps=1)
valid_rmse = round(math.sqrt(valid_results["average_loss"]), 5)
print()
print(
    "############################################################################################"
)
print("# Valid RMSE: {} - {}".format(valid_rmse, valid_results))
print(
    "############################################################################################"
)

test_results = estimator.evaluate(input_fn=test_input_fn, steps=1)
test_rmse = round(math.sqrt(test_results["average_loss"]), 5)
print()
print(
    "############################################################################################"
)
print("# Test RMSE: {} - {}".format(test_rmse, test_results))
print(
    "############################################################################################"
)

# 7. Prediction
import itertools
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from load_data import load_normalized_data
import sys
sys.path.append(par_path_1+'\\test\\')
from dump_data import dump_train_dev_test_to_excel
from plot_utils import plot_normreconvert_relation
from plot_utils import plot_normreconvert_pred

""" (x_train_dev, y_train_dev), (x_train, y_train), (x_dev, y_dev), (
    x_test, y_test), (series_max, series_min) = load_normalized_data(
        "vmd_imf1.xlsx", seed=123)
train_input_fn = lambda: tfrecods_input_fn(files_name_pattern=TRAIN_DATA_FILES_PATTERN,mode=tf.estimator.ModeKeys.PREDICT,batch_size= 5)
dev_input_fn = lambda: tfrecods_input_fn(files_name_pattern=VALID_DATA_FILES_PATTERN,mode=tf.estimator.ModeKeys.PREDICT,batch_size= 5)
predict_input_fn = lambda: tfrecods_input_fn(files_name_pattern= TEST_DATA_FILES_PATTERN,mode= tf.estimator.ModeKeys.PREDICT,batch_size= 5)

train_predictions = estimator.predict(input_fn=train_input_fn)
train_pred_values = list(
    map(lambda item: item['predictions'][0],
        list(itertools.islice(train_predictions, 4332))))
dev_predictions = estimator.predict(input_fn=dev_input_fn)
dev_pred_values = list(
    map(lambda item: item['predictions'][0],
        list(itertools.islice(dev_predictions, 542)))) """

predictions = estimator.predict(input_fn=predict_input_fn)
values = list(
    map(lambda item: item["predictions"][0],
        list(itertools.islice(predictions, 541))))
print()
print("Predicted Values: {}".format(values))

# re-normalized the original dataset and predictions
# convert list to numpy
""" train_predictions = np.array(train_pred_values)
dev_predictions = np.array(dev_pred_values)
test_predictions = np.array(values)
y_train = np.multiply(
        y_train + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
train_predictions = np.multiply(train_predictions + 1,series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
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

dump_train_dev_test_to_excel(
    path=model_path + MODEL_NAME + '.xlsx',
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

# plot the predicted line
plot_normreconvert_pred(
    y_train,
    train_predictions,
    series_max,
    series_min,
    fig_savepath=model_path + MODEL_NAME + '_train_pred.png')
plot_normreconvert_relation(
    y_train,
    train_predictions,
    series_max,
    series_min,
    fig_savepath=model_path + MODEL_NAME + "_train_rela.png")
plot_normreconvert_pred(
    y_dev,
    dev_predictions,
    series_max,
    series_min,
    fig_savepath=model_path + MODEL_NAME + "_dev_pred.png")
# plot the relationship between the records and predcitions
plot_normreconvert_relation(
    y_dev,
    dev_predictions,
    series_max,
    series_min,
    fig_savepath=model_path + MODEL_NAME + "_dev_rela.png")
plot_normreconvert_pred(
    y_test,
    test_predictions,
    series_max,
    series_min,
    fig_savepath=model_path + MODEL_NAME + "_test_pred.png")
# plot the relationship between the records and predcitions
plot_normreconvert_relation(
    y_test,
    test_predictions,
    series_max,
    series_min,
    fig_savepath=model_path + MODEL_NAME + "_test_rela.png") """

# Serving via the Saved Model
import os

export_dir = model_dir + "/export/estimate"

saved_model_dir = export_dir + "/" + os.listdir(path=export_dir)[-1]

print(saved_model_dir)

predictor_fn = tf.contrib.predictor.from_saved_model(
    export_dir=saved_model_dir, signature_def_key="predict")

# output = predictor_fn({'csv_rows': ["0.5,1,ax01,bx02", "-0.5,-1,ax02,bx02"]})
output = predictor_fn({'csv_rows': ["0.5,1,2.2", "-0.5,-1,-2.2"]})
print(output)