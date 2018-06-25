import tensorflow as tf
import csv
import os
print(tf.__version__)

current_path = os.path.dirname(os.path.abspath(__file__))
par_path = os.path.abspath(os.path.join(current_path, os.path.pardir))


# train_data_files = [par_path+'\\data\\sample_vmdimf10_train.csv']
# valid_data_files = [par_path+'\\data\\sample_vmdimf10_dev.csv']
# test_data_files = [par_path+'\\data\\unsample_vmdimf10_test.csv']
train_data_files = [par_path+'\\data\\sample_orig_train.csv']
valid_data_files = [par_path+'\\data\\sample_orig_dev.csv']
test_data_files = [par_path+'\\data\\unsample_orig_test.csv']


HEADER = ['key', 'Y',
'X1',
'X2',
'X3',
'X4',
'X5',
'X6',
# 'X7',
# 'X8',
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
    # [0.0],  #X7
    # [0.0],  #X8
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
    # 'X7',
    # 'X8',
    # 'X9',
    # 'X10'
]
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {
    # 'alpha': ['ax01', 'ax02'],
    # 'beta': ['bx01', 'bx02']
}
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys())
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
TARGET_NAME = 'Y'
UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME})

print('Header: {}'.format(HEADER))
print('Numeric features: {}'.format(NUMERIC_FEATURE_NAMES))
print('Categorical Features: {}'.format(CATEGORICAL_FEATURE_NAMES))
print('Target: {}'.format(TARGET_NAME))
print('Unsed Features: {}'.format(UNUSED_FEATURE_NAMES))


def create_csv_iterator(csv_file_path, skip_header):

    with tf.gfile.Open(csv_file_path) as csv_file:
        reader = csv.reader(csv_file)
        if skip_header:  # Skip the header
            next(reader)
        for row in reader:
            yield row


def create_example(row):
    """
    Returns a tensorflow.Example Protocol Buffer object.
    """
    example = tf.train.Example()

    for i in range(len(HEADER)):

        feature_name = HEADER[i]
        feature_value = row[i]

        if feature_name in UNUSED_FEATURE_NAMES:
            continue

        if feature_name in NUMERIC_FEATURE_NAMES:
            example.features.feature[feature_name].float_list.value.extend(
                [float(feature_value)])

        elif feature_name in CATEGORICAL_FEATURE_NAMES:
            example.features.feature[feature_name].bytes_list.value.extend(
                [bytes(feature_value, 'utf-8')])

        elif feature_name in TARGET_NAME:
            example.features.feature[feature_name].float_list.value.extend(
                [float(feature_value)])

    return example


def create_tfrecords_file(input_csv_file):
    """
    Creates a TFRecords file for the given input data and
    example transofmration function
    """
    output_tfrecord_file = input_csv_file.replace("csv", "tfrecords")
    writer = tf.python_io.TFRecordWriter(output_tfrecord_file)

    print("Creating TFRecords file at", output_tfrecord_file, "...")

    for i, row in enumerate(
            create_csv_iterator(input_csv_file, skip_header=True)):

        if len(row) == 0:
            continue

        example = create_example(row)
        content = example.SerializeToString()
        writer.write(content)

    writer.close()

    print("Finish Writing", output_tfrecord_file)


print("Converting Training Data Files")
for input_csv_file in train_data_files:
    create_tfrecords_file(input_csv_file)
print("")

print("Converting Validation Data Files")
for input_csv_file in valid_data_files:
    create_tfrecords_file(input_csv_file)
print("")

print("Converting Test Data Files")
for input_csv_file in test_data_files:
    create_tfrecords_file(input_csv_file)

# example = next(tf.python_io.tf_record_iterator(
#             par_path+'\\data\\sample_vmdimf2_train.tfrecords'
#         ))
# print(tf.train.Example.FromString(example))