import pandas as pd
import numpy as np
import tensorflow as tf
from tfrecorder import TFrecorder
import os

current_path = os.path.dirname(os.path.abspath(__file__))
# par_path = os.path.abspath(os.path.join(current_path, os.path.pardir))
data_path = current_path+'\\data\\'

def write_tfrecord(path,data,df):
    num_example_per_file = 1024
    num_so_far = 0
    writer = tf.python_io.TFRecordWriter('%s%s_%s'%(path,num_so_far,num_example_per_file))
    tfr = TFrecorder()
    for i in np.arange(data['Y'].size):
        features = {}
        tfr.feature_writer(df.iloc[0],data['Y'][i],features)
        tfr.feature_writer(df.iloc[1], data['X1'][i], features)
        tfr.feature_writer(df.iloc[2], data['X2'][i], features)
        tfr.feature_writer(df.iloc[3], data['X3'][i], features)
        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
        if i%num_example_per_file==0 and i!=0:
            writer.close()
            num_so_far = i
            num_so_far = tf.python_io.TFRecordWriter('%s%s_%s.tf_record'%(path,num_so_far,i+num_example_per_file))
            print('saved %s%s_%s.tfrecord' % (path, num_so_far,i + num_example_per_file))
        writer.close()

def main():
    # Download the full original data set.
    full_data_set = pd.read_excel('F:/ml_fp_lytm/data/vmd_imf1.xlsx').drop( "TM", axis=1)
    # Delete the rows with unkonws
    full_data_set.dropna()
    series_max = full_data_set.max()
    series_min = full_data_set.min()
    full_norm_set = 2 * (full_data_set - series_min) / (series_max - series_min) - 1
    # Get the length of this series
    series_len = len(full_norm_set)
    # Get the length of training and developing set
    train_dev_len = int(round(series_len * 0.9))
    # Get the training and developing set
    train_dev_set = full_norm_set[0:train_dev_len]

    # Get the test set
    test_df = full_norm_set[series_len - 541:series_len]
    # Shuffle the data
    seed = None
    np.random.seed(seed)
    # split the data into train/developing subsets
    train_df = train_dev_set.sample(frac=0.8888889, random_state=seed)
    dev_df = train_dev_set.drop(train_df.index)
    df = pd.DataFrame(
        {'name':['Y','X1','X2','X3'],
        'type':['float32','float32','float32','float32'],
        'shape':[(),(),(),()],
        'isbyte':[False,False,False,False],
        "length_type":['fixed','fixed','fixed','fixed'],
         "default":[np.NaN,np.NaN,np.NaN,np.NaN]
         }
         )

    data_info_path = data_path+'vmd_imf1\\data_info.csv'
    df.to_csv(data_info_path,index=False)
    write_tfrecord(data_path+'vmd_imf1\\train\\',train_df,df)


if __name__ == '__main__':
    main()