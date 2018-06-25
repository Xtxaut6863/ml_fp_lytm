from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import numpy as np
sys.path.append(
    'F:/ml_fp_lytm/tf_projects/test/')
from import_tanmiao import load_normalized_data
from plot_util import plot_normreconvert_relation
from plot_util import plot_normreconvert_pred
from sklearn.metrics import r2_score

sys.path.append('F:/ml_fp_lytm/tf_projects/')
from custom_estimator import my_dnn_regression_fn

STEPS = 100
model_path = 'F:/ml_fp_lytm/tf_projects/imf2_series/models/'

def main(argv):
    """ Builds, Trians, and evaluates the model. """
    assert len(argv) == 1

    # laod data from local disk.
    (x_train_dev,y_train_dev),(x_train, y_train), (x_dev, y_dev), (x_test, y_test), (
        series_mean, series_max,
        series_min) = load_normalized_data("orig_day_full_imf2.xlsx")

    # Build the training input_fn
    def input_train(features=x_train, labels=y_train, batch_size=512):
        """ An input function for training """
        # convert the input to a Dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shffling with a buffer larger than the data set ensures
        # that the examples are will mixed.
        dataset = dataset.shuffle(4000).batch(
            batch_size).repeat().make_one_shot_iterator().get_next()

        return dataset

    # Build the validation input_fn
    def input_dev(features=x_dev, labels=y_dev, batch_size=512):
        # Convert the input to a Dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffling
        dataset = dataset.shuffle(2000).batch(
            batch_size).make_one_shot_iterator().get_next()

        return dataset

    feature_columns = [
        tf.feature_column.numeric_column("X1"),
        tf.feature_column.numeric_column("X2"),
        tf.feature_column.numeric_column("X3"),
        tf.feature_column.numeric_column("X4"),
        tf.feature_column.numeric_column("X5"),
        tf.feature_column.numeric_column("X6"),
        tf.feature_column.numeric_column("X7"),
        tf.feature_column.numeric_column("X8"),
        tf.feature_column.numeric_column("X9"),
        tf.feature_column.numeric_column("X10"),
        tf.feature_column.numeric_column("X11"),
        tf.feature_column.numeric_column("X12"),
        tf.feature_column.numeric_column("X13"),
        tf.feature_column.numeric_column("X14"),
        tf.feature_column.numeric_column("X15"),
        tf.feature_column.numeric_column("X16"),
        tf.feature_column.numeric_column("X17"),
        tf.feature_column.numeric_column("X18"),
        tf.feature_column.numeric_column("X19"),
    ]

    my_check_point_config = tf.estimator.RunConfig(
        save_checkpoints_steps=50,
        keep_checkpoint_max=1000  #Retain the 50most recent checkpoints
    )

    for learning_rate in [0.01]:
        decay_steps = 1000  # Learning rate decay steps
        decay_rate = 0.98  # Learning rate decay rate
        hidden_units = [3]
        batch_size = 512
        drop_rates = [0.0]
        # construct a hyperparameter str
        hparam_str = 'lr' + str(learning_rate) + '_ds' + str(
            decay_steps) + '_dr' + str(decay_rate) + '_hu' + str(
                hidden_units) + '_bs' + str(batch_size) + '_drop'+str(drop_rates)

        model_dir = model_path+hparam_str

        # Build a custom Estimator, using the model_fn
        # 'params' is passed through the 'model_fn'
        model = tf.estimator.Estimator(
            model_fn=my_dnn_regression_fn,
            model_dir=model_dir,
            params={
                'feature_columns': feature_columns,
                'learning_rate': learning_rate,
                # without learning_rate decay
                'decay_steps': decay_steps,
                'decay_rate': decay_rate,
                'optimizer': tf.train.AdamOptimizer,
                'hidden_units': hidden_units,
                'drop_rates':drop_rates
            },
            config=my_check_point_config)


        """ fig = plt.figure(figsize=(16,9))
        ax221 = plt.subplot(2,2,1)
        ax221.set_title('train predict line')
        ax221.set_xlabel('Time(day)')
        ax221.set_ylabel('flow(' + r'$m^3$' + '/s)')
        ax221.grid()
        ax222 = plt.subplot(2, 2, 2)
        ax222.set_title('train predictions and records scatters')
        ax222.set_xlabel('predictions(' + r'$m^3$' + '/s)')
        ax222.set_ylabel('records(' + r'$m^3$' + '/s)')
        ax222.grid()
        ax223 = plt.subplot(2, 2, 3)
        ax223.set_title('develop predict line')
        ax223.set_xlabel('Time(day)')
        ax223.set_ylabel('flow(' + r'$m^3$' + '/s)')
        ax223.grid()
        ax224 = plt.subplot(2, 2, 4)
        ax224.set_title('develop predictions and records scatters')
        ax224.set_xlabel('predictions(' + r'$m^3$' + '/s)')
        ax224.set_ylabel('records(' + r'$m^3$' + '/s)')
        ax224.grid() """



        for i in range(10):
            # Train the model
            model.train(input_fn=input_train, steps=STEPS)

            # Evaluate how the model performs on a data it has not yet seen.
            eval_result = model.evaluate(input_fn=input_dev, steps=STEPS)

            # The evaluation returns a python dictionary. The 'average loss' key is
            # hold the Mean Square Error (MSE)
            average_loss = eval_result['mse']
            # Convert MSE to root mean square error (RMSE)
            print("\n" + 80 * "*")
            print("\nRMS error for the validation set: {:.8f}".format(
                average_loss))
            print()

            train_pred_input_fn = tf.estimator.inputs.pandas_input_fn(
                x_train, shuffle=False)
            dev_pred_input_fn = tf.estimator.inputs.pandas_input_fn(
                x_dev, shuffle=False)

            # predict the training set
            train_pred_results = model.predict(input_fn=train_pred_input_fn)
            # predict the testing set
            dev_pred_results = model.predict(input_fn=dev_pred_input_fn)

            # Convert generator to numpy array
            train_predictions = np.array(list(p['predictions'] for p in train_pred_results))
            dev_predictions = np.array(list(p['predictions'] for p in dev_pred_results))
            train_predictions = train_predictions.reshape(np.array(y_train).shape)
            dev_predictions = dev_predictions.reshape(np.array(y_dev).shape)

            # Renormalize the records and predictions
            y_train = np.multiply(y_train, series_max["Y"] - series_min["Y"]) + series_mean["Y"]
            train_predictions = np.multiply(train_predictions,series_max["Y"] - series_min["Y"]) + series_mean["Y"]
            y_dev = np.multiply(y_dev, series_max["Y"] - series_min["Y"]) + series_mean["Y"]
            dev_predictions = np.multiply(dev_predictions, series_max["Y"] -series_min["Y"]) + series_mean["Y"]

            # compute R square
            r2_train = r2_score(y_train, train_predictions)
            r2_dev = r2_score(y_dev, dev_predictions)
            print('r2_score_train = {:.10f}'.format(r2_train))
            print('r2_score_dev = {:.10f}'.format(r2_dev))

            """ # time series length
            train_t = np.linspace(1,y_train.size,y_train.size)

            # plot predict line for training set
            ax221.cla()
            ax221.plot(train_t,y_train,label='train records',color='blue')
            ax221.plot(train_t,train_predictions,label='train predictions',color='red')


            # plot scatters for training records and predictions
            ax222.cla()
            coeff = np.polyfit(train_predictions, y_train, 1)
            linear_fit = coeff[0] * train_predictions + coeff[1]
            ideal_fit = 1 * train_predictions
            ax222.plot(train_predictions, y_train, 'o', color='blue')
            ax222.plot(train_predictions, linear_fit, '--', color='red', label='Linear fit')
            ax222.plot(train_predictions, ideal_fit, '-', color='black', label='Ideal fit')

            # plot predict line for develop set
            dev_t = np.linspace(1,y_dev.size,y_dev.size)
            ax223.cla()
            ax223.plot(dev_t,y_dev,label='develop records',color='blue')
            ax223.plot(dev_t,dev_predictions,label='develop predictions',color='red')

            # PLOT scatters for developing records and predictions
            ax224.cla()
            coeff = np.polyfit(dev_predictions, y_dev, 1)
            linear_fit = coeff[0] * dev_predictions + coeff[1]
            ideal_fit = 1 * dev_predictions
            ax224.plot(dev_predictions, y_dev, 'o', color='blue')
            ax224.plot(
                dev_predictions,
                linear_fit,
                '--',
                color='red',
                label='Linear fit')
            ax224.plot(
                dev_predictions,
                ideal_fit,
                '-',
                color='black',
                label='Ideal fit')
            plt.pause(1) """


if __name__ == '__main__':
    # The estimator periodically generates "INFO" logs; make this logs visable
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
