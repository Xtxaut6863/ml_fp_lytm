import numpy as np
import matplotlib.pyplot as plt


def plot_relation(records, predictions, fig_savepath, log_reconvert=True):
    """ 
    Plot the relations between the records and predictions.
    Args:
        records: the actual measured records.
        predictions: the predictions obtained by model
        fig_savepath: the path where the plot figure will be saved.
        log_reconvert: If ture, reconvert the records and prediction to orignial data by G=10^(z/2.3)-1.
        where 'G' is the original dataset and 'z' is the transformed data by z = 2.3*log_10(G+1). 
    """
    if log_reconvert:
        records = np.power(10, records / 2.3) - 1
        predictions = np.array(list(predictions))
        predictions = np.power(10, predictions / 2.3) - 1
    else:
        predictions = np.array(list(predictions))

    coeff = np.polyfit(predictions, records, 1)
    linear_fit = coeff[0] * predictions + coeff[1]
    ideal_fit = 1 * predictions

    # compare the records and predictions
    plt.figure(num=1, figsize=(16, 9))
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.3)
    # plt.title('The relationship between records and predictions').
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=18)
    plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=18)
    plt.plot(predictions, records, 'o', color='blue', label='', linewidth=1.0)
    plt.plot(predictions, linear_fit, '--', color='red', label='Linear fit')
    plt.plot(predictions, ideal_fit, '-', color='black', label='Ideal fit')
    # plt.text(24,28,'y={:.2f}'.format(coeff[0])+'*x{:.2f}'.format(coeff[1]),fontsize=18)
    # plt.text(26,24,'y=1.0*x',fontsize=18)
    plt.legend(loc='upper left', shadow=True, fontsize=18)
    plt.savefig(fig_savepath, format='tiff', dpi=1000)
    plt.show()


def plot_pred(records, predictions, fig_savepath, log_reconvert=True):
    """
    Plot lines of records and predictions.
    Args:
        records: record data set.
        predictions: prediction data set.
        fig_savepath: the path where the plot figure will be saved.
        log_reconvert: If ture, reconvert the records and prediction to orignial data by G=10^(z/2.3)-1.
        where 'G' is the original dataset and 'z' is the transformed data by z = 2.3*log_10(G+1).
    """

    length = records.size
    t = np.linspace(start=1, stop=length, num=length)

    if log_reconvert:
        records = np.power(10, records / 2.3) - 1
        predictions = np.array(list(predictions))
        predictions = np.power(10, predictions / 2.3) - 1
    else:
        predictions = np.array(list(predictions))

    plt.figure(figsize=(16, 9))
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.3)
    # plt.title('flow prediction based on DNN')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Time(d)', fontsize=18)
    plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=18)
    plt.plot(t, records, '-', color='blue', label='records')
    plt.plot(t, predictions, '--', color='red', label='predictions')
    plt.legend(loc='upper left', shadow=True, fontsize=18)
    plt.savefig(fig_savepath, format='tiff', dpi=1000)
    plt.show()


def plot_normreconvert_relation(records, predictions,series_max,series_min,fig_savepath):
    """ 
    Plot the relations between the records and predictions.
    Args:
        records: the actual measured records.
        predictions: the predictions obtained by model.
        series_mean: Datafram contains mean value of features and labels.
        series_max: Dataframe contains max value of features and labels.
        series_min: Datafram coontains min value of features and labels.
        fig_savepath: the path where the plot figure will be saved.
    """

    # records = np.multiply(records,series_max["Y"]-series_min["Y"])+series_mean["Y"]
    # records = np.multiply(records + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
    # predictions = np.array(list(predictions))
    # predictions = np.multiply(predictions, series_max["Y"] - series_min["Y"]) + series_mean["Y"]
    # predictions = np.multiply(predictions + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]


    coeff = np.polyfit(predictions, records, 1)
    linear_fit = coeff[0] * predictions + coeff[1]
    ideal_fit = 1 * predictions

    # compare the records and predictions
    plt.figure(num=1, figsize=(16, 9))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.3)
    # plt.title('The relationship between records and predictions').
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=18)
    plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=18)
    plt.plot(predictions, records, 'o', color='blue', label='', linewidth=1.0)
    plt.plot(predictions, linear_fit, '--', color='red', label='Linear fit')
    plt.plot(predictions, ideal_fit, '-', color='black', label='Ideal fit')
    # plt.text(24,28,'y={:.2f}'.format(coeff[0])+'*x{:.2f}'.format(coeff[1]),fontsize=18)
    # plt.text(26,24,'y=1.0*x',fontsize=18)
    plt.legend(loc='upper left', shadow=True, fontsize=18)
    plt.savefig(fig_savepath, format='tiff', dpi=1000)
    plt.show()


def plot_normreconvert_pred(records, predictions,series_max,series_min, fig_savepath):
    """
    Plot lines of records and predictions.
    Args:
        records: record data set.
        predictions: prediction data set.
        fig_savepath: the path where the plot figure will be saved.
        log_reconvert: If ture, reconvert the records and prediction to orignial data by G=10^(z/2.3)-1.
        where 'G' is the original dataset and 'z' is the transformed data by z = 2.3*log_10(G+1).
    """

    length = records.size
    t = np.linspace(start=1, stop=length, num=length)

    # records = np.multiply(records,series_max["Y"]-series_min["Y"])+series_mean["Y"]
    # records = np.multiply(records + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
    # predictions = np.array(list(predictions))
    # predictions = np.multiply(predictions + 1, series_max["Y"] -eries_min["Y"]) / 2 + series_min["Y"]

    plt.figure(figsize=(16, 9))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.3)
    # plt.title('flow prediction based on DNN')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Time(d)', fontsize=18)
    plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=18)
    plt.plot(t, records, '-', color='blue', label='records')
    plt.plot(t, predictions, '--', color='red', label='predictions')
    plt.legend(loc='upper left', shadow=True, fontsize=18)
    plt.savefig(fig_savepath, format='tiff', dpi=1000)
    plt.show()

def plot_rela_pred(records, predictions,series_max,series_min, fig_savepath):
    length = records.size
    t = np.linspace(start=1, stop=length, num=length)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time(d)', fontsize=12)
    plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
    plt.plot(t, records, '-', color='blue', label='records')
    plt.plot(t, predictions, '--', color='red', label='predictions')
    plt.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.005,1.2),
        shadow=True,
        fontsize=12)
    plt.subplot(1, 2, 2)
    coeff = np.polyfit(predictions, records, 1)
    linear_fit = coeff[0] * predictions + coeff[1]
    ideal_fit = 1 * predictions
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
    plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
    plt.plot(predictions, records, 'o', color='blue', label='', linewidth=1.0)
    plt.plot(predictions, linear_fit, '--', color='red', label='Linear fit')
    plt.plot(predictions, ideal_fit, '-', color='black', label='Ideal fit')
    plt.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.05,1),
        shadow=True,
        fontsize=12)
    plt.subplots_adjust(left=0.08, bottom=0.15, right=0.92, top=0.85, hspace=0.1, wspace=0.2)
    plt.savefig(fig_savepath, format='tiff', dpi=1000)
    plt.show()