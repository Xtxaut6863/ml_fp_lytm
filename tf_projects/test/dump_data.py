import pandas as pd
import numpy as np


def dump_train_dev_to_excel(
        path,
        y_train=None,
        train_pred=None,
        y_dev=None,
        dev_pred=None,
        y_test=None,
        test_pred=None,
):
    writer = pd.ExcelWriter(path)
    # convert the test_pred numpy array into Dataframe series
    # y_test = pd.DataFrame(test_pred, columns=['y_test'])['y_test']
    test_pred = pd.DataFrame(test_pred, columns=['test_pred'])['test_pred']

    # y_train = pd.DataFrame(y_train, columns=['y_train'])['y_train']
    train_pred = pd.DataFrame(train_pred, columns=['train_pred'])['train_pred']

    # y_dev = pd.DataFrame(y_dev, columns=['y_dev'])['y_dev']
    dev_pred = pd.DataFrame(dev_pred, columns=['dev_pred'])['dev_pred']
    results = pd.DataFrame(
        pd.concat(
            [y_test, test_pred, y_train, train_pred, y_dev, dev_pred], axis=1))
    results.to_excel(writer, sheet_name='Sheet1')


""" 
y_aa = np.array([1,2,3,4,5,6,7,8,7,9,1,0,11,12,15,14,17])
aa = pd.DataFrame([1,2,3,4,5,6,7,8,9],columns=['aa'])['aa']
y_bb = np.array([11,12,13,14,15,16])
bb = pd.DataFrame(['a','b','c','d','e','f','g','h'],columns=['bb'])['bb']
dump_to_excel('F:/ml_fp_lytm/tf_projects/test/test.xlsx',aa,y_aa,bb,y_bb) 
"""


def dump_train_dev_test_to_excel(
        path,
        y_train=None,
        train_pred=None,
        r2_train=None,
        mse_train=None,
        mae_train=None,
        mape_train=None,
        y_dev=None,
        dev_pred=None,
        r2_dev=None,
        mse_dev=None,
        mae_dev=None,
        mape_dev=None,
        y_test=None,
        test_pred=None,
        r2_test=None,
        mse_test=None,
        mae_test=None,
        mape_test=None
):
    """ 
    Dump training and developing records and predictions as well as r square to excel.
    Args:
        path: The local disk path to dump data into.
        y_train: train records with Dataframe type.
        train_pred: train predictions with numpy array type.
        r2_train: R square value for train records and predictions, type float.
        y_dev: developing records with Dataframe type.
        dev_pred: developing predictions with numpy array type.
        r2_dev: R square value for developing records and predictions, type float.
        y_test: testing records with Dataframe type.
        test_pred: testing predictions with numpy array type.
        r2_test: R square value for testing records and predictions, type float.
    """
    writer = pd.ExcelWriter(path)

    index_train = pd.Index(np.linspace(1, y_train.size, y_train.size))
    index_dev = pd.Index(np.linspace(1, y_dev.size, y_dev.size))
    index_test = pd.Index(np.linspace(1, y_test.size, y_test.size))

    # convert the train_pred numpy array into Dataframe series
    y_train = pd.DataFrame(list(y_train), index=index_train, columns=['y_train'])['y_train']
    train_pred = pd.DataFrame(data=train_pred, index=index_train,columns=['train_pred'])['train_pred']
    r2_train = pd.DataFrame([r2_train], columns=['r2_train'])['r2_train']
    mse_train = pd.DataFrame([mse_train], columns=['mse_train'])['mse_train']
    mae_train = pd.DataFrame([mae_train], columns=['mae_train'])['mae_train']
    mape_train = pd.DataFrame([mape_train], columns=['mape_train'])['mape_train']
    # nsec_train = pd.DataFrame([nsec_train],columns=['nsec_train'])['nsec_train']

    # y_dev = pd.DataFrame(y_dev, columns=['y_dev'])['y_dev']
    y_dev = pd.DataFrame(list(y_dev), index=index_dev, columns=['y_dev'])['y_dev']
    dev_pred = pd.DataFrame(dev_pred, index=index_dev, columns=['dev_pred'])['dev_pred']
    r2_dev = pd.DataFrame([r2_dev], columns=['r2_dev'])['r2_dev']
    mse_dev = pd.DataFrame([mse_dev], columns=['mse_dev'])['mse_dev']
    mae_dev = pd.DataFrame([mae_dev], columns=['mae_dev'])['mae_dev']
    mape_dev = pd.DataFrame([mape_dev], columns=['mape_dev'])['mape_dev']
    # nsec_dev = pd.DataFrame([nsec_dev], columns=['nsec_dev'])['nsec_dev']

    y_test = pd.DataFrame(list(y_test), index=index_test, columns=['y_test'])['y_test']
    test_pred = pd.DataFrame(test_pred, index=index_test, columns=['test_pred'])['test_pred']
    r2_test = pd.DataFrame([r2_test], columns=['r2_test'])['r2_test']
    mse_test = pd.DataFrame([mse_test], columns=['mse_test'])['mse_test']
    mae_test = pd.DataFrame([mae_test], columns=['mae_test'])['mae_test']
    mape_test = pd.DataFrame([mape_test], columns=['mape_test'])['mape_test']
    # nsec_test = pd.DataFrame([nsec_test], columns=['nsec_test'])['nsec_test']

    results = pd.DataFrame(
        pd.concat(
            [
                y_train,
                train_pred,
                r2_train,
                mse_train,
                mae_train,
                mape_train,
                # nsec_train,
                y_dev,
                dev_pred,
                r2_dev,
                mse_dev,
                mae_dev,
                mape_dev,
                # nsec_dev,
                y_test,
                test_pred,
                r2_test,
                mse_test,
                mae_test,
                mape_test,
                # nsec_test,
            ],
            axis=1))
    results.to_excel(writer, sheet_name='Sheet1')
