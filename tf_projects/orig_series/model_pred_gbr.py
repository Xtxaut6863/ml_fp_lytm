import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import sys
sys.path.append('F:/ml_fp_lytm/tf_projects/test/')
from import_tanmiao import load_normalized_data

if __name__ == '__main__':
    # load data
    (x_train_dev,y_train_dev),(x_train, y_train), (x_dev, y_dev), (x_test, y_test), (
        series_mean, series_max,
        series_min) = load_normalized_data("orig_day_full_X.xlsx")

    GBR = GradientBoostingRegressor(
        learning_rate=0.101054,
        max_depth=3,
        max_features=4,
        min_samples_split=69,
        min_samples_leaf=72
        )
    ESVR = SVR(C=0.749735, epsilon=0.000010)

    test_pred_svr = ESVR.fit(x_test, y_test).predict(x_test)
    test_pred_gbr = GBR.fit(x_test,y_test).predict(x_test)
    plt.figure(figsize=(16,9))
    plt.plot(range(y_test.size), y_test, color='blue', label='records')
    plt.plot(
        range(y_test.size),
        test_pred_gbr,
        '--',
        color='black',
        label='GBR')
    plt.plot(range(y_test.size), test_pred_svr,'--', color='orange', label='SVR')
    plt.legend()
    plt.show()
