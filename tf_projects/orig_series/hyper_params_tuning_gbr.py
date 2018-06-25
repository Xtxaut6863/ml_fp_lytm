import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from skopt.space import Real,Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

# import data
import sys
sys.path.append('F:/ml_fp_lytm/tf_projects/test/')
from import_tanmiao import load_normalized_data

if __name__ == '__main__':
    # load data
    (x_train_dev,y_train_dev),(x_train, y_train), (x_dev, y_dev), (x_test, y_test), (
        series_mean, series_max,
        series_min) = load_normalized_data("orig_day_full_X.xlsx")

    # Get the feature num
    n_features = x_train_dev.shape[1]

    reg = GradientBoostingRegressor(n_estimators=50,random_state=0)

    # The list hyper-parameters we want
    space = [
        Integer(1,5,name='max_depth'),
        Real(10**-5,10**0,'log-uniform',name='learning_rate'),
        Integer(1,n_features,name='max_features'),
        Integer(2,100,name='min_samples_split'),
        Integer(1,100,name='min_samples_leaf'),
    ]

    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        return -np.mean(cross_val_score(reg,x_train_dev,y_train_dev,cv=6,n_jobs=-1,scoring='neg_mean_absolute_error'))

    res_gp = gp_minimize(objective,space,n_calls=50,random_state=0)

    print('Best score=%.4f'%res_gp.fun)

    print("""Best parameters:
    - max_depth=%d
    - learning_rate=%.6f
    - max_features=%d
    - min_samples_split=%d
    - min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2], res_gp.x[3],
                                res_gp.x[4]))


    plot_convergence(res_gp)

    # Result:
    # Best score=0.0026
    # Best parameters:
    # - max_depth=3
    # - learning_rate=0.101054
    # - max_features=4
    # - min_samples_split=69
    # - min_samples_leaf=72