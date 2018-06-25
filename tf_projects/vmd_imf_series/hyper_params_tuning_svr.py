import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR,NuSVR
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence


from load_data import load_normalized_data

if __name__ == '__main__':
    (x_train_dev,
     y_train_dev), (x_train, y_train), (x_dev, y_dev), (x_test, y_test), (
         series_max, series_min) = load_normalized_data("vmd_imf4.xlsx")


    reg = SVR(tol=1e-8)

    space = [
        Real(0.1, 25, name='C'),
        Real(10**-10, 10**0, name='epsilon'),
        Real(10**-10, 10**0, name='gamma'),
    ]

    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        return -np.mean(cross_val_score(reg,x_train_dev,y_train_dev,cv=6,n_jobs=-1,scoring='neg_mean_absolute_error'))
    res_gp = gp_minimize(objective,space,n_calls=50,random_state=0,verbose=True)

    print('Best score=%.4f'%res_gp.fun)

    print(""" Best parameters:
     -C = %.8f
     -epsilon = %.8f
     -gamma = %.8f
     """%(res_gp.x[0],res_gp.x[1],res_gp.x[2]))

    plot_convergence(res_gp)

    # Result:
    # Best score=0.0020
    # Best parameters:
    #  -C = 0.749735
    #  -epsilon = 0.0000100