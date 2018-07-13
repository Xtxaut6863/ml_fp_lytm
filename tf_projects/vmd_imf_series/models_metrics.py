from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1, os.path.pardir))
print(10 * '-' + ' Current Path: {}'.format(current_path))
print(10 * '-' + ' Parent Path: {}'.format(par_path_1))
print(10 * '-' + ' Grandpa Path: {}'.format(par_path_2))

# Compute metrics for F-Summary !!!!!!!!!!!
data = pd.read_excel(current_path+'\\models\\finally.xlsx')
# print(data)

mse_a_arma = mean_squared_error(data['records'],data['A-ARMA'])
mse_a_gbr = mean_squared_error(data['records'], data['A-GBR'])
mse_a_svr = mean_squared_error(data['records'], data['A-SVR'])
mse_a_dnn = mean_squared_error(data['records'], data['A-DNN'])

r2_a_arma = r2_score(data['records'], data['A-ARMA'])
r2_a_gbr = r2_score(data['records'], data['A-GBR'])
r2_a_svr = r2_score(data['records'], data['A-SVR'])
r2_a_dnn = r2_score(data['records'], data['A-DNN'])

mae_a_arma = mean_absolute_error(data['records'], data['A-ARMA'])
mae_a_gbr = mean_absolute_error(data['records'], data['A-GBR'])
mae_a_svr = mean_absolute_error(data['records'], data['A-SVR'])
mae_a_dnn = mean_absolute_error(data['records'], data['A-DNN'])

mape_a_arma = np.true_divide(np.sum(np.abs(np.true_divide((data['records'] - data['A-ARMA']),data['records']))), data['records'].size) * 100
mape_a_gbr = np.true_divide(np.sum(np.abs(np.true_divide((data['records'] - data['A-GBR']),data['records']))), data['records'].size) * 100
mape_a_svr = np.true_divide(np.sum(np.abs(np.true_divide((data['records'] - data['A-SVR']),data['records']))), data['records'].size) * 100
mape_a_dnn = np.true_divide(np.sum(np.abs(np.true_divide((data['records'] - data['A-DNN']),data['records']))), data['records'].size) * 100


mse_a_arma  = pd.DataFrame([mse_a_arma], columns=['mse_a_arma'])['mse_a_arma']
mse_a_gbr   = pd.DataFrame([mse_a_gbr], columns=['mse_a_gbr'])['mse_a_gbr']
mse_a_svr   = pd.DataFrame([mse_a_svr], columns=['mse_a_svr'])['mse_a_svr']
mse_a_dnn   = pd.DataFrame([mse_a_dnn], columns=['mse_a_dnn'])['mse_a_dnn']

r2_a_arma   = pd.DataFrame([r2_a_arma], columns=['r2_a_arma'])['r2_a_arma']
r2_a_gbr       = pd.DataFrame([r2_a_gbr], columns=['r2_a_gbr'])['r2_a_gbr']
r2_a_svr    = pd.DataFrame([r2_a_svr], columns=['r2_a_svr'])['r2_a_svr']
r2_a_dnn    = pd.DataFrame([r2_a_dnn], columns=['r2_a_dnn'])['r2_a_dnn']

mae_a_arma  = pd.DataFrame([mae_a_arma], columns=['mae_a_arma'])['mae_a_arma']
mae_a_gbr   = pd.DataFrame([mae_a_gbr], columns=['mae_a_gbr'])['mae_a_gbr']
mae_a_svr   = pd.DataFrame([mae_a_svr], columns=['mae_a_svr'])['mae_a_svr']
mae_a_dnn   = pd.DataFrame([mae_a_dnn], columns=['mae_a_dnn'])['mae_a_dnn']

mape_a_arma     = pd.DataFrame([mape_a_arma], columns=['mape_a_arma'])['mape_a_arma']
mape_a_gbr  = pd.DataFrame([mape_a_gbr], columns=['mape_a_gbr'])['mape_a_gbr']
mape_a_svr  = pd.DataFrame([mape_a_svr], columns=['mape_a_svr'])['mape_a_svr']
mape_a_dnn  = pd.DataFrame([mape_a_dnn], columns=['mape_a_dnn'])['mape_a_dnn']

results = pd.DataFrame(
    pd.concat(
        [
            mse_a_arma ,
            mse_a_gbr  ,
            mse_a_svr  ,
            mse_a_dnn  ,
            r2_a_arma  ,
            r2_a_gbr   ,
            r2_a_svr   ,
            r2_a_dnn   ,
            mae_a_arma ,
            mae_a_gbr  ,
            mae_a_svr  ,
            mae_a_dnn  ,
            mape_a_arma,
            mape_a_gbr ,
            mape_a_svr ,
            mape_a_dnn ,
        ],
        axis=1))
writer = pd.ExcelWriter(current_path+'\\models\\ensemble_metrics.xlsx')
results.to_excel(writer, sheet_name='Sheet1')
writer.close()






