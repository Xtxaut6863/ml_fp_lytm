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

data = pd.read_excel(current_path+'\\models\\finally.xlsx')
# print(data)
length = len(data)
t = np.linspace(start=1,stop=length,num=length)
plt.figure(figsize=(16, 9))
plt.subplots_adjust(left=0.05, bottom=0.08, right=0.92, top=0.9, hspace=0.2, wspace=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(
    t,
    data['records'],
    '-',
    color='blue',
    linewidth=1.5,
    label='records')
plt.plot(
    t,
    data['A-GBR'],
    's',
    color='darkgreen',
    markersize=3,
    # linewidth=1.5,
    label='A-GBR')
plt.plot(
    t,
    data['A-SVR'],
    'p',
    color='tomato',
    markersize=3,
    # linewidth=1.5,
    label='A-SVR')
plt.plot(
    t,
    data['A-ARMA'],
    'H',
    color='slategray',
    markersize=3,
    # linewidth=1.5,
    label='A-ARMA')
plt.plot(
    t,
    data['A-DNN'],
    'D',
    color='brown',
    markersize=3,
    # linewidth=1.5,
    label='A-DNN')
plt.plot(
    t,
    data['O-GBR'],
    '>',
    color='forestgreen',
    markersize=3,
    # linewidth=1.5,
    label='O-GBR')
plt.plot(
    t,
    data['O-SVR'],
    '+',
    color='navy',
    markersize=3,
    # linewidth=1.5,
    label='O-SVR')
plt.plot(
    t,
    data['O-ARMA'],
    '^',
    color='darkred',
    markersize=3,
    # linewidth=1.5,
    label='O-ARMA')
plt.plot(
    t,
    data['O-DNN'],
    '*',
    color='indigo',
    markersize=3,
    # linewidth=1.5,
    label='O-DNN')
plt.plot(
    t,
    data['DNN-GBR'],
    '--',
    color='black',
    linewidth=1.5,
    label='DNN-GBR')
plt.plot(
    t,
    data['DNN-SVR'],
    '--',
    color='yellow',
    linewidth=1.5,
    label='DNN-SVR')
plt.plot(
    t,
    data['DNN-ARMA'],
    '--',
    color='green',
    linewidth=1.5,
    label='DNN-ARMA')
plt.plot(
    t,
    data['DNN-DNN'],
    '--',
    color='red',
    linewidth=1.5,
    label='DNN-DNN')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=12)
plt.savefig(current_path+'\\models\\Multi_model_compare.tif', format='tiff', dpi=600)
plt.show()


mse_a_arma = mean_squared_error(data['records'],data['A-ARMA'])
mse_a_gbr = mean_squared_error(data['records'], data['A-GBR'])
mse_a_svr = mean_squared_error(data['records'], data['A-SVR'])
# mse_a_dnn = mean_squared_error(data['records'], data['A-DNN'])

r2_a_arma = r2_score(data['records'], data['A-ARMA'])
r2_a_gbr = r2_score(data['records'], data['A-GBR'])
r2_a_svr = r2_score(data['records'], data['A-SVR'])
# r2_a_dnn = r2_score(data['records'], data['A-DNN'])

mae_a_arma = mean_absolute_error(data['records'], data['A-ARMA'])
mae_a_gbr = mean_absolute_error(data['records'], data['A-GBR'])
mae_a_svr = mean_absolute_error(data['records'], data['A-SVR'])
# mae_a_dnn = mean_absolute_error(data['records'], data['A-DNN'])

mape_a_arma = np.true_divide(np.sum(np.abs(np.true_divide((data['records'] - data['A-ARMA']),data['records']))), data['records'].size) * 100
mape_a_gbr = np.true_divide(np.sum(np.abs(np.true_divide((data['records'] - data['A-GBR']),data['records']))), data['records'].size) * 100
mape_a_svr = np.true_divide(np.sum(np.abs(np.true_divide((data['records'] - data['A-SVR']),data['records']))), data['records'].size) * 100
# mape_a_dnn = np.true_divide(np.sum(np.abs(np.true_divide((data['records'] - data['A-DNN']),data['records']))), data['records'].size) * 100


mse_a_arma  = pd.DataFrame([mse_a_arma], columns=['mse_a_arma'])['mse_a_arma']
mse_a_gbr   = pd.DataFrame([mse_a_gbr], columns=['mse_a_gbr'])['mse_a_gbr']
mse_a_svr   = pd.DataFrame([mse_a_svr], columns=['mse_a_svr'])['mse_a_svr']
# mse_a_dnn   = pd.DataFrame([mse_a_dnn], columns=['mse_a_dnn'])['mse_a_dnn']

r2_a_arma   = pd.DataFrame([r2_a_arma], columns=['r2_a_arma'])['r2_a_arma']
r2_a_gbr       = pd.DataFrame([r2_a_gbr], columns=['r2_a_gbr'])['r2_a_gbr']
r2_a_svr    = pd.DataFrame([r2_a_svr], columns=['r2_a_svr'])['r2_a_svr']
# r2_a_dnn    = pd.DataFrame([r2_a_dnn], columns=['r2_a_dnn'])['r2_a_dnn']

mae_a_arma  = pd.DataFrame([mae_a_arma], columns=['mae_a_arma'])['mae_a_arma']
mae_a_gbr   = pd.DataFrame([mae_a_gbr], columns=['mae_a_gbr'])['mae_a_gbr']
mae_a_svr   = pd.DataFrame([mae_a_svr], columns=['mae_a_svr'])['mae_a_svr']
# mae_a_dnn   = pd.DataFrame([mae_a_dnn], columns=['mae_a_dnn'])['mae_a_dnn']

mape_a_arma     = pd.DataFrame([mape_a_arma], columns=['mape_a_arma'])['mape_a_arma']
mape_a_gbr  = pd.DataFrame([mape_a_gbr], columns=['mape_a_gbr'])['mape_a_gbr']
mape_a_svr  = pd.DataFrame([mape_a_svr], columns=['mape_a_svr'])['mape_a_svr']
# mape_a_dnn  = pd.DataFrame([mape_a_dnn], columns=['mape_a_dnn'])['mape_a_dnn']

results = pd.DataFrame(
    pd.concat(
        [
            mse_a_arma ,
            mse_a_gbr  ,
            mse_a_svr  ,
            # mse_a_dnn  ,
            r2_a_arma  ,
            r2_a_gbr   ,
            r2_a_svr   ,
            # r2_a_dnn   ,
            mae_a_arma ,
            mae_a_gbr  ,
            mae_a_svr  ,
            # mae_a_dnn  ,
            mape_a_arma,
            mape_a_gbr ,
            mape_a_svr ,
            # mape_a_dnn ,
        ],
        axis=1))
writer = pd.ExcelWriter(current_path+'\\models\\ensemble_metrics.xlsx')
results.to_excel(writer, sheet_name='Sheet1')
writer.close()


ideal_fit = 1 * data['A-ARMA']
pp = np.linspace(start=1,stop=60,num=60)
coeff_a_ARMA = np.polyfit(data['A-ARMA'], data['records'], 1)
linear_fit_a_arma = coeff_a_ARMA[0] * data['A-ARMA'] + coeff_a_ARMA[1]

coeff_a_GBR = np.polyfit(data['A-GBR'], data['records'], 1)
linear_fit_a_gbr = coeff_a_GBR[0] * data['A-ARMA'] + coeff_a_GBR[1]

coeff_a_SVR = np.polyfit(data['A-SVR'], data['records'], 1)
linear_fit_a_SVR = coeff_a_SVR[0] * data['A-ARMA'] + coeff_a_SVR[1]

coeff_a_DNN = np.polyfit(data['A-DNN'], data['records'], 1)
linear_fit_a_DNN = coeff_a_DNN[0] * data['A-ARMA'] + coeff_a_DNN[1]

coeff_O_ARMA = np.polyfit(data['O-ARMA'], data['records'], 1)
linear_fit_O_arma = coeff_O_ARMA[0] * data['A-ARMA'] + coeff_O_ARMA[1]

coeff_O_GBR = np.polyfit(data['O-GBR'], data['records'], 1)
linear_fit_O_gbr = coeff_O_GBR[0] * data['A-ARMA'] + coeff_O_GBR[1]

coeff_O_SVR = np.polyfit(data['O-SVR'], data['records'], 1)
linear_fit_O_SVR = coeff_O_SVR[0] * data['A-ARMA'] + coeff_O_SVR[1]

coeff_O_DNN = np.polyfit(data['O-DNN'], data['records'], 1)
linear_fit_O_DNN = coeff_O_DNN[0] * data['A-ARMA'] + coeff_O_DNN[1]

coeff_DNN_ARMA = np.polyfit(data['DNN-ARMA'], data['records'], 1)
linear_fit_DNN_arma = coeff_DNN_ARMA[0] * data['A-ARMA'] + coeff_DNN_ARMA[1]

coeff_DNN_GBR = np.polyfit(data['DNN-GBR'], data['records'], 1)
linear_fit_DNN_gbr = coeff_DNN_GBR[0] * data['A-ARMA'] + coeff_DNN_GBR[1]

coeff_DNN_SVR = np.polyfit(data['DNN-SVR'], data['records'], 1)
linear_fit_DNN_SVR = coeff_DNN_SVR[0] * data['A-ARMA'] + coeff_DNN_SVR[1]

coeff_DNN_DNN = np.polyfit(data['DNN-DNN'], data['records'], 1)
linear_fit_DNN_DNN = coeff_DNN_DNN[0] * data['A-ARMA'] + coeff_DNN_DNN[1]



plt.figure(figsize=(16, 9))
plt.subplots_adjust(
    left=0.05, bottom=0.08, right=0.92, top=0.9, hspace=0.2, wspace=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
# A-ARMA
plt.plot(data['A-ARMA'], data['records'], 'o', color='blue', label='A-ARMA', linewidth=1.0)
plt.plot(data['A-ARMA'], linear_fit_a_arma, '--', color='red', label='Linear fit of A-ARMA')


# A-GBR
plt.plot(data['A-GBR'],data['records'],'*',color='red',label='A-GBR',linewidth=1.0)
plt.plot(data['A-GBR'], linear_fit_a_gbr, '--', color='blueviolet', label='Linear fit of A-GBR')


# A-SVR
plt.plot(data['A-SVR'], data['records'], '+', color='crimson', label='A-SVR', linewidth=1.0)
plt.plot(data['A-SVR'], linear_fit_a_SVR, '--', color='green', label='Linear fit of A-SVR')


# A-DNN
plt.plot(data['A-DNN'],data['records'],'D',color='teal',label='A-DNN',linewidth=1.0)
plt.plot(data['A-DNN'], linear_fit_a_DNN, '--', color='purple', label='Linear fit of A-DNN')

# O-ARMA
plt.plot(
    data['O-ARMA'],
    data['records'],
    'h',
    color='navy',
    label='O-ARMA',
    linewidth=1.0)
plt.plot(
    data['O-ARMA'],
    linear_fit_O_arma,
    '--',
    color='indigo',
    label='Linear fit of O-ARMA')

# O-GBR
plt.plot(
    data['O-GBR'],
    data['records'],
    's',
    color='olive',
    label='O-GBR',
    linewidth=1.0)
plt.plot(
    data['O-GBR'],
    linear_fit_O_gbr,
    '--',
    color='firebrick',
    label='Linear fit of O-GBR')

# O-SVR
plt.plot(
    data['O-SVR'],
    data['records'],
    '4',
    color='g',
    label='O-SVR',
    linewidth=1.0)
plt.plot(
    data['O-SVR'],
    linear_fit_O_SVR,
    '--',
    color='brown',
    label='Linear fit of O-SVR')

# O-DNN
plt.plot(
    data['O-DNN'],
    data['records'],
    'x',
    color='skyblue',
    label='O-DNN',
    linewidth=1.0)
plt.plot(
    data['O-DNN'],
    linear_fit_O_DNN,
    '--',
    color='dimgrey',
    label='Linear fit of O-DNN')

# DNN-ARMA
plt.plot(
    data['DNN-ARMA'],
    data['records'],
    '1',
    color='peru',
    label='DNN-ARMA',
    linewidth=1.0)
plt.plot(
    data['DNN-ARMA'],
    linear_fit_DNN_arma,
    '--',
    color='seagreen',
    label='Linear fit of DNN-ARMA')

# DNN-GBR
plt.plot(
    data['DNN-GBR'],
    data['records'],
    '2',
    color='darkcyan',
    label='DNN-GBR',
    linewidth=1.0)
plt.plot(
    data['DNN-GBR'],
    linear_fit_DNN_gbr,
    '--',
    color='sienna',
    label='Linear fit of DNN-GBR')

# DNN-SVR
plt.plot(
    data['DNN-SVR'],
    data['records'],
    '3',
    color='deepskyblue',
    label='DNN-SVR',
    linewidth=1.0)
plt.plot(
    data['DNN-SVR'],
    linear_fit_DNN_SVR,
    '--',
    color='saddlebrown',
    label='Linear fit of DNN-SVR')

# DNN-DNN
plt.plot(
    data['DNN-DNN'],
    data['records'],
    '^',
    color='fuchsia',
    label='DNN-DNN',
    linewidth=1.0)
plt.plot(
    data['DNN-DNN'],
    linear_fit_DNN_DNN,
    '--',
    color='maroon',
    label='Linear fit of DNN-DNN')


plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=12)
plt.savefig(
    current_path + '\\models\\Multi_model_compare_pred_rela.tif', format='tiff', dpi=600)
plt.show()
