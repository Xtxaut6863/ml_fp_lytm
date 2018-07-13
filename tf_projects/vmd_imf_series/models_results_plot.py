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
plt.figure(figsize=(8, 10))
plt.subplots_adjust(left=0.08, bottom=0.05, right=0.92, top=0.95, hspace=0.4, wspace=0.3)

# GBR
plt.subplot(4, 2, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t,data['records'],'-',color='blue',linewidth=1.5,label='records')
plt.plot(t,data['A-GBR'],'--',color='red', linewidth=1.5,label='A-GBR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

plt.subplot(4, 2, 2)
coeff_a_gbr = np.polyfit(data['A-GBR'], data['records'], 1)
linear_fit_a_gbr = coeff_a_gbr[0] * data['A-GBR'] + coeff_a_gbr[1]
idea_fit_a_gbr = 1 * data['A-GBR']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(data['A-GBR'], data['records'], 'o', color='blue', label='', linewidth=1.0)
plt.plot(data['A-GBR'], linear_fit_a_gbr, '--', color='red', label='Linear fit of A-GBR')
plt.plot(data['A-GBR'], idea_fit_a_gbr, '-', color='black', label='Ideal fit of A-GBR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

# SVR
plt.subplot(4, 2, 3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t, data['records'], '-', color='blue', linewidth=1.5, label='records')
plt.plot(t, data['A-SVR'], '--', color='red', linewidth=1.5, label='A-SVR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

plt.subplot(4, 2, 4)
coeff_a_svr = np.polyfit(data['A-SVR'], data['records'], 1)
linear_fit_a_svr = coeff_a_svr[0] * data['A-SVR'] + coeff_a_svr[1]
idea_fit_a_svr = 1 * data['A-SVR']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(data['A-SVR'], data['records'], 'o', color='blue', label='', linewidth=1.0)
plt.plot(data['A-SVR'], linear_fit_a_svr, '--', color='red', label='Linear fit of A-SVR')
plt.plot(data['A-SVR'], idea_fit_a_svr, '-', color='black', label='Ideal fit of A-SVR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

# ARMA
plt.subplot(4, 2, 5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t, data['records'], '-', color='blue', linewidth=1.5, label='records')
plt.plot(t, data['A-ARMA'], '--', color='red', linewidth=1.5, label='A-ARMA')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)


plt.subplot(4, 2, 6)
coeff_a_arma = np.polyfit(data['A-ARMA'], data['records'], 1)
linear_fit_a_arma = coeff_a_arma[0] * data['A-ARMA'] + coeff_a_arma[1]
idea_fit_a_arma = 1 * data['A-ARMA']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(data['A-ARMA'],data['records'],'o',color='blue',label='',linewidth=1.0)
plt.plot(data['A-ARMA'], linear_fit_a_arma, '--', color='red', label='Linear fit of A-ARMA')
plt.plot(data['A-ARMA'], idea_fit_a_arma, '-', color='black', label='Ideal fit of A-ARMA')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

# DNN
plt.subplot(4, 2, 7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t, data['records'], '-', color='blue', linewidth=1.5, label='records')
plt.plot(t, data['A-DNN'], '--', color='red', linewidth=1.5, label='A-DNN')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

plt.subplot(4, 2, 8)
coeff_a_dnn = np.polyfit(data['A-DNN'], data['records'], 1)
linear_fit_a_dnn = coeff_a_dnn[0] * data['A-DNN'] + coeff_a_dnn[1]
idea_fit_a_dnn = 1 * data['A-DNN']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(data['A-DNN'], data['records'], 'o', color='blue', label='', linewidth=1.0)
plt.plot(data['A-DNN'], linear_fit_a_dnn, '--', color='red', label='Linear fit of A-DNN')
plt.plot(data['A-DNN'], idea_fit_a_dnn, '-', color='black', label='Ideal fit of A-DNN')

plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)
plt.savefig(current_path+'\\models\\Multi_model_compare_subplots_Addtive.tif', format='tiff', dpi=600)
plt.show()



#################################################
#################################################
#################################################

plt.figure(figsize=(8, 10))
plt.subplots_adjust(
    left=0.08, bottom=0.05, right=0.92, top=0.95, hspace=0.4, wspace=0.3)

# GBR
plt.subplot(4, 2, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t, data['records'], '-', color='blue', linewidth=1.5, label='records')
plt.plot(t, data['DNN-GBR'], '--', color='red', linewidth=1.5, label='DNN-GBR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

plt.subplot(4, 2, 2)
coeff_dnn_gbr = np.polyfit(data['DNN-GBR'], data['records'], 1)
linear_fit_dnn_gbr = coeff_dnn_gbr[0] * data['DNN-GBR'] + coeff_dnn_gbr[1]
idea_fit_dnn_gbr = 1 * data['DNN-GBR']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(
    data['DNN-GBR'], data['records'], 'o', color='blue', label='', linewidth=1.0)
plt.plot(
    data['DNN-GBR'],
    linear_fit_dnn_gbr,
    '--',
    color='red',
    label='Linear fit of DNN-GBR')
plt.plot(
    data['DNN-GBR'],
    idea_fit_dnn_gbr,
    '-',
    color='black',
    label='Ideal fit of DNN-GBR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

# SVR
plt.subplot(4, 2, 3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t, data['records'], '-', color='blue', linewidth=1.5, label='records')
plt.plot(t, data['DNN-SVR'], '--', color='red', linewidth=1.5, label='DNN-SVR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

plt.subplot(4, 2, 4)
coeff_dnn_svr = np.polyfit(data['DNN-SVR'], data['records'], 1)
linear_fit_dnn_svr = coeff_dnn_svr[0] * data['DNN-SVR'] + coeff_dnn_svr[1]
idea_fit_dnn_svr = 1 * data['DNN-SVR']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(
    data['DNN-SVR'], data['records'], 'o', color='blue', label='', linewidth=1.0)
plt.plot(
    data['DNN-SVR'],
    linear_fit_dnn_svr,
    '--',
    color='red',
    label='Linear fit of DNN-SVR')
plt.plot(
    data['DNN-SVR'],
    idea_fit_dnn_svr,
    '-',
    color='black',
    label='Ideal fit of DNN-SVR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

# ARMA
plt.subplot(4, 2, 5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t, data['records'], '-', color='blue', linewidth=1.5, label='records')
plt.plot(t, data['DNN-ARMA'], '--', color='red', linewidth=1.5, label='DNN-ARMA')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

plt.subplot(4, 2, 6)
coeff_dnn_arma = np.polyfit(data['DNN-ARMA'], data['records'], 1)
linear_fit_dnn_arma = coeff_dnn_arma[0] * data['DNN-ARMA'] + coeff_dnn_arma[1]
idea_fit_dnn_arma = 1 * data['DNN-ARMA']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(
    data['DNN-ARMA'],
    data['records'],
    'o',
    color='blue',
    label='',
    linewidth=1.0)
plt.plot(
    data['DNN-ARMA'],
    linear_fit_dnn_arma,
    '--',
    color='red',
    label='Linear fit of DNN-ARMA')
plt.plot(
    data['DNN-ARMA'],
    idea_fit_dnn_arma,
    '-',
    color='black',
    label='Ideal fit of DNN-ARMA')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

# DNN
plt.subplot(4, 2, 7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t, data['records'], '-', color='blue', linewidth=1.5, label='records')
plt.plot(t, data['DNN-DNN'], '--', color='red', linewidth=1.5, label='DNN-DNN')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

plt.subplot(4, 2, 8)
coeff_dnn_dnn = np.polyfit(data['DNN-DNN'], data['records'], 1)
linear_fit_dnn_dnn = coeff_dnn_dnn[0] * data['DNN-DNN'] + coeff_dnn_dnn[1]
idea_fit_dnn_dnn = 1 * data['DNN-DNN']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(
    data['DNN-DNN'], data['records'], 'o', color='blue', label='', linewidth=1.0)
plt.plot(
    data['DNN-DNN'],
    linear_fit_dnn_dnn,
    '--',
    color='red',
    label='Linear fit of DNN-DNN')
plt.plot(
    data['DNN-DNN'],
    idea_fit_dnn_dnn,
    '-',
    color='black',
    label='Ideal fit of DNN-DNN')

plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)
plt.savefig(
    current_path + '\\models\\Multi_model_compare_ensemble_subplots.tif',
    format='tiff',
    dpi=600)
plt.show()




#=============+++++++++++++++++++++++++++++++
#--------------------------------------------------------------------------------------------------
#################################################************************************************
plt.figure(figsize=(8, 10))
plt.subplots_adjust(
    left=0.08, bottom=0.05, right=0.92, top=0.95, hspace=0.4, wspace=0.3)

# GBR
plt.subplot(4, 2, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t, data['records'], '-', color='blue', linewidth=1.5, label='records')
plt.plot(t, data['O-GBR'], '--', color='red', linewidth=1.5, label='O-GBR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

plt.subplot(4, 2, 2)
coeff_O_gbr = np.polyfit(data['O-GBR'], data['records'], 1)
linear_fit_O_gbr = coeff_O_gbr[0] * data['O-GBR'] + coeff_O_gbr[1]
idea_fit_O_gbr = 1 * data['O-GBR']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(
    data['O-GBR'],
    data['records'],
    'o',
    color='blue',
    label='',
    linewidth=1.0)
plt.plot(
    data['O-GBR'],
    linear_fit_O_gbr,
    '--',
    color='red',
    label='Linear fit of O-GBR')
plt.plot(
    data['O-GBR'],
    idea_fit_O_gbr,
    '-',
    color='black',
    label='Ideal fit of O-GBR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

# SVR
plt.subplot(4, 2, 3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t, data['records'], '-', color='blue', linewidth=1.5, label='records')
plt.plot(t, data['O-SVR'], '--', color='red', linewidth=1.5, label='O-SVR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

plt.subplot(4, 2, 4)
coeff_O_svr = np.polyfit(data['O-SVR'], data['records'], 1)
linear_fit_O_svr = coeff_O_svr[0] * data['O-SVR'] + coeff_O_svr[1]
idea_fit_O_svr = 1 * data['O-SVR']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(
    data['O-SVR'],
    data['records'],
    'o',
    color='blue',
    label='',
    linewidth=1.0)
plt.plot(
    data['O-SVR'],
    linear_fit_O_svr,
    '--',
    color='red',
    label='Linear fit of O-SVR')
plt.plot(
    data['O-SVR'],
    idea_fit_O_svr,
    '-',
    color='black',
    label='Ideal fit of O-SVR')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

# ARMA
plt.subplot(4, 2, 5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t, data['records'], '-', color='blue', linewidth=1.5, label='records')
plt.plot(
    t, data['O-ARMA'], '--', color='red', linewidth=1.5, label='O-ARMA')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

plt.subplot(4, 2, 6)
coeff_O_arma = np.polyfit(data['O-ARMA'], data['records'], 1)
linear_fit_O_arma = coeff_O_arma[0] * data['O-ARMA'] + coeff_O_arma[1]
idea_fit_O_arma = 1 * data['O-ARMA']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(
    data['O-ARMA'],
    data['records'],
    'o',
    color='blue',
    label='',
    linewidth=1.0)
plt.plot(
    data['O-ARMA'],
    linear_fit_O_arma,
    '--',
    color='red',
    label='Linear fit of O-ARMA')
plt.plot(
    data['O-ARMA'],
    idea_fit_O_arma,
    '-',
    color='black',
    label='Ideal fit of O-ARMA')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

# DNN
plt.subplot(4, 2, 7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time(d)', fontsize=12)
plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=12)
plt.plot(t, data['records'], '-', color='blue', linewidth=1.5, label='records')
plt.plot(t, data['O-DNN'], '--', color='red', linewidth=1.5, label='O-DNN')
plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)

plt.subplot(4, 2, 8)
coeff_O_dnn = np.polyfit(data['O-DNN'], data['records'], 1)
linear_fit_O_dnn = coeff_O_dnn[0] * data['O-DNN'] + coeff_O_dnn[1]
idea_fit_O_dnn = 1 * data['O-DNN']
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=12)
plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=12)
plt.plot(
    data['O-DNN'],
    data['records'],
    'o',
    color='blue',
    label='',
    linewidth=1.0)
plt.plot(
    data['O-DNN'],
    linear_fit_O_dnn,
    '--',
    color='red',
    label='Linear fit of O-DNN')
plt.plot(
    data['O-DNN'],
    idea_fit_O_dnn,
    '-',
    color='black',
    label='Ideal fit of O-DNN')

plt.legend(
    loc=0,
    # bbox_to_anchor=(0.55, 1),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=8)
plt.savefig(
    current_path + '\\models\\Multi_model_compare_orig_subplots.tif',
    format='tiff',
    dpi=600)
plt.show()