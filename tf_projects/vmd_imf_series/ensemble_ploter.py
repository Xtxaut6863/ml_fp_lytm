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



