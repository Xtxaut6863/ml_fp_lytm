import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1, os.path.pardir))
print(10 * '-' + ' Current Path: {}'.format(current_path))
print(10 * '-' + ' Parent Path: {}'.format(par_path_1))
print(10 * '-' + ' Grandpa Path: {}'.format(par_path_2))

q = np.linspace(start=1,stop=21,num=21)
R2 = pd.read_excel(current_path+'\\ARMA_PQ_SELECT.xlsx')
MSE = pd.read_excel(current_path+'\\ARMA_PQ_SELECT.xlsx','imf1-MSE')
plt.figure(figsize=(10, 6))
# plt.title('flow prediction based on DNN')

plt.xlabel('q', fontsize=12)
plt.ylabel("R2", fontsize=12)
plt.yscale('logit')
plt.xticks(fontsize=12)
plt.yticks([0.8,0.1,0.9],fontsize=12)
plt.plot(q,R2['[1,q]'],'-',color='blue',marker='o',label='[1,q]')
plt.plot(q, R2['[2,q]'], '-', color='black', marker='o', label='[2,q]')
plt.plot(q,R2['[3,q]'],'-',color='orange',marker='o',label='[3,q]')
plt.plot(q, R2['[4,q]'], '-', color='green', marker='o', label='[4,q]')
plt.plot(
    q,
    R2['[5,q]'],
    '-',
    color='orchid',
    marker='o',
    label='[5,q]')
plt.plot(
    q,
    R2['[6,q]'],
    '-',
    color='slategray',
    marker='o',
    label='[6,q]')
plt.plot(
    q, R2['[7,q]'], '-', color='olive', marker='o', label='[7,q]')
plt.plot(q, R2['[8,q]'], '-', color='indigo', marker='o', label='[8,q]')
plt.plot(
    q,
    R2['[9,q]'],
    '-',
    color='darkolivegreen',
    marker='o',
    label='[9,q]')
plt.plot(
    q,
    R2['[10,q]'],
    '-',
    color='burlywood',
    marker='o',
    label='[10,q]')
plt.plot(
    q,
    R2['[11,q]'],
    '-',
    color='chartreuse',
    marker='o',
    label='[11,q]')
plt.plot(
    q,
    R2['[12,q]'],
    '-',
    color='chocolate',
    marker='o',
    label='[12,q]')
plt.plot(
    q, R2['[13,q]'], '-', color='coral', marker='o', label='[13,q]')
plt.plot(
    q,
    R2['[14,q]'],
    '-',
    color='cornflowerblue',
    marker='o',
    label='[14,q]')
plt.plot(
    q,
    R2['[15,q]'],
    '-',
    color='brown',
    marker='o',
    label='[15,q]')
plt.plot(
    q,
    R2['[16,q]'],
    '-',
    color='crimson',
    marker='o',
    label='[16,q]')
plt.plot(
    q, R2['[17,q]'], '-', color='cyan', marker='o', label='[17,q]')
plt.plot(
    q,
    R2['[18,q]'],
    '-',
    color='darkblue',
    marker='o',
    label='[18,q]')
plt.plot(
    q,
    R2['[19,q]'],
    '-',
    color='darkcyan',
    marker='o',
    label='[19,q]')
plt.plot(
    q,
    R2['[20,q]'],
    '-',
    color='darkgoldenrod',
    marker='o',
    label='[20,q]')
plt.plot(
    q,
    R2['[21,q]'],
    '-',
    color='red',
    marker='o',
    label='[21,q]')
plt.legend(
    loc=0,
    bbox_to_anchor=(1.2, 1.02),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=12)
plt.subplots_adjust(
    left=0.15, bottom=0.1, right=0.85, top=0.9, hspace=0.5, wspace=0.5)
plt.savefig(
    current_path+'\\arma_pq_select_imf1_r2.tif',
    format='tiff',
    dpi=600)
plt.show()


plt.figure(figsize=(10, 6))
# plt.title('flow prediction based on DNN')
plt.xlabel('q', fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.yscale('logit')
plt.xticks(fontsize=12)
plt.yticks([0,0.1,0.3],fontsize=12)
plt.plot(q,MSE['[1,q]'],'-',color='blue',marker='o',label='[1,q]')
plt.plot(q, MSE['[2,q]'], '-', color='black', marker='o', label='[2,q]')
plt.plot(q,MSE['[3,q]'],'-',color='orange',marker='o',label='[3,q]')
plt.plot(q, MSE['[4,q]'], '-', color='green', marker='o', label='[4,q]')
plt.plot(
    q,
    MSE['[5,q]'],
    '-',
    color='orchid',
    marker='o',
    label='[5,q]')
plt.plot(
    q,
    MSE['[6,q]'],
    '-',
    color='slategray',
    marker='o',
    label='[6,q]')
plt.plot(
    q, MSE['[7,q]'], '-', color='olive', marker='o', label='[7,q]')
plt.plot(
    q, MSE['[8,q]'], '-', color='indigo', marker='o', label='[8,q]')
plt.plot(
    q,
    MSE['[9,q]'],
    '-',
    color='darkolivegreen',
    marker='o',
    label='[9,q]')
plt.plot(
    q,
    MSE['[10,q]'],
    '-',
    color='burlywood',
    marker='o',
    label='[10,q]')
plt.plot(
    q,
    MSE['[11,q]'],
    '-',
    color='chartreuse',
    marker='o',
    label='[11,q]')
plt.plot(
    q,
    MSE['[12,q]'],
    '-',
    color='chocolate',
    marker='o',
    label='[12,q]')
plt.plot(
    q, MSE['[13,q]'], '-', color='coral', marker='o', label='[13,q]')
plt.plot(
    q,
    MSE['[14,q]'],
    '-',
    color='cornflowerblue',
    marker='o',
    label='[14,q]')
plt.plot(
    q,
    MSE['[15,q]'],
    '-',
    color='brown',
    marker='o',
    label='[15,q]')
plt.plot(
    q,
    MSE['[16,q]'],
    '-',
    color='crimson',
    marker='o',
    label='[16,q]')
plt.plot(
    q, MSE['[17,q]'], '-', color='cyan', marker='o', label='[17,q]')
plt.plot(
    q,
    MSE['[18,q]'],
    '-',
    color='darkblue',
    marker='o',
    label='[18,q]')
plt.plot(
    q,
    MSE['[19,q]'],
    '-',
    color='darkcyan',
    marker='o',
    label='[19,q]')
plt.plot(
    q,
    MSE['[20,q]'],
    '-',
    color='darkgoldenrod',
    marker='o',
    label='[20,q]')
plt.plot(
    q,
    MSE['[21,q]'],
    '-',
    color='red',
    marker='o',
    label='[21,q]')
plt.legend(
    loc=0,
    bbox_to_anchor=(1.2, 1.02),
    # ncol=3,
    fancybox=True,
    shadow=True,
    fontsize=12)
plt.subplots_adjust(
    left=0.15, bottom=0.1, right=0.85, top=0.9, hspace=0.5, wspace=0.5)
plt.savefig(
    current_path+'\\arma_pq_select_imf1_mse.tif',
    format='tiff',
    dpi=600)
plt.show()