import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1, os.path.pardir))
print(10 * '-' + ' Current Path: {}'.format(current_path))
print(10 * '-' + ' Parent Path: {}'.format(par_path_1))
print(10 * '-' + ' Grandpa Path: {}'.format(par_path_2))

# Download the dataset
imfs = pd.read_excel(par_path_1+"\\data\\logtrans_eemd_imfs.xlsx")

fig = plt.figure(figsize=(16, 24))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax1 = plt.subplot(13, 1, 1)
plt.ylabel('PACF0',fontsize=18)
fig = sm.graphics.tsa.plot_pacf(imfs['raw_logtrans'], lags=20, ax=ax1,title='',color='red')

ax2 = plt.subplot(13,1,2)
plt.ylabel('PACF1',fontsize=18)
fig = sm.graphics.tsa.plot_pacf(imfs['imf1'], lags=20, ax=ax2,title="",color='red')

ax3 = plt.subplot(13, 1, 3)
plt.ylabel('PACF2', fontsize=18)
fig = sm.graphics.tsa.plot_pacf(
    imfs['imf2'], lags=20, ax=ax3, title="", color='red')

ax4 = plt.subplot(13, 1, 4)
plt.ylabel('PACF3', fontsize=18)
fig = sm.graphics.tsa.plot_pacf(
    imfs['imf3'], lags=20, ax=ax4, title="", color='red')

ax5 = plt.subplot(13, 1, 5)
plt.ylabel('PACF4', fontsize=18)
fig = sm.graphics.tsa.plot_pacf(
    imfs['imf4'], lags=20, ax=ax5, title="", color='red')

ax6 = plt.subplot(13, 1, 6)
plt.ylabel('PACF5', fontsize=18)
fig = sm.graphics.tsa.plot_pacf(
    imfs['imf5'], lags=20, ax=ax6, title="", color='red')

ax7 = plt.subplot(13, 1, 7)
plt.ylabel('PACF6', fontsize=18)
fig = sm.graphics.tsa.plot_pacf(
    imfs['imf6'], lags=20, ax=ax7, title="", color='red')

ax8 = plt.subplot(13, 1, 8)
plt.ylabel('PACF7', fontsize=18)
fig = sm.graphics.tsa.plot_pacf(
    imfs['imf7'], lags=20, ax=ax8, title="", color='red')

ax9 = plt.subplot(13, 1, 9)
plt.ylabel('PACF8', fontsize=18)
fig = sm.graphics.tsa.plot_pacf(
    imfs['imf8'], lags=20, ax=ax9, title="", color='red')

ax10 = plt.subplot(13, 1, 10)
plt.ylabel('PACF9', fontsize=18)
fig = sm.graphics.tsa.plot_pacf(
    imfs['imf9'], lags=20, ax=ax10, title="", color='red')

ax11 = plt.subplot(13, 1, 11)
plt.ylabel('PACF10', fontsize=18)
fig = sm.graphics.tsa.plot_pacf(
    imfs['imf10'], lags=20, ax=ax11, title="", color='red')

ax12 = plt.subplot(13, 1, 12)
plt.ylabel('PACF11', fontsize=18)
fig = sm.graphics.tsa.plot_pacf(
    imfs['imf11'], lags=20, ax=ax12, title="", color='red')

ax13 = plt.subplot(13, 1, 13)
plt.ylabel('PACF12', fontsize=18)
fig = sm.graphics.tsa.plot_pacf(
    imfs['R12'], lags=20, ax=ax13, title="", color='red')

plt.xlabel('Lag',fontsize=18)
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.2, wspace=0.5)
plt.tight_layout()
plt.savefig(par_path_1+"\\PACFs\\logtrans_tanmiao_day_PACFS.tif",format='tiff',dpi=1000)
plt.show()
