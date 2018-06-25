import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
par_path_1 = os.path.abspath(os.path.join(current_path, os.path.pardir))
par_path_2 = os.path.abspath(os.path.join(par_path_1, os.path.pardir))
print(10 * '-' + ' Current Path: {}'.format(current_path))
print(10 * '-' + ' Parent Path: {}'.format(par_path_1))
print(10 * '-' + ' Grandpa Path: {}'.format(par_path_2))

# Download the dataset
imfs = pd.read_excel(par_path_1+"\\data\\orig_eemd_imfs.xlsx")

# create the time series number
t = np.linspace(start=1,stop=imfs['raw'].size,num=imfs['raw'].size)

plt.figure(figsize=(8,10))

plt.subplot(13,1,1)
plt.xticks([])
plt.plot(t,imfs['raw'],color='blue')
plt.ylabel('x',fontsize=12)

plt.subplot(13,1,2)
plt.xticks([])
plt.plot(t,imfs['imf1'],color='blue')
plt.ylabel('IMF1',fontsize=12)

plt.subplot(13, 1, 3)
plt.xticks([])
plt.plot(t, imfs['imf2'], color='blue')
plt.ylabel('IMF2',fontsize=12)

plt.subplot(13, 1, 4)
plt.xticks([])
plt.plot(t, imfs['imf3'], color='blue')
plt.ylabel('IMF3',fontsize=12)

plt.subplot(13, 1, 5)
plt.xticks([])
plt.plot(t, imfs['imf4'], color='blue')
plt.ylabel('IMF4',fontsize=12)

plt.subplot(13, 1, 6)
plt.xticks([])
plt.plot(t, imfs['imf5'], color='blue')
plt.ylabel('IMF5',fontsize=12)

plt.subplot(13, 1, 7)
plt.xticks([])
plt.plot(t, imfs['imf6'], color='blue')
plt.ylabel('IMF6',fontsize=12)

plt.subplot(13, 1, 8)
plt.xticks([])
plt.plot(t, imfs['imf7'], color='blue')
plt.ylabel('IMF7',fontsize=12)

plt.subplot(13, 1, 9)
plt.xticks([])
plt.plot(t, imfs['imf8'], color='blue')
plt.ylabel('IMF8',fontsize=12)

plt.subplot(13, 1, 10)
plt.xticks([])
plt.plot(t, imfs['imf9'], color='blue')
plt.ylabel('IMF9',fontsize=12)

plt.subplot(13, 1, 11)
plt.xticks([])
plt.plot(t, imfs['imf10'], color='blue')
plt.ylabel('IMF10',fontsize=12)

plt.subplot(13, 1, 12)
plt.xticks([])
plt.plot(t, imfs['imf11'], color='blue')
plt.ylabel('IMF11',fontsize=12)

plt.subplot(13, 1, 13)
plt.plot(t, imfs['r12'], color='blue')
plt.ylabel('R12',fontsize=12)
plt.xlabel('Time(day)',fontsize=12)

plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, hspace=0.5, wspace=0.5)
plt.savefig(par_path_1+"\\imfs\\orig_tanmiao_day_imfs.tif",format='tiff',dpi=1000)
# plt.tight_layout()
plt.show()
