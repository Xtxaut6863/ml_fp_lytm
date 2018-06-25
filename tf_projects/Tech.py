import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VMD_IMFs = pd.read_excel('F:/ml_fp_lytm/data/VMD_IMFS.xlsx')
raw = VMD_IMFs['Y']
raw_t = np.linspace(start=1,stop=raw.size,num=raw.size)
figure1 = plt.figure(figsize=(2,0.5))
plt.subplots_adjust(left=0.005, bottom=0.1, right=0.995, top=0.9, hspace=0.2, wspace=0.3)
plt.plot(raw_t,raw,'-',color='blue',linewidth=1.5)
plt.axis([0, 5419, 0, 800])
plt.xticks([])
plt.yticks([])
plt.savefig('F:/ml_fp_lytm/Tech/raw.png', format='png', dpi=600)
plt.show()

figure2 = plt.figure(figsize=(2, 0.5))
plt.subplots_adjust(left=0.005, bottom=0.1, right=0.995, top=0.9, hspace=0.2, wspace=0.3)
plt.plot(raw_t, VMD_IMFs['X1'], '-', color='blue', linewidth=1.5)
plt.axis([0, 5419, -2, 40])
plt.xticks([])
plt.yticks([])
plt.savefig(
    'F:/ml_fp_lytm/Tech/IMF1.png',
    format='png',
    dpi=600)
plt.show()

figure3 = plt.figure(figsize=(2, 0.5))
plt.subplots_adjust(left=0.005, bottom=0.1, right=0.995, top=0.9, hspace=0.2, wspace=0.3)
plt.plot(raw_t, VMD_IMFs['X2'], '-', color='blue', linewidth=1.5)
plt.axis([0, 5419, -50, 50])
plt.xticks([])
plt.yticks([])
plt.savefig(
    'F:/ml_fp_lytm/Tech/IMF2.png',
    format='png',
    dpi=600)
plt.show()

figure4 = plt.figure(figsize=(2, 0.5))
plt.subplots_adjust(left=0.005, bottom=0.1, right=0.995, top=0.9, hspace=0.2, wspace=0.3)
plt.plot(raw_t, VMD_IMFs['X10'], '-', color='blue', linewidth=1.5)
plt.axis([0, 5419, -70, 70])
plt.xticks([])
plt.yticks([])
plt.savefig(
    'F:/ml_fp_lytm/Tech/IMF10.png',
    format='png',
    dpi=600)
plt.show()