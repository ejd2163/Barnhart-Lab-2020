import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
# from Resample import resample

# Read the csv file
parent_path = "/Users/zhengyuanlu/Desktop/Ctrl/"
F_path = os.path.join(parent_path, "all_F.csv")
t_path = os.path.join(parent_path, "all_t.csv")
print(F_path, t_path)

GC = pd.read_csv(F_path)
time = pd.read_csv(t_path)

print(GC.info)
GC.dropna(axis=1, inplace=True)
time.dropna(axis=1, inplace=True)
print(GC.info)
GC_n = GC.drop(columns=['sample', 'mito_number', 'trial'])
time_n = time.drop(columns=['sample', 'mito_number', 'trial'])
print(GC_n.info)

start = int(min(abs(time_n).idxmin(axis=1)))

print("Start: ", start)
baseline = GC_n.iloc[:, 0:start]
print("Baseline: ", baseline)
F = np.mean(baseline, axis=1)
print(F)

GC_norm = ((GC_n.T - F) / F).T
# GC_norm = GC_n

plt.imshow(GC_norm, cmap="hot")
plt.colorbar()
plt.savefig(parent_path+"GC_norm-4.pdf")
plt.show()
print(GC_norm)
GC_norm.to_csv(parent_path + "all_F_norm.csv", index=False)
time_n.to_csv(parent_path + "all_t_n.csv", index=False)

GC_mean = np.mean(GC_norm, axis=0)
GC_std = np.std(GC_norm, axis=0)

t = np.mean(time_n, axis=0)

"""""""""
for i in range(530):
    x = time_n.iloc[i]
    #y = (GC_norm.iloc[i]).rolling(10).mean()
    y = signal.savgol_filter(GC_norm.iloc[i], 73, 3)
    #y = GC_norm.iloc[i]
    plt.plot(x, y, alpha=0.8, linewidth=0.3)
"""""""""""

plt.plot(t, GC_mean, color='k')
# plt.fill_between(t, GC_mean+GC_std, GC_mean-GC_std, color='b', alpha=0.2)

plt.xlabel("Time/s")
plt.ylabel("dF/F")
plt.savefig(parent_path+"GC-norm-1.pdf")
# plt.show()