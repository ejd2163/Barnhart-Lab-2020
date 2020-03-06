import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

parent_path = "/Users/zhengyuanlu/Desktop/Ctrl/"

def resample(t, y, f):
    """
    Resample the signal at frequency f.
    :param t: time (s)
    :param y: signal (e.g., GCaMP fluorescence data)
    :param f: resampling frequency (Hz)
    :return: resampled signal data and timepoint
    """

    # loc = np.linspace(max(t["1"]), min(t["273"]), num)

    limit_l = int(f * max(t.iloc[:, 0])) / f
    limit_r = int(f * min(t.iloc[:, -1])) / f

    num = (limit_r - limit_l) * f
    loc = np.linspace(limit_l, limit_r, num + 1)
    # print(loc)

    data = np.array([])
    plot_list = random.sample(range(t.shape[0]), 10)

    for i in range(t.shape[0]):
        t_row = t.iloc[i]
        y_row = y.iloc[i]
        row = np.array([])
        k = 0
        for j in range(len(loc)):
            # find the interval
            while True:
                if t_row[k] <= loc[j] <= t_row[k + 1]:
                    break
                k += 1
                # print(k)
            row = np.append(row, (y_row[k + 1] - y_row[k]) * (loc[j] - t_row[k]) / (t_row[k + 1] - t_row[k]) + y_row[k])
        # print(row)
        if i == 0:
            data = row
        else:
            data = np.vstack((data, row))

        # plot 10 example traces
        # if i in plot_list:
        #     plt.figure()
        #     plt.plot(t_row, y_row, 'b', linewidth=0.5)
        #     plt.plot(loc, row, 'ro', markersize=0.5)
        #     plt.savefig(parent_path + "/Resampled/GC_r-ex{}.pdf".format(i))

    return data, loc


GC_norm = pd.read_csv(parent_path + "all_F_norm.csv")  # normalized F, numerical only
time_n = pd.read_csv(parent_path + "all_t_n.csv")  # t, numerical only

GC_r, t_r = resample(time_n, GC_norm, 10)
#
GC_r = pd.DataFrame(GC_r)
GC_r.to_csv(parent_path + "F_resampled.csv", index=False)

# GC_r = pd.read_csv(parent_path + "F_resampled.csv")

# Plot
plt.figure()
GC_mean = np.mean(GC_r, axis=0)
GC_std = np.std(GC_r, axis=0)
plt.plot(t_r, GC_mean, color='k')
# plt.fill_between(t_r, GC_mean + GC_std, GC_mean - GC_std, color='b', alpha=0.2)
# print(GC_r.info)
plt.savefig(parent_path + "GC-resampled-1.pdf")

plt.figure()
plt.imshow(GC_r, cmap="hot")

print("t_r= ", t_r)
x = np.where(t_r == 0)
print("X= ", x)
x = x[0][0]
plt.plot([x, x], [0, GC_r.shape[0]-2], '--w')
plt.colorbar()

# plt.xlabel("Time/s")
plt.ylabel("Mito#")
plt.xlabel("Time point#")
plt.savefig(parent_path + "GC-resampled-heatmap.pdf")