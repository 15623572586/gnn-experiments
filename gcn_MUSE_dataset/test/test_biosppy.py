import os.path

import numpy as np
import pandas as pd
from biosppy import storage
from biosppy.signals import ecg
import torch
from matplotlib import pyplot as plt

from process.variables import processed_data, dataset_path

df = pd.read_csv(os.path.join(dataset_path, 'ECGDataDenoised/MUSE_20180111_155115_19000.csv'), header=None)
ecg_data = np.array(df.values).T

# R_peaks = ecg.ASI_segmenter(signal=ecg_data[1], sampling_rate=500)
# print(R_peaks[0])
# count = 0
# for i in range(1, len(R_peaks[0])):
#     count += R_peaks[0][i] - R_peaks[0][i-1]
# print(count / (len(R_peaks[0]) - 1) / 500)

# 更准确
R_peaks = ecg.christov_segmenter(signal=ecg_data[1], sampling_rate=500)
print(R_peaks)
RR_In = []
for i in range(1, len(R_peaks[0])):
    RR_In.append(R_peaks[0][i] - R_peaks[0][i-1])
RR_In = np.array(RR_In)
print('均值：' + str(np.mean(RR_In) / 500))
# print(count / (len(R_peaks[0]) - 1) / 500)

# R_peaks = ecg.engzee_segmenter(signal=ecg_data[1], sampling_rate=500)
# print(R_peaks)
# count = 0
# for i in range(1, len(R_peaks[0])):
#     count += R_peaks[0][i] - R_peaks[0][i-1]
# print(count / (len(R_peaks[0]) - 1) / 500)
#
# R_peaks = ecg.gamboa_segmenter(signal=ecg_data[1], sampling_rate=500)
# print(R_peaks)
# count = 0
# for i in range(1, len(R_peaks[0])):
#     count += R_peaks[0][i] - R_peaks[0][i-1]
# print(count / (len(R_peaks[0]) - 1) / 500)
#
# R_peaks = ecg.ssf_segmenter(signal=ecg_data[1], sampling_rate=500)
# print(R_peaks)
# count = 0
# for i in range(1, len(R_peaks[0])):
#     count += R_peaks[0][i] - R_peaks[0][i-1]
# print(count / (len(R_peaks[0]) - 1) / 500)
#
# R_peaks = ecg.ecg(signal=ecg_data[1], sampling_rate=500, show=False)[2]
# print(R_peaks)
# count = 0
# for i in range(1, len(R_peaks)):
#     count += R_peaks[i] - R_peaks[i-1]
# print(count / (len(R_peaks) - 1) / 500)

# out = ecg.ecg(signal=ecg_data[1], sampling_rate=500, show=True)
# plt.figure(figsize=(60, 30))
# for i in range(len(ecg_data)):
#     plt.subplot(12, 1, i + 1)
#     plt.plot(ecg_data[i])
# plt.show()
