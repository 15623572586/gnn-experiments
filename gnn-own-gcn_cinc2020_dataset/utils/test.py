import math
import os

import pandas as pd
import pywt
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas
import wfdb.processing
from pandas import DataFrame
from scipy.signal import medfilt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from torch import nn

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

from process.cinc_preprocess import classes
from process.variables import dataset_path, processed_path, processed_data

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def test_resample(log_file):
    ecg_data, meta_data = wfdb.rdsamp(os.path.join(dataset_path, 'HR00001'))
    all_sig = ecg_data.T
    all_sig_lr = []
    for sig in all_sig:
        data = wfdb.processing.resample_sig(x=sig, fs=500, fs_target=100)
        all_sig_lr.append(data[0])

    res = pandas.read_csv(os.path.join('../output/logs', log_file + '.csv'))
    train_loss = np.array(res['train_loss'])
    epoch = [i for i in range(0, train_loss.size)]
    plt.plot(epoch, train_loss, label='train_loss')
    plt.legend()
    plt.show()


# 中值滤波去基线：中值滤波可以有效去除心电信号中的基线漂移噪声，对S-T段有一定的保护作用。
def filter_1():
    filter = int(0.8 * 100)
    ecg_data, meta_data = wfdb.rdsamp(
        os.path.join(r'D:\learning\科研\数据\PhysioNetChallenge2020\1_PhysioNetChallenge2020_Training_CPSC', 'A3329'))
    ecg_data = ecg_data.T
    all_sig_lr = []
    for sig in ecg_data:
        data = wfdb.processing.resample_sig(x=sig, fs=500, fs_target=100)
        all_sig_lr.append(data[0])
    ecg_data = all_sig_lr
    baseline = []
    for idx, lead in enumerate(leads):
        baseline.append(medfilt(ecg_data[idx], filter + 1))
    filtered_data = ecg_data - np.array(baseline)

    plt.figure(figsize=(60, 5))
    plt.plot(ecg_data[2])
    plt.plot(filtered_data[2])
    plt.show()
    return filtered_data


# 小波滤波
def filter_2(ecg_data):
    w = pywt.Wavelet('db8')
    filtered_data = []
    # 选用Daubechies8小波
    for data in ecg_data:
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        threshold = 0.1  # Threshold for filtering
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解
        # 这里就是对每一层的coffe进行更改
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        filtered_data.append(pywt.waverec(coeffs, 'db8'))  # 将信号进行小波重构
    filtered_data = np.array(filtered_data)
    # plt.figure(figsize=(60, 5))
    # plt.plot(ecg_data[2])
    # plt.plot(filtered_data[2])
    # plt.legend(['Before', 'After'])
    # plt.show()


def test_zero():
    # arr =np.zeros(3)
    arr = [0, 0.0, 1]
    print(np.any(arr))


if __name__ == '__main__':
    # drawing_ecg('2022-10-03-19-21-12')
    # data = filter_1()
    # filter_2(data)
    # features = torch.rand((32, 1000, 12))
    # features = features.permute(0, 2, 1)  # [32, 12, 1000]
    # attention = SelfAttention(2, 1000, 12, 0.5)
    # result = attention.forward(features)
    # print(result.shape)
    over_sample_test()
    # test_zero()
    # print(format(22, '05d'))
