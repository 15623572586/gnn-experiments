import os

import pywt
import wfdb

from biosppy import ecg, tools
import numpy as np
import pandas as pd
import scipy
from pandas import DataFrame
from tqdm import tqdm

from process.variables import processed_path, dataset_path, features_path


def cal_entropy(coeff):
    coeff = pd.Series(coeff).value_counts()  # Series(coeff)生成列表List，统计列表中每个值有多少个重复的
    entropy = scipy.stats.entropy(coeff)  # 计算给定离散序列的分布熵
    return entropy / 10


def cal_statistics(signal):
    n5 = np.percentile(signal, 5)  # single排序后5%的分位数
    n25 = np.percentile(signal, 25)
    n75 = np.percentile(signal, 75)
    n95 = np.percentile(signal, 95)
    median = np.percentile(signal, 50)  # 中位数
    mean = np.mean(signal)  # 平均数位数
    std = np.std(signal)  # 标准差
    var = np.var(signal)  # 方差
    return [n5, n25, n75, n95, median, mean, std, var]


# def extract_lead_heart_rate(signal, sampling_rate):
#     # extract heart rate for single-lead ECG: may return empty list
#     rpeaks, = ecg.hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)
#     rpeaks, = ecg.correct_rpeaks(signal=signal, rpeaks=rpeaks, sampling_rate=sampling_rate, tol=0.05)
#     _, heartrates = tools.get_heart_rate(beats=rpeaks, sampling_rate=500, smooth=True, size=3)
#     return list(heartrates / 100)  # divided by 100
#
#
# def extract_heart_rates(ecg_data, sampling_rate=500):
#     # extract heart rates using 12-lead since rpeaks can not be detected on some leads
#     heartrates = []
#     for signal in ecg_data.T:
#         lead_heartrates = extract_lead_heart_rate(signal=signal, sampling_rate=sampling_rate)
#         heartrates += lead_heartrates
#     return cal_statistics(heartrates)


def extract_lead_features(signal):
    # extract expert features for single-lead ECGs: statistics, shannon entropy
    lead_features = cal_statistics(signal)  # statistic of signal
    coeffs = pywt.wavedec(signal, 'db10', level=4)  # 小波变换去噪 使用db10小波基4层分解对single信号去噪
    for coeff in coeffs:
        lead_features.append(cal_entropy(coeff))  # shannon entropy of coefficients
        lead_features += cal_statistics(coeff)  # statistics of coefficients
    return lead_features


def extract_features(ecg_data, sapling_rate=500):  # ecg_data: (seq_len, 12)
    # extract expert features for 12-lead ECGs
    # may include heart rates later
    all_features = []
    # comment out below line to extract heart rates
    # all_features += extract_heart_rates(ecg_data, sapling_rate=sampling_rate)
    for signal in ecg_data.T:
        all_features += [extract_lead_features(signal)]
    return all_features


if __name__ == '__main__':
    label_csv = os.path.join(processed_path, 'labels_cinc.csv')
    labels = pd.read_csv(label_csv)
    data_dir = {
        'A': r'D:\learning\科研\数据\PhysioNetChallenge2020\1_PhysioNetChallenge2020_Training_CPSC',
        'Q': r'D:\learning\科研\数据\PhysioNetChallenge2020\2_PhysioNetChallenge2020_Training_2_China 12-Lead ECG Challenge Database',
        'S': r'D:\learning\科研\数据\PhysioNetChallenge2020\4_PhysioNetChallenge2020_Training_PTB',
        'H': r'D:\learning\科研\数据\PhysioNetChallenge2020\5_PhysioNetChallenge2020_Training_PTB-XL',
        'E': r'D:\learning\科研\数据\PhysioNetChallenge2020\6_PhysioNetChallenge2020_Training_E'
    }
    for i, row in tqdm(labels.iterrows()):
        ecg_id = row['ecg_id']
        features_path_file = os.path.join(features_path, str(ecg_id) + '.csv')
        filename = row.ecg_id
        ecg_data, _ = wfdb.rdsamp(os.path.join(data_dir[ecg_id[0:1]], filename))
        features = extract_features(ecg_data)
        df = DataFrame(features, dtype=float).T
        df.to_csv(features_path_file, index=None, header=None)
        # print(features)
