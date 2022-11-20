import ast
import math
import os.path
from collections import Counter
from glob import glob

import torch
import numpy as np
import pandas as pd
import pywt
import wfdb
import wfdb.processing
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.signal import medfilt
from tqdm import tqdm

from process.variables import processed_path, processed_data

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
sub_classes = ['SB', 'SR', 'AFIB', 'ST', 'AF', 'SI', 'SVT', 'AT', 'AVNRT', 'AVRT', 'SAAWR']
super_classes = ['AFIB', 'GSVT', 'SB', 'SR']
class_dic = {
    'SB': 'SB',
    'SR': 'SR',
    'SI': 'SR',
    'AFIB': 'AFIB',
    'AF': 'AFIB',
    'SVT': 'GSVT',
    'AT': 'GSVT',
    'SAAWR': 'GSVT',
    'AVNRT': 'GSVT',
    'ST': 'GSVT',
    'AVRT': 'GSVT'
}
sex_dic = {
    'MALE': 1,
    'FEMALE': 0
}
feature_columns = ['PatientAge', 'Gender', 'VentricularRate', 'AtrialRate', 'QRSDuration', 'QTInterval', 'QTCorrected',
                   'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'TOffset']


def resample_data():
    df = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\Diagnostics.csv'))
    for i, row in tqdm(df.iterrows()):
        file_name = row['FileName'] + '.csv'
        df_data = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', file_name), header=None)
        ecg_data = df_data.values.T
        all_sig_lr = []
        for sig in ecg_data:
            data = wfdb.processing.resample_sig(x=sig, fs=500, fs_target=100)
            all_sig_lr.append(data[0])
        all_sig_lr = np.array(all_sig_lr).T
        df_data_lr = DataFrame(all_sig_lr)
        df_data_lr.to_csv(os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised100', file_name), header=False,
                          index=False)


# 数据清洗
def harmonize_data(features):
    # 对数据进行归一化
    # 首先是归一化函数
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    # 我的数据集有38列，前36列为数值，第37列为时间，第38列为字符串类型，因此只对前36列做数值归一
    for title in feature_columns:
        if title == 'Gender':
            continue
        features[title] = features[[title]].apply(max_min_scaler)
    # 把FEMALE定义为0
    features.loc[features['Gender'] == 'FEMALE', 'Gender'] = 0
    features.loc[features['Gender'] == 'MALE', 'Gender'] = 1
    return features


def extract_R_features_lead2(file_name):
    df = pd.read_csv(os.path.join(processed_data, file_name), header=None)
    ecg_data = np.array(df).T
    lead2_data = ecg_data[1]
    print(1)


def gen_label_muse_csv(label_csv):
    df = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\Diagnostics.csv'))
    features = harmonize_data(df[feature_columns])
    print(features)
    results = []
    for i, row in tqdm(df.iterrows()):
        file_name = row['FileName']
        rhythm = row['Rhythm']
        results.append([file_name, super_classes.index(class_dic[rhythm]), ])
    columns = ['file_name', 'class']
    df_label = pd.DataFrame(data=results, columns=columns)
    df_label[feature_columns] = features
    n = len(df_label)
    folds = np.zeros(n, dtype=np.int8)
    for i in range(10):
        start = int(n * i / 10)
        end = int(n * (i + 1) / 10)
        folds[start:end] = i + 1
    df_label['fold'] = np.random.permutation(folds)
    df_label.to_csv(label_csv, index=None)


if __name__ == '__main__':
    label_csv = os.path.join(r'E:\01_科研\dataset\MUSE', 'labels.csv')
    gen_label_muse_csv(label_csv)
    # resample_data()
