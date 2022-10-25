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


def gen_label_muse_csv(label_csv):
    df = pd.read_csv(os.path.join(r'E:\01_科研\dataset\MUSE\Diagnostics.csv'))
    results = []
    for i, row in tqdm(df.iterrows()):
        file_name = row['FileName']
        rhythm = row['Rhythm']
        results.append([file_name, super_classes.index(class_dic[rhythm])])

    df_label = pd.DataFrame(data=results, columns=['file_name', 'class'])
    n = len(df_label)
    folds = np.zeros(n, dtype=np.int8)
    for i in range(10):
        start = int(n * i / 10)
        end = int(n * (i + 1) / 10)
        folds[start:end] = i + 1
    df_label['fold'] = np.random.permutation(folds)
    df_label.to_csv(label_csv, index=None)


if __name__ == '__main__':
    # label_csv = os.path.join(r'E:\01_科研\dataset\MUSE', 'labels.csv')
    # gen_label_muse_csv(label_csv)
    resample_data()
