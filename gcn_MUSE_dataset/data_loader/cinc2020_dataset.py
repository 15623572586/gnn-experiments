import os

import torch
from matplotlib import pyplot as plt
from scipy.signal import medfilt
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb
import wfdb.processing

from process.cinc_preprocess import classes
from process.variables import features_path, dataset_path


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000
    return sig


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5:
            sig = scaling(sig)
        if np.random.randn() > 0.5:
            sig = shift(sig)
    return sig


class ECGCincDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, folds, seq_len):
        super(ECGCincDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        self.data_dir = {
            'A': r'D:\learning\科研\数据\PhysioNetChallenge2020\1_PhysioNetChallenge2020_Training_CPSC',
            'Q': r'D:\learning\科研\数据\PhysioNetChallenge2020\2_PhysioNetChallenge2020_Training_2_China 12-Lead ECG Challenge Database',
            'S': r'D:\learning\科研\数据\PhysioNetChallenge2020\4_PhysioNetChallenge2020_Training_PTB',
            'H': r'D:\learning\科研\数据\PhysioNetChallenge2020\5_PhysioNetChallenge2020_Training_PTB-XL',
            'E': r'D:\learning\科研\数据\PhysioNetChallenge2020\6_PhysioNetChallenge2020_Training_E',
            'G': r''
        }
        self.features_dir = features_path
        self.labels = df
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.seq_len = seq_len
        self.n_leads = len(self.leads)
        self.classes = classes
        # self.classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        # self.features = ['age', 'sex', 'weight']
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        ecg_id = row['ecg_id']
        filename = str(ecg_id) + '.csv'
        # # filename = row.filename_hr
        record_path = os.path.join(dataset_path, filename)
        ecg_data_csv = pd.read_csv(record_path, header=None)
        ecg_data = np.array(ecg_data_csv)

        # ecg_data, meta_data = wfdb.rdsamp(record_path)
        # if np.isnan(ecg_data).any():
        ecg_data = np.nan_to_num(ecg_data)  # 有些样本导联缺失（某个导联数据全为零），与处理过后，就会变成nan
        # # 降采样到100Hz
        # all_sig = ecg_data.T
        # all_sig_lr = []
        # fs = meta_data['fs']  # 原始频率
        # for sig in all_sig:
        #     data, _ = wfdb.processing.resample_sig(x=sig, fs=fs, fs_target=100)
        #     all_sig_lr.append(data)
        # ecg_data = filtered_data.T

        ecg_data = transform(ecg_data, self.phase == 'train')
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-self.seq_len:, -self.n_leads:]
        result = np.zeros((self.seq_len, self.n_leads))
        result[-nsteps:, :] = ecg_data
        # if np.any(self.label_dict.get(ecg_id)):
        #     labels = np.any(self.label_dict.get(ecg_id))
        # else:
        labels = row[self.classes].to_numpy(dtype=np.float32)
            # self.label_dict[ecg_id] = labels
        # z_score归一化
        data_mean = np.mean(result, axis=0)
        data_std = np.std(result, axis=0)
        data_std = [1 if i == 0 else i for i in data_std]
        result = (result - data_mean) / data_std
        label_index = np.argmax(labels)
        x, y = torch.from_numpy(result.transpose()).float(), torch.tensor(label_index)  # ecg数据
        # features = pd.read_csv(os.path.join(features_path, str(ecg_id) + '.csv')).to_numpy(dtype=np.float32)

        return x, y, torch.tensor([1], dtype=torch.float)

    def __len__(self):
        return len(self.labels)


