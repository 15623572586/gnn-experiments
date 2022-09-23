import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb


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


class ECGPtbXLDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, folds, seq_len):
        super(ECGPtbXLDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        self.labels = df
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.seq_len = seq_len
        self.n_leads = len(self.leads)
        self.classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        ecg_id = row['ecg_id']
        filename = row.filename_lr
        ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, filename))
        ecg_data = transform(ecg_data, self.phase == 'train')
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-self.seq_len:, -self.n_leads:]
        result = np.zeros((self.seq_len, self.n_leads))
        result[-nsteps:, :] = ecg_data
        if self.label_dict.get(ecg_id):
            labels = self.label_dict.get(ecg_id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[ecg_id] = labels
        # z_score归一化
        data_mean = np.mean(result, axis=0)
        data_std = np.std(result, axis=0)
        data_std = [1 if i == 0 else i for i in data_std]
        result = (result - data_mean) / data_std
        label_index = np.argmax(labels)
        x, y = torch.from_numpy(result.transpose()).float(), torch.tensor(label_index)
        return x, y

    def __len__(self):
        return len(self.labels)


