import os
import numpy as np
import math
import torch
import pandas as pd
import scipy.io as sio
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft

# data_dir = os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', 'MUSE_20180113_171327_27000.csv')  # AFIB
data_dir = os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', 'MUSE_20180209_172046_21000.csv')  # SR
# data_dir = os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', 'MUSE_20180112_073319_29000.csv')  # SB

fs = 500


def DE_PSD(data, stft_para):
    """
    compute DE and PSD
    --------
    input:  data [n*m]          n electrodes, m time points
            stft_para.stftn     frequency domain sampling rate 频域采样率
            stft_para.fStart    start frequency of each frequency band 各频段起始频率
            stft_para.fEnd      end frequency of each frequency band 每个频段的结束频率
            stft_para.window    window length of each sample point(seconds) 每个样本点的窗口长度(秒)
            stft_para.fs        original frequency  # 原始信号频率
    output: psd,DE [n*l*k]        n electrodes, l windows, k frequency bands
    """
    # initialize the parameters
    STFTN = stft_para['stftn']
    fStart = stft_para['fStart']
    fEnd = stft_para['fEnd']
    fs = stft_para['fs']
    window = stft_para['window']

    WindowPoints = fs * window

    fStartNum = np.zeros([len(fStart)], dtype=int)
    fEndNum = np.zeros([len(fEnd)], dtype=int)
    for i in range(0, len(stft_para['fStart'])):
        fStartNum[i] = int(fStart[i] / fs * STFTN)
        fEndNum[i] = int(fEnd[i] / fs * STFTN)

    # print(fStartNum[0],fEndNum[0])
    n = data.shape[0]
    m = data.shape[1]

    # print(m,n,l)
    psd = np.zeros([n, len(fStart)])
    de = np.zeros([n, len(fStart)])
    # Hanning window
    Hlength = int(window * fs)
    # Hwindow=hanning(Hlength)
    Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength + 1)) for n in range(1, Hlength + 1)])

    WindowPoints = fs * window
    dataNow = data[0:n]
    for j in range(0, n):
        temp = dataNow[j]
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata, STFTN)
        magFFTdata = abs(FFTdata[0:int(STFTN / 2)])
        for p in range(0, len(fStart)):
            E = 0
            # E_log = 0
            for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                E = E + magFFTdata[p0] * magFFTdata[p0]
            #    E_log = E_log + log2(magFFTdata(p0)*magFFTdata(p0)+1)
            E = E / (fEndNum[p] - fStartNum[p] + 1)
            psd[j][p] = E
            de[j][p] = math.log(100 * E, 2)
            # de(j,i,p)=log2((1+E)^4)

    return psd, de


if __name__ == '__main__':
    df = pd.read_csv(data_dir, header=None)
    ecg_data = np.array(df.values)

    # z_score归一化
    data_mean = np.mean(ecg_data, axis=0)
    data_std = np.std(ecg_data, axis=0)
    data_std = [1 if i == 0 else i for i in data_std]
    ecg_data = (ecg_data - data_mean) / data_std
    ecg_data = ecg_data.T
    stft_para = {
        'stftn': 0.1,
        'fStart': [0.5, 2, 0.5],
        'fEnd': [35, 20, 10],
        'fs': 500,
        'window': 1
    }
    psd, de = DE_PSD(ecg_data, stft_para)

    plt.figure()
    for i in range(len(psd)):
        plt.plot(psd[i][0])
    plt.show()

