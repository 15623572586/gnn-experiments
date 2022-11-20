import os.path

import matplotlib.pyplot as plt
import pandas as pd
import pywt
import torch
import numpy as np
from scipy.signal import welch
from torch.fft import fft, rfft
import seaborn as sns

sns.set(font_scale=1.2)

# data_dir = os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', 'MUSE_20180113_171327_27000.csv')  # AFIB
# data_dir = os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', 'MUSE_20180209_172046_21000.csv')  # SR
data_dir = os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', 'MUSE_20180112_073319_29000.csv')  # SB

fs = 500
len_sample = 500 * 10


def get_pywt(x):
    # f_values, fft_values = get_fft_values(x, nfft=4096, fs=500)
    # plt.figure()
    # plt.plot(fft_values[0:200])
    # plt.show()
    wp = pywt.WaveletPacket(data=x, wavelet='db6', mode='symmetric', maxlevel=10)  # 选用db1小波，分解层数为3
    # 根据频段频率（freq）进行排序
    # level1_node = [node.path for node in wp.get_level(1, 'freq')]
    # level2_node = [node.path for node in wp.get_level(2, 'freq')]
    # level3_node = [node.path for node in wp.get_level(3, 'freq')]
    # level4_node = [node.path for node in wp.get_level(4, 'freq')]
    # print('第一层小波包节点:', level1_node)  # 第一层小波包节点
    # print('第二层小波包节点:', level2_node)  # 第二层小波包节点
    # print('第三层小波包节点:', level3_node)  # 第三层小波包节点
    # print('第四层小波包节点:', level4_node)  # 第三层小波包节点
    # aaaa = wp['aaaa'].data
    # print(aaaa)
    # print('aaa的长度:', aaaa.shape[0])
    # print('x的长度:', x.shape[0])
    # print('理论上第4层每个分解系数的长度:', x.shape[0] / 2048)

    n = 10
    re = []  # 第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
    # 第n层能量特征
    energy = []
    for i in re:
        energy.append(pow(np.linalg.norm(i, ord=None), 2))
    # for i in range(len(energy)):
    #     print('最后一层第{0}个小波的能量为：{1}'.format(i, energy[i]))

    # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
    plt.figure(figsize=(40, 7), dpi=80)
    # 再创建一个规格为 1 x 1 的子图
    # plt.subplot(1, 1, 1)
    # 柱子总数
    N = 100
    values = energy[0:100]
    # 包含每个柱子下标的序列
    index = np.arange(N)
    # 柱子的宽度
    width = 0.3
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p2 = plt.bar(index, values, width, label="num", color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('clusters')
    # 设置纵轴标签
    plt.ylabel('Wavel energy')
    # 添加标题
    plt.title('Cluster Distribution')
    # 添加纵横轴的刻度
    plt.xticks(index, (str(i) for i in range(7, 107)))
    # plt.yticks(np.arange(0, 10000, 10))
    # 添加图例
    plt.legend(loc="upper right")
    plt.show()


def get_fft_values(y_values, nfft, fs):
    f_values = np.linspace(0.0, fs / 2.0, nfft // 2)
    fft_values_ = np.fft.fft(y_values)
    fft_values = 2.0 / nfft * np.abs(fft_values_[0:nfft // 2])
    return f_values, fft_values


def get_psd_values(x, f_s):
    # Define sampling frequency and time vector
    fs = f_s
    time = np.arange(len(x)) / fs
    win = 1 * fs  # Define window length (4 seconds)
    freqs, psd = welch(x, fs=f_s, nperseg=win)
    return freqs, psd


# 需要分析的3个频段
iter_freqs = [
    {'name': 'P', 'fmin': 0.5, 'fmax': 20},
    {'name': 'QRS', 'fmin': 0.5, 'fmax': 38},
    {'name': 'T', 'fmin': 0.5, 'fmax': 8},
    {'name': 'all', 'fmin': 0.5, 'fmax': 40},
]

if __name__ == '__main__':
    df = pd.read_csv(data_dir, header=None)
    ecg_data = np.array(df.values)

    # z_score归一化
    # data_mean = np.mean(ecg_data, axis=0)
    # data_std = np.std(ecg_data, axis=0)
    # data_std = [1 if i == 0 else i for i in data_std]
    # ecg_data = (ecg_data - data_mean) / data_std
    ecg_data = ecg_data.T
    colors = ['darkorange', 'cornflowerblue', 'blueviolet', 'skyblue']
    # Define delta lower and upper limits
    low, high = 0.5, 8
    # Find intersecting values in frequency vector
    idx = 2
    for i in range(2, 12):
        freqs, psd = get_psd_values(ecg_data[i], fs)
        idx_k = np.logical_and(freqs >= iter_freqs[idx]['fmin'], freqs <= iter_freqs[idx]['fmax'])
        # idx_qrs = np.logical_and(freqs >= low, freqs <= high)
        # idx_t = np.logical_and(freqs >= low, freqs <= high)
        # idx_all = np.logical_and(freqs >= low, freqs <= high)
        # plt.plot(psd[i][:100])
        # Plot the power spectral density and fill the delta area
        plt.figure(figsize=(20, 10))
        plt.plot(freqs, psd, lw=2, color='k')
        plt.fill_between(freqs, psd, where=idx_k, color=colors[idx])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (uV^2 / Hz)')
        plt.xlim([0, 60])
        plt.ylim([0, psd.max() * 1.1])
        plt.title("Welch's periodogram")
        sns.despine()
        break
    plt.show()

    # for i in range(12):
    #     get_pywt(ecg_data[i])
    #     break
