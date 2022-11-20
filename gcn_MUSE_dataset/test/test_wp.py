import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import torch
import seaborn as sns
from scipy.signal import welch

from test.test_psd import get_psd_values

# 需要分析的3个频段
iter_freqs = [
    {'name': 'P', 'fmin': 0.5, 'fmax': 20},
    {'name': 'QRS', 'fmin': 0.5, 'fmax': 38},
    {'name': 'T', 'fmin': 0.5, 'fmax': 8},
    {'name': 'all', 'fmin': 0.5, 'fmax': 40},
]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 连续小波变换##########
# totalscal小波的尺度，对应频谱分析结果也就是分析几个（totalscal-1）频谱
def TimeFrequencyCWT(data, fs, totalscal, wavelet='cgau8'):
    # 采样数据的时间维度
    t = np.arange(data.shape[0]) / fs
    # 中心频率
    wcf = pywt.central_frequency(wavelet=wavelet)
    # 计算对应频率的小波尺度
    cparam = 2 * wcf * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    # 连续小波变换
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavelet, 1.0 / fs)
    # 绘图
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t, data)
    plt.xlabel(u"time(s)")
    plt.title(u"Time spectrum")
    plt.subplot(212)
    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)
    plt.show()


# 小波包变换-重构造分析不同频段的特征(注意maxlevel，如果太小可能会导致部分波段分析不到)
def TimeFrequencyWP(data, fs, wavelet, maxlevel=8):
    # 小波包变换这里的采样频率为500，如果maxlevel太小部分波段分析不到
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # 计算maxlevel最小频段的带宽
    freqBand = fs / (2 ** maxlevel)
    # 根据实际情况计算频谱对应关系，这里要注意系数的顺序
    # 绘图显示
    fig, axes = plt.subplots(len(iter_freqs) + 1, 1, figsize=(20, 7), sharex=True, sharey=False)
    # 绘制原始数据
    axes[0].plot(data)
    axes[0].set_title('原始数据')
    new_data_list = {'all': data}
    for iter in range(len(iter_freqs)):
        # 构造空的小波包
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
        for i in range(len(freqTree)):
            str = freqTree[i]
            freq_data = wp[str].data  # 频段数据
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax:
                # 给新构造的小波包参数赋值
                new_wp[freqTree[i]] = freq_data
        # 绘制对应频率的数据
        new_data = new_wp.reconstruct(update=True)
        new_data_list[iter_freqs[iter]['name']] = new_data
        axes[iter + 1].plot(new_data)
        # 设置图名
        axes[iter + 1].set_title(iter_freqs[iter]['name'])
    plt.show()
    return new_data_list


########小波包计算3个频段的能量分布
def WPEnergy(data, fs, wavelet, maxlevel=6):
    # 小波包分解
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # 计算maxlevel最小频段的带宽
    freqBand = fs / (2 ** maxlevel)
    # 定义能量数组
    energy = []
    data_freqs = []  # 各个频段的序列
    de = np.zeros([12, len(iter_freqs)])
    # 循环遍历计算4个频段对应的能量
    for iter in range(len(iter_freqs)):
        iterEnergy = 0.0
        data_freq = []
        for i in range(len(freqTree)):
            str = freqTree[i]
            freq_data = wp[str].data  # 频段数据
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax:
                # 计算对应频段的累加和
                iterEnergy += pow(np.linalg.norm(freq_data, ord=None), 2)
                data_freq += freq_data.tolist()
        data_freqs.append(data_freq)
        # 保存3个频段对应的能量和
        energy.append(iterEnergy)
    # 绘制能量分布图
    plt.plot([xLabel['name'] for xLabel in iter_freqs], energy, lw=0, marker='o')
    plt.title('能量分布')
    plt.show()


# data_dir = os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', 'MUSE_20180113_171327_27000.csv')  # AFIB
# data_dir = os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', 'MUSE_20180209_172046_21000.csv')  # SR
data_dir = os.path.join(r'E:\01_科研\dataset\MUSE\ECGDataDenoised', 'MUSE_20180112_073319_29000.csv')  # SB
fs = 500

sns.set(font_scale=1.2)
if __name__ == '__main__':
    df = pd.read_csv(data_dir, header=None)
    ecg_data = np.array(df.values).T
    lead = 0
    new_data_list = TimeFrequencyWP(ecg_data[lead], fs=fs, wavelet='db8', maxlevel=9)

    freqs, psd = welch(new_data_list['T'], fs=fs, nperseg=4*fs, return_onesided=True)
    # freqs, psd = get_psd_values(new_data_list['QRS'], fs)
    idx_k = np.logical_and(freqs >= 0.5, freqs <= 20)
    plt.figure(figsize=(20, 10))
    plt.plot(freqs, psd, lw=2, color='k')
    # plt.fill_between(freqs, psd, where=idx_k, color='darkorange')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (uV^2 / Hz)')
    plt.xlim([0, 40])
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Welch's periodogram")
    sns.despine()
    plt.show()
    # WPEnergy(ecg_data[lead], fs=fs, wavelet='db8', maxlevel=9)
