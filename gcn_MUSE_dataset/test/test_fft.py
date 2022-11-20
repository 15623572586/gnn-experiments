import numpy as np
from scipy.fftpack import fft, ifft  # fft快速傅里叶变换、ifft其逆变换
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import torch

# 振幅谱推演代码
"""mpl.rcParams['font.sans-serif']=['SimHei'] #显示中文   rcParams通过rc参数可以修改默认的属性，包括窗体大小、 每英寸的点数、线条宽度、颜色、样式、坐标轴、坐标和网络属性、文本、字体等。
mpl.rcParams['axes.unicode_minus']=False #显示负号
X=np.linspace(0,1,1400)#采样点应选择最大频率2.5倍，这里最大是600赫兹，所以选择1400
y=7*np.sin(2*np.pi*200*X)+5*np.sin(2*np.pi*400*X)+3*np.sin(2*np.pi*600*X)#200Hz、400Hz、600Hz三个正玄波

'''#1.原始波形
plt.figure()
plt.plot(X,y)
plt.title("原始波型")

#2.原始波形，前50组
plt.figure()
plt.plot(X[0:50],y[0:50])
plt.title("原始波型(前50组)")
plt.show()'''

#3.快速傅里叶变换
fft_y=fft(y)
print(len(fft_y))
print(fft_y[0:5])
N=1400
X=np.arange(N)#频率个数
abs_y=np.abs(fft_y)#取复数的绝对值，即复数的模（双边频谱）
angle_y=np.angle(fft_y)#取复数的角度

'''plt.figure()
plt.plot(X,abs_y)
plt.title("双边振幅谱（未归一化）")

plt.figure()
plt.plot(X,angle_y)
plt.title("双边相位谱（未归一化）")
plt.show()'''

#4.将振幅进行归一化与取半处理
#4.1归一化处理
normalization_y=abs_y/N
plt.figure()
plt.plot(X,normalization_y,'g')
plt.title("双边频谱（归一化）",)
#plt.show()
#4.2取半处理
half_X=X[range(int(N/2))]#取一半区间
normalization_half_y=normalization_y[range(int(N/2))]#由于对称性，也取一半区间
plt.figure()
plt.plot(half_X,normalization_half_y,'b')
plt.title("单边频谱（归一化）")#单边频谱即最终的振幅谱
plt.show()"""

# 振幅谱完整代码
#
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
# mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
# X = np.linspace(0, 1, 1400)  # 采样点应选择最大频率2.5倍，这里最大是600赫兹，所以选择1400
# y = 7 * np.sin(2 * np.pi * 200 * X) + 5 * np.sin(2 * np.pi * 400 * X) + 3 * np.sin(
#     2 * np.pi * 600 * X)  # 200Hz、400Hz、600Hz三个正玄波
# fft_y = fft(y)  # 快速傅里叶
# N = 1400
# X = np.arange(N)  # 频率个数
# half_X = X[range(int(N / 2))]  # 取一半区间
# abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模（双边频谱）
# angle_y = np.angle(fft_y)  # 取复数的角度
# normalization_y = abs_y / N  # 归一化处理
# normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，也取一半区间
# plt.subplot(2, 3, 1)
# plt.plot(X, y)
# plt.title("原始波型")
#
# plt.subplot(2, 3, 2)
# plt.plot(X, fft_y)
# plt.title("双边振幅谱(未求绝对值)")
#
# plt.subplot(2, 3, 3)
# plt.plot(X, abs_y)
# plt.title("双边振幅谱(未归一化)")
#
# plt.subplot(2, 3, 4)
# plt.plot(X, angle_y)
# plt.title("双边相位谱(未归一化)")
#
# plt.subplot(2, 3, 5)
# plt.plot(X, normalization_y)
# plt.title("双边振幅谱(归一化)")
#
# plt.subplot(2, 3, 6)
# plt.plot(half_X, normalization_half_y)
# plt.title("单边振幅谱(归一化)")
# plt.show()


import torch

input = torch.randn(32, 12, 64)
# 旧版pytorch。参数normalized对这篇文章的结论没有影响，加上只是跟开头同步
# output_fft_old = torch.rfft(input, signal_ndim=2, normalized=False, onesided=False)
# output_ifft_old = torch.irfft(output_fft_old , signal_ndim=2, normalized=False, onesided=False)
# 新版
output_fft_new = torch.fft.fft2(input, dim=(-2, -1))
# output_fft_new_2dim = torch.stack((output_fft_new.real, output_fft_new.imag), -1)
output_ifft_new = torch.fft.ifft2(torch.complex(output_fft_new.real, output_fft_new.imag), dim=(-2, -1))    # 如果运行了torch.stack()
# output_ifft_new = torch.fft.ifft2(output_fft_new_2dim, dim=(-2, -1))  # 没有运行torch.stack()

output_ifft_new = torch.abs(output_ifft_new)
print(output_ifft_new)

