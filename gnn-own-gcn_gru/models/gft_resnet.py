import sys

import numpy as np
import torch
from scipy.stats import stats
import scipy.sparse as sp
from torch import nn
import torch.nn.functional as F

# 门控单元paper: Language Modeling with Gated Convolutional Networks
# 作用:1．序列深度建模;2．减轻梯度弥散，加速收敛
from models.resnet import BasicBlock1d
from models.resnet34 import resnet34, ResNet1d


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, seq_len, leads):
        super(StockBlockLayer, self).__init__()
        self.glu_len = 3
        self.seq_len = seq_len
        self.leads = leads
        # self.stack_cnt = stack_cnt
        # self.multi = multi_layer
        self.weight = nn.Parameter(torch.Tensor(1, self.leads, self.leads))  # 考虑将igft参数变成[1,12,12]，这样可以大幅减少参数，降低复杂度
        # self.weight = nn.Parameter(torch.Tensor(1, self.seq_len, self.seq_len))  # 考虑将igft参数变成[1,12,12]，这样可以大幅减少参数，降低复杂度
        nn.init.xavier_normal_(self.weight)
        self.block_out = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.LeakyReLU(),
            nn.BatchNorm1d(leads),
            nn.Dropout(p=0.5),
            nn.Linear(self.seq_len, self.seq_len)
        )
        self.resnet = ResNet1d(BasicBlock1d, [3, 4, 6, 3], seq_len)

        # self.forecast = nn.Linear(self.seq_len, self.seq_len)
        # self.forecast_result = nn.Linear(self.seq_len, self.seq_len)

        # if self.stack_cnt == 0:
        #     self.backcast = nn.Linear(self.seq_len * self.multi, self.seq_len)
        # self.backcast_short_cut = nn.Linear(self.seq_len, self.seq_len)
        #
        # self.relu = nn.ReLU()
        # self.GLUs = nn.ModuleList()
        # self.output_channel = 4 * self.multi
        # for i in range(self.glu_len):
        #     if i == 0:
        #         self.GLUs.append(GLU(self.seq_len * 4, self.seq_len * self.output_channel))
        #         self.GLUs.append(GLU(self.seq_len * 4, self.seq_len * self.output_channel))
        #     elif i == 1:
        #         self.GLUs.append(GLU(self.seq_len * self.output_channel, self.seq_len * self.output_channel))
        #         self.GLUs.append(GLU(self.seq_len * self.output_channel, self.seq_len * self.output_channel))
        #     else:
        #         self.GLUs.append(GLU(self.seq_len * self.output_channel, self.seq_len * self.output_channel))
        #         self.GLUs.append(GLU(self.seq_len * self.output_channel, self.seq_len * self.output_channel))

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, seq_len = input.size()
        input = input.view(batch_size, -1, node_cnt, seq_len)
        # 旧版
        ffted = torch.rfft(input, 1, onesided=False)
        # 1.12版本
        # ffted = torch.fft.fft(input)
        # ffted = torch.stack((ffted.real, ffted.imag), -1)  # 新版torch1.12中实部和虚部不再是二维数组，而是以复数的形式存在，需要用stack把他们堆叠
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        for i in range(self.glu_len):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        # 旧版用的
        seq_len_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        iffted = torch.irfft(seq_len_as_inner, 1, onesided=False)
        # 新版1.12用的
        # iffted = torch.fft.ifft(torch.complex(real, img), dim=-1)
        # return torch.abs(iffted)  # 复数转float
        # return iffted.float()  # 复数转float
        return iffted  # 旧版

    def forward(self, x, mul_L):
        x = x.unsqueeze(1)
        gfted = torch.matmul(mul_L, x)
        # gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        gconv_input = gfted.permute(0, 1, 3, 2)
        igfted = torch.matmul(gconv_input, self.weight)
        igfted = igfted.permute(0, 1, 3, 2)
        igfted = torch.sum(igfted, dim=1)  # 32x1x12x len
        resnet_out = self.resnet(igfted.squeeze())
        block_out = self.block_out(igfted.squeeze())
        return block_out


class EcgModel(torch.nn.Module):
    def __init__(self, seq_len, num_classes, dropout_rate=0.5, leaky_rate=0.2):
        super(EcgModel, self).__init__()
        self.leads = 12
        self.seq_len = seq_len
        self.num_classes = num_classes
        # self.stack_cnt = 1
        # self.multi_layer = 1
        # self-attention
        # 对GRU的最后一个隐藏状态R使用self-attention的方式计算邻接矩阵
        self.weight_key = nn.Parameter(torch.zeros(size=(self.leads, 1)))  # 12是节点数
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)  # 初始化K矩阵
        self.weight_query = nn.Parameter(torch.zeros(size=(self.leads, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)  # 初始化Q矩阵
        self.leaky_relu = nn.LeakyReLU(leaky_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.GRU = nn.GRU(input_size=self.seq_len, hidden_size=self.leads)  # 输入维度：序列长度；输出维度：12，因为最后需要得到一个12x12的邻接矩阵
        self.resnet34 = ResNet1d(BasicBlock1d, [3, 4, 6, 3], seq_len=seq_len)
        # self.resnet18 = ResNet1d(BasicBlock1d, [2, 2, 2, 2], seq_len=seq_len)
        self.stock_block1 = StockBlockLayer(self.seq_len, self.leads)
        self.stock_block2 = StockBlockLayer(self.seq_len, self.leads)
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.leads, self.seq_len),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.seq_len, self.num_classes)
        )

    def forward(self, x):
        # part 1:
        mul_L, attention = self.latent_correlation_layer(x.permute(0, 2, 1))  # mul_L拉普拉斯邻接矩阵
        # mul_L = self.build_laplacian_matrix(x)  # 皮尔逊相关系数计算邻接矩阵
        resnet_x = self.resnet34(x)
        # part 2:
        # x: (batch,leads, seq_len)
        res1 = self.stock_block1(x, mul_L)  # res:(batch,leads, seq_len)
        res2 = self.stock_block2(resnet_x + res1, mul_L)  # res:(batch,leads, seq_len)
        res = res2
        res = res.view(res.size(0), -1)  # res:(batch,leads*seq_len)
        res = self.fc(res)
        return res

    def latent_correlation_layer(self, x):
        xx = x.permute(2, 0, 1).contiguous()
        input, _ = self.GRU(xx)

        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)  # 32，12，12
        attention = torch.mean(attention, dim=0)  # attention shap 12x12
        degree = torch.sum(attention, dim=1)  # degree shape : torch.Size([12])
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)  # 返回一个以degree为对角线元素的2D矩阵，torch.Size([12，12])

        diagonal_degree_hat = torch.diag(
            1 / (torch.sqrt(degree) + 1e-7))  # diagonal_degree_hat shape: torch.Size([12,12])
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention,
                                              diagonal_degree_hat))  # laplacian shape: torch.Size([12，12)
        mul_L = self.cheb_polynomial(laplacian)  # mul_L shape: torch.Size([4,12,12])
        return mul_L, attention

    def self_graph_attention(self, input):
        # input shape here: (batch,sequence,output_size)
        input = input.permute(0, 2, 1).contiguous()
        # after trans:(batch,output_size, sequence)
        # this is why input output ?
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        # key shape : torch. Size([32，140，1])
        query = torch.matmul(input, self.weight_query)
        # torch.repeat当参数有三个时:(通道数的重复倍数，行的重复倍数，列的重复倍数)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        # data shape: torch. Size([32，140 *140，1])
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leaky_relu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        # data shape: torch. Size([32，140 *140，1])
        return attention

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian
