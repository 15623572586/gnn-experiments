import math

import torch
from matplotlib import pyplot as plt
from torch import nn

import torch
import numpy as np


class GraphLearning(nn.Module):
    def __init__(self, channel=12, width=120):
        super(GraphLearning, self).__init__()
        self.channel = channel
        self.width = width
        self.conv0 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=width, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel * channel, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, leads, 4, seq_len)
        x = x.permute(0, 2, 1, 3)  # (batch, 4, leads, seq_len)
        x = self.relu(self.conv0(x))  # (batch, 1, leads, seq_len)
        x = x.permute(0, 3, 2, 1)  # (batch, seq_len, leads, 1)
        x = self.relu(self.conv1(x))  # (batch, 1, leads, 1)
        x = x.permute(0, 2, 1, 3)
        x = self.relu(self.conv2(x))
        x = x.squeeze()  # (batch, 144)
        x = x.reshape(x.shape[0], self.channel, self.channel)
        x = torch.mean(x, dim=0)
        adj = torch.relu(x)  # 过滤掉负数
        degree = torch.sum(adj, dim=1)
        adj = 0.5 * (adj + adj.T)  # 对称化
        # adj = 0.5 * (adj + adj.permute(1, 0))  # 对称化
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - adj, diagonal_degree_hat))
        return self.cheb_polynomial(laplacian)
        # return adj
        # return self.calculate_laplacian(adj).unsqueeze(1)

    # def calculate_laplacian(self, matrix):
    #     # matrix = matrix + torch.eye(matrix.size(0))
    #     row_sum = matrix.sum(1)  # 度
    #     d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()  # 度开根号 (batch*leads, 1)
    #     # d_inv_sqrt = d_inv_sqrt.view(self.batch, self.leads)  # (batch, leads)
    #     d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    #     d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # 开根号后的度矩阵  (batch, leads, leads)
    #     normalized_laplacian = torch.matmul(torch.matmul(d_mat_inv_sqrt, matrix), d_mat_inv_sqrt)  # D^0.5·A·D^0.5
    #     return normalized_laplacian

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


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=True):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight = nn.init.xavier_uniform_(self.weight)  # xavier初始化，就是论文里的glorot初始化
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        # inputs: (b, 4, 1, 12, features)
        # adj: sparse_matrix (4, 12, 12)
        adj = adj.unsqueeze(1)  # (4, 1, 12, 12)
        support = torch.matmul(self.dropout(inputs), self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class EcgGCNModel(torch.nn.Module):
    def __init__(self, features, num_classes=4, batch_size=32, leads=12, dropout_rate=0.2, device='cuda',
                 gcn_layer_num=2):
        super(EcgGCNModel, self).__init__()
        self.leads = leads
        self.features = features
        self.num_classes = num_classes
        self.batch = batch_size
        self.device = device
        self.gcn_layer_num = gcn_layer_num
        self.graph_learning = GraphLearning(channel=leads, width=features)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.gcn_layers = nn.Sequential()
        for i in range(gcn_layer_num):
            if i == 0:
                self.gcn_layers.append(GraphConvolution(features, features, 0.2))
            elif i < gcn_layer_num-1:
                self.gcn_layers.append(GraphConvolution(features, features, 0.2))
            else:
                self.gcn_layers.append(GraphConvolution(features, features, 0.2))
        # self.gc1 = GraphConvolution(features, 256, 0.2)
        # self.gc2 = GraphConvolution(256, 256, 0.2)
        # self.gc3 = GraphConvolution(256, 256, 0.2)
        self.relu = nn.ReLU()
        # self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        # self.adaptivemaxpool = nn.AdaptiveMaxPool1d(4)
        self.fc = nn.Sequential(
            nn.Linear(features * 12, 512),  # 将这里改成64试试看
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 512),  # 将这里改成64试试看
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        adj = self.graph_learning(x)
        x = x.permute(0, 2, 1, 3).unsqueeze(2)  # (b, 4, 1, 12, features)
        input = x
        for i in range(self.gcn_layer_num):
            x = self.relu(self.gcn_layers[i](x, adj))
            x = x + input  # 残差连接，防止训练一段时间后梯度爆炸，导致loss is nan
        x = x.squeeze().sum(1)
        x = x.reshape(x.size(0), -1)  # res:(batch,leads*4)
        x = self.fc(x)
        return x
