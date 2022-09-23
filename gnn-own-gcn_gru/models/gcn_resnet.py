import math

import numpy as np
import torch
import scipy.sparse as sp
from torch import nn
from models.resnet import BasicBlock1d
from models.resnet34 import ResNet1d


class GraphLearning(nn.Module):
    def __init__(self, batch, n_leads, device='cuda'):
        super(GraphLearning, self).__init__()
        self.batch = batch
        self.leads = n_leads
        self.device = device
        self.K = 3
        self.weight = nn.Parameter(torch.Tensor(1, n_leads, n_leads)).to(device)  # 邻接矩阵权重向量
        # nn.init.xavier_normal_(self.weight)
        self.reset_parameters()
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()

    def reset_parameters(self):  # 参数随机初始化函数
        stdv = 1. / math.sqrt(self.weight.size(1))  # stdv=1/根号(out_features)
        self.weight.data.uniform_(-stdv, stdv)  # weight在区间(-stdv, stdv)之间均匀分布随机初始化
        # if self.bias is not None:  # 变量是否不是None
        #     self.bias.data.uniform_(-stdv, stdv)  # bias均匀分布随机初始化

    def forward(self, x):
        # 计算向量之间的欧氏距离
        dists = []  # (batch, leads, leads)
        for m in x:
            adj = torch.norm(m[:, None] - m, dim=2, p=2)   # 计算行向量之间的距离
            adj = adj.cpu().numpy()
            adj = adj + sp.eye(adj.shape[0])  # 添加自环 得 A'
            degree = np.array(np.sum(adj, axis=1)).reshape(-1)  # 度
            d_matrix = np.diag(1. / np.sqrt(degree))  # 求D^-1/2
            laplacian = np.dot(np.dot(d_matrix, adj), d_matrix)  # 计算D^-1/2 · A' · D^-1/2
            dists.append(laplacian)
        dists = torch.tensor(np.array(dists), dtype=torch.float, device=self.device)
        weight_adj = torch.matmul(self.weight, dists)
        weight_adj = self.relu(weight_adj)
        weight_adj = nn.functional.log_softmax(weight_adj, dim=-1)
        weight_adj_mean = torch.mean(weight_adj, dim=0)
        # mul_L = self.cheb_polynomial_K(weight_adj_mean)
        mul_L = self.cheb_polynomial_4(weight_adj_mean)
        return mul_L

    def cheb_polynomial_K(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        # multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.matmul(laplacian, multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2]
        return multi_order_laplacian

    def cheb_polynomial_4(self, laplacian):
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


class StockBlockLayer(nn.Module):
    def __init__(self, seq_len, leads):
        super(StockBlockLayer, self).__init__()
        self.seq_len = seq_len
        self.leads = leads
        self.weight = nn.Parameter(torch.Tensor(1, seq_len, seq_len))
        nn.init.xavier_normal_(self.weight)
        self.resnet = ResNet1d(BasicBlock1d, [3, 4, 6, 3], seq_len=seq_len)
        self.block_out = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.LeakyReLU(),
            nn.BatchNorm1d(leads),
            nn.Dropout(p=0.5),
            nn.Linear(self.seq_len, self.seq_len)
        )

    def forward(self, x, adj):
        x = x.unsqueeze(1)
        x = torch.matmul(adj, x)
        x = torch.sum(x, dim=1).squeeze()
        resnet_out = self.resnet(x)
        gcn_out = torch.matmul(resnet_out, self.weight)  # (adj · x) · weight
        block_out = self.block_out(gcn_out.squeeze())
        return block_out


class EcgGCNModel(torch.nn.Module):
    def __init__(self, seq_len, num_classes, batch_size, dropout_rate=0.5, device='cuda'):
        super(EcgGCNModel, self).__init__()
        self.leads = 12
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.graph_learning = GraphLearning(batch=batch_size, n_leads=12, device=device)
        # self.resnet34 = ResNet1d(BasicBlock1d, [3, 4, 6, 3], seq_len=seq_len)
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
        mul_L = self.graph_learning(x)
        # mul_L = self.graph_learning(x)
        # resnet_x = self.resnet34(x)
        # part 2:
        # x: (batch,leads, seq_len)
        res1 = self.stock_block1(x, mul_L)  # res:(batch,leads, seq_len)
        res2 = self.stock_block2(x - res1, mul_L)  # res:(batch,leads, seq_len)
        res = res2 + res1
        res = res.view(res.size(0), -1)  # res:(batch,leads*seq_len)

        res = self.fc(res)
        return res

