import math

import numpy as np
import torch
import scipy.sparse as sp
from torch import nn
from models.resnet import BasicBlock1d
from models.resnet34 import ResNet1d


class GraphLearning(nn.Module):
    def __init__(self, batch, n_leads, seq_len, step_len, device='cuda'):
        super(GraphLearning, self).__init__()
        self.batch = batch
        self.leads = n_leads
        self.seq_len = seq_len
        self.device = device
        self.gru = nn.GRU(input_size=step_len, hidden_size=step_len, num_layers=1, batch_first=True)
        self.key_weight = nn.Parameter(torch.Tensor(batch, step_len, n_leads)).to(device)  # 邻接矩阵权重向量
        self.query_weight = nn.Parameter(torch.Tensor(batch, step_len, n_leads)).to(device)  # 邻接矩阵权重向量
        self.value_weight = nn.Parameter(torch.Tensor(batch, step_len, n_leads)).to(device)  # 邻接矩阵权重向量
        nn.init.xavier_normal_(self.key_weight)
        nn.init.xavier_normal_(self.query_weight)
        nn.init.xavier_normal_(self.value_weight)
        # self.reset_parameters()

    def reset_parameters(self):  # 参数随机初始化函数
        stdv = 1. / math.sqrt(self.weight.size(1))  # stdv=1/根号(out_features)
        self.weight.data.uniform_(-stdv, stdv)  # weight在区间(-stdv, stdv)之间均匀分布随机初始化
        # if self.bias is not None:  # 变量是否不是None
        #     self.bias.data.uniform_(-stdv, stdv)  # bias均匀分布随机初始化

    def forward(self, x):
        # input:(batch,leads,step_len)
        x, _ = self.gru(x)
        # 计算 k,q,v
        batch, _, _ = x.shape
        key = torch.matmul(x, self.key_weight)
        query = torch.matmul(x, self.query_weight)
        value = torch.matmul(x, self.value_weight)
        # 计算注意力得分
        attention_score = torch.matmul(query, key.permute(0, 2, 1))  # query x key^T
        attention_score = torch.softmax(attention_score, dim=1)  # 按行进行归一化
        adj = torch.matmul(attention_score, value)  # 输出注意力矩阵
        adj = torch.relu(adj)  # 过滤掉负数
        adj = 0.5 * (adj + adj.permute(0, 2, 1))  # 注意力输出矩阵是非对称的，转成对称
        # normalized_laplacian = self.calculate_laplacian(adj)
        return adj

    def calculate_laplacian(self, matrix):
        # matrix = matrix + torch.eye(matrix.size(0))
        row_sum = matrix.sum(1)  # 度
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()  # 度开根号 (batch*leads, 1)
        d_inv_sqrt = d_inv_sqrt.view(self.batch, self.leads)  # (batch, leads)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # 开根号后的度矩阵  (batch, leads, leads)
        normalized_laplacian = torch.matmul(torch.matmul(d_mat_inv_sqrt, matrix), d_mat_inv_sqrt)  # D^0.5·A·D^0.5
        return normalized_laplacian


class StockBlockLayer(nn.Module):
    def __init__(self, seq_len, step_len, leads, batch_size, gru_num_layers):
        super(StockBlockLayer, self).__init__()
        self.seq_len = seq_len
        self.step_len = step_len
        self.leads = leads
        self.batch_size = batch_size
        self.weight = nn.Parameter(torch.Tensor(batch_size, step_len, step_len))
        # nn.init.xavier_normal_(self.weight)
        nn.init.xavier_uniform_(self.weight)
        # batch_first=True:x(batch, seq, feature)
        self.gru = nn.GRU(input_size=step_len, hidden_size=step_len, num_layers=gru_num_layers, batch_first=True)
        self.resnet = ResNet1d(BasicBlock1d, [3, 4, 6, 3], seq_len=step_len)
        self.block_out = nn.Sequential(
            nn.Linear(self.step_len, self.step_len),
            nn.LeakyReLU(),
            nn.BatchNorm1d(leads),
            nn.Dropout(p=0.5),
            nn.Linear(self.step_len, self.step_len)
        )

    def forward(self, x, adj, hidden_in):
        batch, _, _ = x.shape
        x = torch.matmul(adj, x)
        gru_out, hidden = self.gru(x, hidden_in)
        gcn_out = torch.matmul(gru_out, self.weight)  # (adj · x) · weight
        # gru_out, hidden = self.gru(gcn_out, hidden)
        block_out = self.block_out(gcn_out)
        # gcn_out = torch.relu(gcn_out)
        return block_out, hidden
    # def forward(self, x, adj):
    #     batch, _, _ = x.shape
    #     res_out = self.resnet(x)
    #     gcn_out = torch.matmul(torch.matmul(adj, res_out), self.weight)  # (adj · x) · weight
    #     stock_out = torch.tanh(gcn_out)
    #     return stock_out + x


class EcgGCNGRUModel(torch.nn.Module):
    def __init__(self, seq_len, step_len, num_classes, batch_size, leads, gru_num_layers, dropout_rate=0.5,
                 device='cuda'):
        super(EcgGCNGRUModel, self).__init__()
        self.leads = leads
        self.seq_len = seq_len
        self.step_len = step_len
        self.num_classes = num_classes
        self.batch = batch_size
        self.device = device
        self.gru_num_layers = gru_num_layers
        self.resnet = ResNet1d(BasicBlock1d, [3, 4, 6, 3], seq_len=seq_len)
        self.graph_learning = GraphLearning(batch=batch_size, n_leads=self.leads, seq_len=seq_len, step_len=step_len,
                                            device=device)
        self.stock_block = StockBlockLayer(seq_len=seq_len, leads=leads, step_len=step_len, batch_size=batch_size,
                                           gru_num_layers=gru_num_layers)

        self.end_conv_1 = nn.Conv2d(in_channels=leads,
                                    out_channels=1,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.fc = nn.Sequential(
            nn.Linear(12 * self.seq_len, self.seq_len),  # 将这里改成64试试看
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.seq_len, self.seq_len),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.seq_len, self.num_classes),
        )

    def forward(self, x):
        # x: (batch, leads, seq_len)
        n_step = self.seq_len // self.step_len  #
        res = torch.Tensor().to(self.device)
        hidden = torch.zeros(self.gru_num_layers, self.batch, self.step_len).to(
            self.device)  # (D∗num_layers,N,Hout)（是否双向乘以层数，batch size大小，输出维度大小）
        resnet_out = self.resnet(x)
        for i in range(n_step):
            start_time = i * self.step_len
            end_time = (i + 1) * self.step_len
            xx = x[:, :, start_time:end_time]
            # part 1:
            mul_L = self.graph_learning(xx)
            res1, hidden = self.stock_block(xx, mul_L, hidden)  # res:(batch,leads, seq_len)
            res2, hidden = self.stock_block(xx - res1, mul_L, hidden)  # res:(batch, leads, seq_len)
            res = torch.cat((res, res2 + res1), dim=-1)
        res += resnet_out
        # resnet_out = self.resnet(x)
        # for i in range(n_step):
        #     start_time = i * self.step_len
        #     end_time = (i + 1) * self.step_len
        #     xx = x[:, :, start_time:end_time]
        #     # part 1:
        #     mul_L = self.graph_learning(xx)
        #     res1 = self.stock_block(xx, mul_L)  # res:(batch,leads, seq_len)
        #     # resnet_out1 = self.resnet(res1)
        #     res2 = self.stock_block(res1, mul_L)  # res:(batch,leads, seq_len)
        #     res3 = self.stock_block(res2, mul_L)  # res:(batch,leads, seq_len)
        #     # resnet_out2 = self.resnet(res2)
        #     res = torch.cat((res, res1 + res2 + res3), dim=-1)
        # res += resnet_out
        # resnet_out = self.resnet(res)  # resnet增加网络深度





        # part 1:
        # mul_L = self.graph_learning(x)
        # # part 2:
        # hidden_0 = torch.zeros(2 * 1, self.batch, self.seq_len).to('cuda')  # (D∗num_layers,N,Hout)（是否双向乘以层数，batch size大小，输出维度大小）
        # res1, hidden = self.stock_block1(x, mul_L, hidden_0)  # res:(batch,leads, seq_len)
        # res2, _ = self.stock_block2(x - res1, mul_L, hidden)  # res:(batch,leads, seq_len)
        # res = res2 + res1

        res = res.view(res.size(0), -1)  # res:(batch,leads*seq_len)

        return self.fc(res)
