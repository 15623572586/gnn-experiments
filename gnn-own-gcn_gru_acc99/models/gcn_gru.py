import math

import torch
from torch import nn


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
        return adj


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
        block_out = self.block_out(gcn_out)
        block_out += x
        return block_out, hidden


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
        self.graph_learning = GraphLearning(batch=batch_size, n_leads=self.leads, seq_len=seq_len, step_len=step_len,
                                            device=device)
        self.stock_block = StockBlockLayer(seq_len=seq_len, leads=leads, step_len=step_len, batch_size=batch_size,
                                           gru_num_layers=gru_num_layers)
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
        for i in range(n_step):
            start_time = i * self.step_len
            end_time = (i + 1) * self.step_len
            xx = x[:, :, start_time:end_time]
            # part 1:
            mul_L = self.graph_learning(xx)
            res1, hidden = self.stock_block(xx, mul_L, hidden)  # res:(batch,leads, seq_len)
            res2, hidden = self.stock_block(res1, mul_L, hidden)  # res:(batch, leads, seq_len)
            res3, hidden = self.stock_block(res2, mul_L, hidden)  # res:(batch, leads, seq_len)
            res = torch.cat((res, res3), dim=-1)
        res += x
        res = res.view(res.size(0), -1)  # res:(batch,leads*seq_len)
        return self.fc(res)
