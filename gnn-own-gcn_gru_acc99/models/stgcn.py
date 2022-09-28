import math

import torch
from torch import nn
import torch.nn.functional as f
from models.resnet34 import ResNet1d, BasicBlock1d
from models.tcn import TemporalConvNet


class CnnFeatures(nn.Module):
    def __init__(self):
        super(CnnFeatures, self).__init__()
        cnn = []
        # 7x1  --  7x64
        # in_channels 输入信号的通道，即1维向量
        # out_channels 卷积输出信号通道
        # kernel_size 卷积核大小 kernel_size * in_channels
        # stride 步长
        # padding 填充
        cnn.append(nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2, stride=1,
                             padding=1))  # 输入通道为1，即1维向量，卷积核大小为kernel_size *in_channels
        cnn.append(nn.ReLU())  # ReLu激活函数：一部分神经元输出为0，增加输出的矩阵稀疏性，使得在训练时更容易发现其中的规律
        cnn.append(nn.BatchNorm1d(64))  # 保持输入数据的均值和方差恒定，使后面网络不用不停调参来适应输入变化，实现网络各层解耦
        cnn.append(nn.Dropout(0.3))  #
        cnn.append(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1))
        cnn.append(nn.ReLU())
        cnn.append(nn.BatchNorm1d(64))
        self.cnn = nn.Sequential(*cnn)

        self.cnn_l = nn.Sequential(
            nn.Linear(5 * 64, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.5),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.cnn_l(x)


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
        return self.calculate_laplacian(adj)

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
        # self.gru = nn.GRU(input_size=step_len, hidden_size=step_len, num_layers=gru_num_layers, batch_first=True)
        self.tcn = TemporalConvNet(num_inputs=12, num_channels=[64, 128, 128, 12])
        self.block_out = nn.Sequential(
            nn.Linear(self.step_len, self.step_len),
            nn.LeakyReLU(),
            nn.BatchNorm1d(leads),
            nn.Dropout(p=0.5),
            nn.Linear(self.step_len, self.step_len)
        )

    def forward(self, x, adj):
        batch, _, _ = x.shape
        x = torch.matmul(adj, x)
        tcn_out = self.tcn(x)
        gcn_out = torch.matmul(tcn_out, self.weight)  # (adj · x) · weight
        block_out = self.block_out(gcn_out)
        # block_out += x
        return block_out


class EcgGCNTCNModel(torch.nn.Module):
    def __init__(self, seq_len, step_len, num_classes, batch_size, leads, gru_num_layers, dropout_rate=0.5,
                 device='cuda'):
        super(EcgGCNTCNModel, self).__init__()
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
        self.resnet = ResNet1d(BasicBlock1d, [3, 4, 6, 3], seq_len=seq_len)
        # self.cnn = CnnFeatures()
        self.tcn = TemporalConvNet(num_inputs=12, num_channels=[32, 64, 64, 12], kernel_size=3)
        self.fc = nn.Sequential(
            nn.Linear(12 * self.seq_len, self.seq_len),  # 将这里改成64试试看
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.seq_len, self.seq_len),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.seq_len, self.num_classes),
        )
        # self.fc_out = nn.Sequential(
        #     nn.Linear(10, 64),  # 将这里改成64试试看
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(64, 64),
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(64, self.num_classes),
        # )

    def forward(self, x):
        # x: (batch, leads, seq_len)
        n_step = self.seq_len // self.step_len  #
        res = torch.Tensor().to(self.device)
        res_out = self.resnet(x)
        for i in range(n_step):
            start_time = i * self.step_len
            end_time = (i + 1) * self.step_len
            xx = x[:, :, start_time:end_time]
            # part 1:
            mul_L = self.graph_learning(xx)
            res1 = self.stock_block(xx, mul_L)  # res:(batch,leads, seq_len)
            res = torch.cat((res, res1), dim=-1)
        res += res_out
        res = res.view(res.size(0), -1)  # res:(batch,leads*seq_len)
        res = self.fc(res)

        return res
