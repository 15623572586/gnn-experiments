import math

import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as f

from models.gcn import GraphConvolution
from models.resnet34 import ResNet1d, BasicBlock1d


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


import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F


#
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob=0.5):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(0.5)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class GraphLearning(nn.Module):
    def __init__(self, batch, n_leads, seq_len, step_len, device='cuda'):
        super(GraphLearning, self).__init__()
        self.batch = batch
        self.leads = n_leads
        self.seq_len = seq_len
        self.device = device
        self.gru = nn.GRU(input_size=seq_len, hidden_size=seq_len, num_layers=1, batch_first=True)
        self.attention = SelfAttention(2, seq_len, n_leads, 0.5)

    def reset_parameters(self):  # 参数随机初始化函数
        stdv = 1. / math.sqrt(self.weight.size(1))  # stdv=1/根号(out_features)
        self.weight.data.uniform_(-stdv, stdv)  # weight在区间(-stdv, stdv)之间均匀分布随机初始化
        # if self.bias is not None:  # 变量是否不是None
        #     self.bias.data.uniform_(-stdv, stdv)  # bias均匀分布随机初始化

    def forward(self, x):
        # input:(batch,leads,step_len)
        x, _ = self.gru(x)
        att_out = self.attention(x)
        att_out = torch.mean(att_out, dim=0)
        adj = torch.relu(att_out)  # 过滤掉负数
        adj = 0.5 * (adj + adj.permute(1, 0))  # 注意力输出矩阵是非对称的，转成对称
        return self.calculate_laplacian(adj)

    def calculate_laplacian(self, matrix):
        # matrix = matrix + torch.eye(matrix.size(0))
        row_sum = matrix.sum(1)  # 度
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()  # 度开根号 (batch*leads, 1)
        # d_inv_sqrt = d_inv_sqrt.view(self.batch, self.leads)  # (batch, leads)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # 开根号后的度矩阵  (batch, leads, leads)
        normalized_laplacian = torch.matmul(torch.matmul(d_mat_inv_sqrt, matrix), d_mat_inv_sqrt)  # D^0.5·A·D^0.5
        return normalized_laplacian


# class GraphLearning_bak(nn.Module):
#     def __init__(self, batch, n_leads, seq_len, step_len, device='cuda'):
#         super(GraphLearning, self).__init__()
#         self.batch = batch
#         self.leads = n_leads
#         self.seq_len = seq_len
#         self.device = device
#         self.gru = nn.GRU(input_size=step_len, hidden_size=step_len, num_layers=1, batch_first=True)
#         self.key_weight = nn.Parameter(torch.Tensor(batch, step_len, n_leads)).to(device)  # 邻接矩阵权重向量
#         self.query_weight = nn.Parameter(torch.Tensor(batch, step_len, n_leads)).to(device)  # 邻接矩阵权重向量
#         self.value_weight = nn.Parameter(torch.Tensor(batch, step_len, n_leads)).to(device)  # 邻接矩阵权重向量
#         nn.init.xavier_normal_(self.key_weight)
#         nn.init.xavier_normal_(self.query_weight)
#         nn.init.xavier_normal_(self.value_weight)
#         # self.reset_parameters()
#
#     def reset_parameters(self):  # 参数随机初始化函数
#         stdv = 1. / math.sqrt(self.weight.size(1))  # stdv=1/根号(out_features)
#         self.weight.data.uniform_(-stdv, stdv)  # weight在区间(-stdv, stdv)之间均匀分布随机初始化
#         # if self.bias is not None:  # 变量是否不是None
#         #     self.bias.data.uniform_(-stdv, stdv)  # bias均匀分布随机初始化
#
#     def forward(self, x):
#         # input:(batch,leads,step_len)
#         x, _ = self.gru(x)
#         # 计算 k,q,v
#         batch, _, _ = x.shape
#         key = torch.matmul(x, self.key_weight)
#         query = torch.matmul(x, self.query_weight)
#         value = torch.matmul(x, self.value_weight)
#         # 计算注意力得分
#         attention_score = torch.matmul(query, key.permute(0, 2, 1))  # query x key^T
#         attention_score = torch.softmax(attention_score, dim=1)  # 按行进行归一化
#         adj = torch.matmul(attention_score, value)  # 输出注意力矩阵
#         adj = torch.relu(adj)  # 过滤掉负数
#         adj = 0.5 * (adj + adj.permute(0, 2, 1))  # 注意力输出矩阵是非对称的，转成对称
#         return self.calculate_laplacian(adj)
#
#     def calculate_laplacian(self, matrix):
#         # matrix = matrix + torch.eye(matrix.size(0))
#         row_sum = matrix.sum(1)  # 度
#         d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()  # 度开根号 (batch*leads, 1)
#         d_inv_sqrt = d_inv_sqrt.view(self.batch, self.leads)  # (batch, leads)
#         d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
#         d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # 开根号后的度矩阵  (batch, leads, leads)
#         normalized_laplacian = torch.matmul(torch.matmul(d_mat_inv_sqrt, matrix), d_mat_inv_sqrt)  # D^0.5·A·D^0.5
#         return normalized_laplacian


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
        self.gcn = GraphConvolution(step_len, step_len, 0.5)
        # batch_first=True:x(batch, seq, feature)
        self.gru = nn.GRU(input_size=step_len, hidden_size=step_len, num_layers=gru_num_layers, batch_first=True)
        # self.resnet = ResNet1d(BasicBlock1d, [3, 4, 6, 3], seq_len=step_len)
        self.block_out = nn.Sequential(
            nn.Linear(self.step_len, self.step_len),
            nn.LeakyReLU(),
            nn.BatchNorm1d(leads),
            nn.Dropout(p=0.5),
            nn.Linear(self.step_len, self.step_len)
        )

    # def forward(self, x, adj, hidden_in):
    #     batch, _, _ = x.shape
    #     # self.drawing(x.cpu().detach().numpy())
    #     x = torch.matmul(adj, x)
    #     # self.drawing(x.cpu().detach().numpy())
    #     gru_out, hidden = self.gru(x, hidden_in)
    #     # self.drawing(gru_out.cpu().detach().numpy())
    #     gcn_out = torch.matmul(gru_out, self.weight)  # (adj · x) · weight
    #     # self.drawing(gcn_out.cpu().detach().numpy())
    #     block_out = self.block_out(gcn_out)
    #     # block_out += x
    #     return block_out, hidden

    def forward(self, x, adj, hidden_real, hidden_imag):
        gcn_out = self.gcn(x, adj)
        # 频域学习
        ffted = torch.fft.fft2(gcn_out, dim=(-2, -1))
        ffted_real = ffted.real
        ffted_imag = ffted.imag
        real_gru, hidden_real = self.gru(ffted_real, hidden_real)
        imag_gru, hidden_imag = self.gru(ffted_imag, hidden_imag)
        iffted = torch.fft.ifft2(torch.complex(real_gru, imag_gru), dim=(-2, -1))
        iffted_abs_norm = torch.abs(iffted) / self.step_len
        # 残差
        # res_out = self.resnet(gcn_out)
        # gru_res_out = res_out + iffted_abs_norm
        gcn_out = self.gcn(iffted_abs_norm, adj)
        # gcn_out = self.gcn(iffted_abs_norm, adj)
        # gru_out, hidden = self.gru(gcn_out, hidden_real, hidden_imag )
        block_out = self.block_out(gcn_out)
        return block_out, hidden_real, hidden_imag

    def drawing(self, ecg_data):
        ecg_data = np.array(ecg_data[0])
        plt.figure(figsize=(20, 5))
        plt.plot(ecg_data[0])
        plt.legend(['Before', 'After'])
        plt.show()


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
        self.attention = SelfAttention(num_attention_heads=2, input_size=seq_len, hidden_size=seq_len)
        self.resnet = ResNet1d(BasicBlock1d, [3, 4, 6, 3], seq_len=seq_len)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(12 * 4, 64),  # 将这里改成64试试看
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 64),  # 将这里改成64试试看
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, x, features=None):
        # x: (batch, leads, seq_len)
        n_step = self.seq_len // self.step_len  #
        res = torch.Tensor().to(self.device)
        hidden_real, hidden_imag = torch.zeros(self.gru_num_layers, self.batch, self.step_len).to(
            self.device), torch.zeros(self.gru_num_layers, self.batch, self.step_len).to(
            self.device)  # (D∗num_layers,N,Hout)（是否双向乘以层数，batch size大小，输出维度大小）
        res_out = self.resnet(x)
        mul_L = self.graph_learning(x)
        for i in range(n_step):
            start_time = i * self.step_len
            end_time = (i + 1) * self.step_len
            xx = x[:, :, start_time:end_time]
            # part 1:
            # mul_L = self.graph_learning(xx)
            res1, hidden_real, hidden_imag = self.stock_block(xx, mul_L, hidden_real, hidden_imag)  # res:(batch,leads, seq_len)
            # res2, hidden_real, hidden_imag = self.stock_block(res1, mul_L, hidden_real, hidden_imag)  # res:(batch, leads, seq_len)
            # res3, hidden_real, hidden_imag = self.stock_block(res2, mul_L, hidden_real, hidden_imag)  # res:(batch, leads, seq_len)
            res = torch.cat((res, res1 + xx), dim=-1)
        x1 = self.adaptivemaxpool(res)
        x2 = self.adaptiveavgpool(res)
        res = torch.cat((x1, x2), dim=2)
        res = torch.cat((res, res_out), dim=2)
        res = res.view(res.size(0), -1)  # res:(batch,leads*seq_len)
        res = self.fc(res)
        return res
