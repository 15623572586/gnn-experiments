import math

import torch
from matplotlib import pyplot as plt
from torch import nn

import torch
import numpy as np
import math


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
    def __init__(self, batch, n_leads, features, device='cuda'):
        super(GraphLearning, self).__init__()
        self.batch = batch
        self.leads = n_leads
        self.features = features
        self.device = device
        self.gru = nn.GRU(input_size=features, hidden_size=features, num_layers=1, batch_first=True)
        self.attention = SelfAttention(2, features, n_leads, 0.5)

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


def drawing(ecg_data):
    ecg_data = np.array(ecg_data[0])
    plt.figure(figsize=(20, 5))
    plt.plot(ecg_data[0])
    plt.legend(['Before', 'After'])
    plt.show()


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # xavier初始化，就是论文里的glorot初始化
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        support = torch.matmul(self.dropout(inputs), self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class EcgGCNModel(torch.nn.Module):
    def __init__(self, features, num_classes=4, batch_size=32, leads=12, dropout_rate=0.5, device='cuda'):
        super(EcgGCNModel, self).__init__()
        self.leads = leads
        self.features = features
        self.num_classes = num_classes
        self.batch = batch_size
        self.device = device
        self.graph_learning = GraphLearning(batch=batch_size, n_leads=self.leads, features=features,
                                            device=device)

        self.gc1 = GraphConvolution(features, 256, 0.5)
        self.gc2 = GraphConvolution(256, 64, 0.5)
        self.relu = nn.ReLU()
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        # self.adaptivemaxpool = nn.AdaptiveMaxPool1d(4)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),  # 将这里改成64试试看
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 128),  # 将这里改成64试试看
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        # x: (batch, leads, seq_len)
        adj = self.graph_learning(x)
        x = self.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        x = x.permute(0, 2, 1)
        x = self.adaptiveavgpool(x)
        x = x.squeeze()
        x = x.view(x.size(0), -1)  # res:(batch,leads*4)
        x = self.fc(x)
        return x
