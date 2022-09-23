import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.math_graph import build_graph

device = torch.device('cuda')


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class STGCN(nn.Module):
    """Build the base model.
    n_his: int, size of historical records for training.
    ks: int, kernel size of spatial convolution.
    kt: int, kernel size of temporal convolution.
    blocks: list, channel configs of StConv blocks.
    return: tensor, [batch_size, 1, n_his, n_route].
    """

    def __init__(self, args, blocks):
        super(STGCN, self).__init__()
        n_his, ks, kt, dilation, keep_prob = args.seq_len, 3, 3, 1, 0.3
        # n_his, ks, kt, dilation, keep_prob = args.n_his, args.ks, args.kt, args.dilation, args.keep_prob
        ko = n_his
        self.weight_key = nn.Parameter(torch.zeros(size=(12, 1)))  # 12是节点数
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)  # 初始化K矩阵
        self.weight_query = nn.Parameter(torch.zeros(size=(12, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)  # 初始化Q矩阵
        self.GRU = nn.GRU(input_size=args.seq_len, hidden_size=12)  # 输入维度：序列长度；输出维度：12，因为最后需要得到一个12x12的邻接矩阵
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.st_block = nn.ModuleList()
        # ST-Block
        for channels in blocks:
            self.st_block.append(StConvBlock(ks, kt, channels, dilation, keep_prob))
            ko -= 2 * (kt - 1) * dilation
            # ko>0: kernel size of temporal convolution in the output layer.
        if ko > 1:
            self.out_layer = OutputLayer(blocks[-1][-1], ko, 1)
        else:
            raise ValueError(f'ERROR: kernel size ko must be larger than 1, but recieved {ko}')

    def forward(self, x):
        # x: (batch_size, c_in, time_step, n_route)
        # graph_kernel = build_graph(x)
        graph_kernel, attention = self.latent_correlation_layer(x.permute(0, 2, 1))  # mul_L拉普拉斯邻接矩阵
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        for i in range(len(self.st_block)):
            x = self.st_block[i](x, graph_kernel)
        if self.out_layer:
            x = self.out_layer(x)
        return x

    def latent_correlation_layer(self, x):
        # x:(batch, seqeunce, features)
        # GRU default input:(sequence, batch, features), default batch_first=False
        # However, input is (features, batch, sequence) here
        # torch.Size([140, 32, 12]) sequence(window_size)=12, but is equals 140 here
        xx = x.permute(2, 0, 1).contiguous()
        input, _ = self.GRU(xx)
        # print( " ----GRU output shape: ", input.shape)
        # last state output shape of GRU in doc:(D *num_layers,batch，output_size(self.leadss)
        # However,(sequence,batch，D*Hout(output_size)) when batch_first=False here ???
        # Only all output features is senseful in this situation
        # torch.Size([140，32，140])

        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)  # attention shap 12x12
        degree = torch.sum(attention, dim=1)  # degree shape : torch.Size([12])
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)  # 返回一个以degree为对角线元素的2D矩阵，torch.Size([2，12])

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



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GConv(nn.Module):
    """ Spectral-based graph convolution function.
    x: tensor, [batch_size, c_in, time_step, n_route].
    theta: tensor, [ks*c_in, c_out], trainable kernel parameters.
    ks: int, kernel size of graph convolution.
    c_in: int, size of input channel.
    c_out: int, size of output channel.
    return: tensor, [batch_size, c_out, time_step, n_route].
    """

    #
    def __init__(self, ks, c_in, c_out):
        super(GConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.ks = ks
        self.theta = nn.Linear(c_in, c_out).to(device)

    def forward(self, x, kernel):
        # graph kernel: tensor, [n_route, ks*n_route]
        # kernel = self.graph_kernel
        # time_step, n_route
        _, _, t, n = x.shape
        # x:[batch_size, c_in, time_step, n_route] -> [batch_size, time_step, c_in, n_route]
        x_tmp = x.transpose(1, 2).contiguous()
        kernel = kernel.unsqueeze(0).permute(1, 0, 2, 3)
        # x_ker = x_tmp * ker -> [batch_size, time_step, c_in, ks*n_route]
        x_ker = torch.matmul(x_tmp, torch.sum(kernel, 0))
        # -> [batch_size, time_step, c_in*ks, n_route] -> [batch_size, time_step, n_route, c_in*ks]
        x_ker = x_ker.reshape(-1, t, self.c_in, n).transpose(2, 3)
        # x_ker = x_ker.reshape(-1, t, self.c_in * self.ks, n).transpose(2, 3)
        # -> [batch_size, time_step, n_route, c_out]
        x_fig = self.theta(x_ker)
        # -> [batch_size, c_out, time_step, n_route]
        return x_fig.permute(0, 3, 1, 2).contiguous()


class TemporalConvLayer(nn.Module):
    """ Temporal convolution layer.
    x: tensor, [batch_size, c_in, time_step, n_route].
    kt: int, kernel size of temporal convolution.
    c_in: int, size of input channel.
    c_out: int, size of output channel.
    act_func: str, activation function.
    return: tensor, [batch_size, c_out, time_step-dilation*(kt-1), n_route].
    """

    def __init__(self, kt, c_in, c_out, dilation, act_func='relu'):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.c_out = c_out
        self.c_in = c_in
        self.act_func = act_func
        self.dilation = dilation
        self.conv2d = nn.Conv2d(c_in, c_out, (kt, 1), dilation=(dilation, 1)).to(device)
        self.glu = nn.Conv2d(c_in, 2 * c_out, (kt, 1), dilation=(dilation, 1)).to(device)
        self.modify = nn.Conv2d(c_in, c_out, 1).to(device)

    def forward(self, x):
        batch, _, seq_len, leads = x.shape
        if self.c_in > self.c_out:
            x_input = self.modify(x)
        elif self.c_in < self.c_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            x_input = torch.cat((x, torch.zeros(batch, self.c_out - self.c_in, seq_len, leads, device=device)), 1)
        else:
            x_input = x

        # keep the original input for residual connection.
        x_input = x_input[:, :, self.dilation * (self.kt - 1):, :]

        if self.act_func == 'GLU':
            # gated liner unit
            x_conv = self.glu(x)
            return (x_conv[:, 0:self.c_out, :, :] + x_input) * torch.sigmoid(x_conv[:, -self.c_out:, :, :])
        else:
            x_conv = self.conv2d(x)
            if self.act_func == 'linear':
                return x_conv
            elif self.act_func == 'sigmoid':
                return torch.sigmoid(x_conv)
            elif self.act_func == 'relu':
                return F.relu(x_conv + x_input)
            else:
                raise ValueError(f'ERROR: activation function "{self.act_func}" is not defined.')


class SpatioConvLayer(nn.Module):
    """Spatial graph convolution layer.
    x: tensor, [batch_size, c_in, time_step, n_route].
    ks: int, kernel size of spatial convolution.
    c_in: int, size of input channel.
    c_out: int, size of output channel.
    return: tensor, [batch_size, c_out, time_step, n_route].
    """

    def __init__(self, ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gconv = GConv(ks, c_in, c_out)
        self.modify = nn.Conv2d(c_in, c_out, 1).to(device)

    def forward(self, x, graph_kernel):
        _, _, t, n, = x.shape
        if self.c_in > self.c_out:
            # bottleneck down-sampling
            x_input = self.modify(x)
        elif self.c_in < self.c_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            x_input = torch.cat((x, torch.zeros((x.shape[0], self.c_out - self.c_in, t, n), device=device)), 1)
        else:
            x_input = x

        x_gconv = self.gconv(x, graph_kernel)
        return F.relu(x_gconv + x_input)


class StConvBlock(nn.Module):
    """Spatio-temporal convolutional block, which contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    x: tensor, [batch_size, c_in, time_step, n_route].
    ks: int, kernel size of spatial convolution.
    kt: int, kernel size of temporal convolution.
    channels: list, channel configs of a single st_conv block.
    scope: str, variable scope.
    keep_prob: hyper parameter, prob of dropout.
    act_func: str, activation function.
    return: tensor, [batch_size,  c_out, time_step-, n_route]
    """

    def __init__(self, ks, kt, channels, dilation, keep_prob, act_func='relu'):
        super(StConvBlock, self).__init__()
        c_si, c_t, c_oo = channels
        self.keep_prob = keep_prob
        self.tcLayer1 = TemporalConvLayer(kt, c_si, c_t, dilation, act_func)
        self.scLayer = SpatioConvLayer(ks, c_t, c_t)
        self.tcLayer2 = TemporalConvLayer(kt, c_t, c_oo, dilation)
        # self.layer = nn.Sequential(
        #     TemporalConvLayer(kt, c_si, c_t, dilation, act_func),
        #     SpatioConvLayer(ks, c_t, c_t),
        #     TemporalConvLayer(kt, c_t, c_oo, dilation),
        #     nn.BatchNorm2d(c_oo).to(device)
        # )

    def forward(self, x, graph_kernel):
        x = self.tcLayer1(x)
        x = self.scLayer(x, graph_kernel)
        x = self.tcLayer2(x)
        return F.dropout(x, self.keep_prob)


class OutputLayer(nn.Module):
    def __init__(self, c_in, kt, dilation):
        super(OutputLayer, self).__init__()
        self.layer = nn.Sequential(
            TemporalConvLayer(kt, c_in, c_in, dilation, act_func='GLU'),
            nn.BatchNorm2d(c_in).to(device),
            TemporalConvLayer(1, c_in, c_in, dilation, act_func='sigmoid'),
            nn.Conv2d(c_in, 1, 1).to(device)
        )
        self.fc = nn.Sequential(
            nn.Linear(12, 12),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(12, 5)
        )

    def forward(self, x):
        # x: tensor, shape is (batch_size, c_in, time_step, n_route)
        # Returns: shape is (batch_size, 1, 1, n_route)
        x = self.layer(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
