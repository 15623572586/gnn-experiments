import numpy as np
import torch
import torch.nn as nn

from models.gcn_gru import SelfAttention


class BasicBlock1d(nn.Module):
    expansion = 1

    # inplanes 提供给残差块的通道数
    # planes block的输出通道
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 1维残差网络
class ResNet1d(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=12, inplanes=64):
        super(ResNet1d, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, _):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        # return self.adaptivemaxpool(x)
        return self.fc(x)


# class Net_A(nn.Module):
#     def __init__(self, seq_len, step_len, num_classes, batch_size, leads, gru_num_layers, dropout_rate=0.5,
#                  device='cuda'):
#         super(Net_A, self).__init__()
#         self.gru_num_layers = gru_num_layers
#         self.batch = batch_size
#         self.resnet = ResNet1d(BasicBlock1d, [3, 4, 6, 3], num_classes=9)
#         self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=gru_num_layers, batch_first=True)
#         self.attention = SelfAttention(num_attention_heads=2, input_size=128, hidden_size=128)
#         self.fc = nn.Sequential(
#             nn.Linear(12 * 128, 128),  # 将这里改成64试试看
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(128, 9),
#         )
#
#     def forward(self, x, features=None):
#         res_out = self.resnet(x)
#         hidden = torch.zeros(self.gru_num_layers, self.batch, 128).to('cuda')
#         gru_out, _ = self.gru(res_out, hidden)
#         # att_out = self.attention(gru_out)
#         res = gru_out.reshape(gru_out.size(0), -1)  # res:(batch,leads*seq_len)
#         fc = self.fc(res)
#         return fc


def resnet34(**kwargs):  # **kwargs接收带键值对的参数
    model = ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)
    return model
