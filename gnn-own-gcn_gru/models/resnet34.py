import torch
import torch.nn as nn


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
    def __init__(self, block, layers, seq_len, input_channels=12, inplanes=12, num_classes=9):
        super(ResNet1d, self).__init__()
        self.inplanes = inplanes
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=1, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(BasicBlock1d, 64, layers[2], stride=1)
        self.layer4 = self._make_layer(BasicBlock1d, 12, layers[3], stride=1)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(int(seq_len / 2))
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(int(seq_len / 2))
        self.fc = nn.Linear(12 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):  # x: 32 x 12 x 500
        x = self.conv1(x)  # x: 32x64x250
        x = self.bn1(x)    # x: 32x64x250
        x = self.relu(x)  # x: 32x64x250
        x = self.maxpool(x)  # x: 32x64x125
        x = self.layer1(x)  # x: 32x64x125
        x = self.layer2(x)  # x: 32x128x63
        x = self.layer3(x)  # x: 32x128x32
        x = self.layer4(x)  # x: 32x128x16
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=2)
        # x = x.view(x.size()[0], -1)
        return x


def resnet18(**kwargs):
    model = ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):  # **kwargs接收带键值对的参数
    model = ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)
    return model
