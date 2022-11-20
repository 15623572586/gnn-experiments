import torch
import torch.nn as nn


class GraphLearning(nn.Module):
    def __init__(self, channel, height, width):
        super(GraphLearning, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width
        self.conv1 = nn.Conv2d(in_channels=width, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel * channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.conv2(x)
        x = x.squeeze()  # (batch, 144)
        x = x.reshape(x.shape[0], self.channel, self.channel)
        return x


if __name__ == '__main__':
    batch = 32
    C = 12
    H = 1
    W = 100
    x = torch.rand((batch, C, H, W))
    graph_learning = GraphLearning(C, H, W)
    out = graph_learning(x)
    print(out.shape)  # (4, 256, 225, 225)
