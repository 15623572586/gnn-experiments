import torch
from torch import nn

ma = torch.randn([64, 12, 4, 100])


class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(in_dim, out_dim)
        self.linear_right = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


gru = nn.GRU(input_size=100, hidden_size=100, num_layers=1, batch_first=True)

gru_out = torch.FloatTensor()
for i in range(4):
    out, hidden = gru(ma[:, :, i, :])
    out = out.unsqueeze(2)
    gru_out = torch.cat((gru_out, out), 2)
    print(out)
# w1 = ma[:, :, 0, :]
# w2 = ma[:, :, 1, :]
# w3 = ma[:, :, 2, :]
# w4 = ma[:, :, 3, :]
#
#
# glu = GLU(100, 100)
# mao = glu(ma)
# w1o = glu(w1)
# w2o = glu(w2)
# w3o = glu(w3)
# w4o = glu(w4)
# print(w1)
# print(w1o)
#
# magruo = gru(w1)
#
# print(magruo)
