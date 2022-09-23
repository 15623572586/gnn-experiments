import numpy as np
import torch

# a = torch.ones(32, 998, 32, 12)
# b = torch.ones(4, 12, 12)
# b = torch.sum(b, 0)
# c = torch.matmul(a, b)
# print(c.size())
#
# vec1 = torch.ones(1, 3, 3)
# vec2 = torch.ones(2, 3, 4)
# d = torch.matmul(vec1, vec2)
# print(d.size())

# s = np.array([0.1,0.2,0.3])
# sy = np.array([0.2,0.1,0.4])
# print(sy>s)
# nums = [1, 2, 3, 4, 5, 6, 9, 9]
# print(s.tolist().index(max(s)))
# print nums.index(1)

# 数据
# a = np.array([
#                [2, 4, 4],
#                [4, 16, 12],
#                [4, 12, 10],
#           ])
# a = torch.Tensor(a)
# s = torch.softmax(a, dim=1)
# print(s)
#
# x = 6.3379e-02+4.6831e-01+4.6831e-01
# print(x)
# res = torch.Tensor()
# hn = torch.randn(11, 12, 1000)  # 假设隐藏层
# hn = torch.mean(hn, dim=0).unsqueeze(0)
# print(hn)
# hb = torch.randn(2, 3, 4)
# cat = torch.cat((res, hb), dim=-1)
# print(cat)


class_dict = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
labels = [0] * 5
clazz = 'NORM'
if clazz in class_dict:
    print(class_dict.index(clazz))
    labels[class_dict.index(clazz)] = 1
print(labels)