from itertools import cycle

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from process.ptbxl_preprocess import class_dict

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


# class_dict = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
# labels = [0] * 5
# clazz = 'NORM'
# if clazz in class_dict:
#     print(class_dict.index(clazz))
#     labels[class_dict.index(clazz)] = 1
# print(labels)

# print(isinstance(class_dict, torch.tensor))
# if type(class_dict) != 'tensor':
#     torch.tensor(class_dict)

pred_list = [1, 1, 4, 1, 4, 0, 2, 2, 3, 3, 0]
target_list = [1, 2, 4, 1, 4, 0, 3, 2, 3, 3, 0]
target_one_hot = label_binarize(target_list, classes=np.arange(5))
pred_scores = [
    [0.1, 0.3, 0.4, 0.1, 0.1],
    [0.1, 0.3, 0.4, 0.1, 0.1],
    [0.1, 0.3, 0.4, 0.1, 0.1],
    [0.1, 0.3, 0.4, 0.1, 0.1],
    [0.1, 0.3, 0.4, 0.1, 0.1],
    [0.1, 0.3, 0.4, 0.1, 0.1],
    [0.1, 0.3, 0.4, 0.1, 0.1],
    [0.1, 0.3, 0.4, 0.1, 0.1],
    [0.1, 0.3, 0.4, 0.1, 0.1],
    [0.1, 0.3, 0.4, 0.1, 0.1],
    [0.1, 0.3, 0.4, 0.1, 0.1],
]
conf_matrix = torch.zeros(5, 5)  # 混淆矩阵
for p, t in zip(pred_list, target_list):
    conf_matrix[p, t] += 1
conf_matrix = np.array(conf_matrix)
# print(conf_matric)
# print(f1_score(y_true=target_list, y_pred=pred_list, average='weighted'))
#
# arrays = []
# arr1 = torch.tensor([1, 2, 3]).tolist()
# arr2 = torch.tensor([2, 2, 3]).tolist()
# arrays = arrays + arr2
# arrays = arrays + arr1
# print(arrays)

def roc_auc():
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # 计算每一类roc_auc
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_true=np.array(target_one_hot)[:, i],
                                      y_score=np.array(pred_scores)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # macro（方法二）
    fpr["macro"], tpr["macro"], _ = roc_curve(np.array(target_one_hot).ravel(), np.array(pred_scores).ravel())
    roc_auc["macro"] = roc_auc_score(y_true=target_list, y_score=torch.softmax(torch.tensor(pred_scores), dim=1).tolist(),
                                     average='macro',
                                     multi_class='ovr')

    print(fpr)
    print(tpr)
    print(roc_auc)
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_dict[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc="lower right")
    plt.savefig('../output/roc_auc/roc_auc.svg')
    plt.show()
def plot_confusion_matrix(confusion_mat):
    # 画混淆矩阵图，配色风格使用cm.Greens
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Greens)

    # 显示colorbar
    plt.colorbar()

    # 使用annotate在图中显示混淆矩阵的数据
    for x in range(len(confusion_mat)):
        for y in range(len(confusion_mat)):
            plt.annotate(confusion_mat[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.title('Confusion Matrix')  # 图标title
    plt.xlabel('True label')  # 坐标轴标签
    plt.ylabel('Predicted label')  # 坐标轴标签

    tick_marks = np.arange(5)
    plt.xticks(tick_marks, class_dict)
    plt.yticks(tick_marks, class_dict)
    plt.savefig('../output/conf_matric/conf_matric.svg')
    plt.show()

if __name__ == '__main__':
    plot_confusion_matrix(confusion_mat=conf_matrix)