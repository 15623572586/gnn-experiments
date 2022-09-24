import os.path
from itertools import cycle

import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
import wfdb
from sklearn.preprocessing import label_binarize

from process.ptbxl_preprocess import class_dict


def split_data(seed=42):
    folds = range(1, 11)  # 不包含11
    folds = np.random.RandomState(seed).permutation(folds)  # 根据seec随机排序
    return folds[:9], folds[9:]


def confusion_matrix(pred_list, target_list, args):
    conf_matrix = torch.zeros(5, 5)  # 混淆矩阵
    for p, t in zip(pred_list, target_list):
        conf_matrix[p, t] += 1
    return conf_matrix


def performance(pred_list, target_list, pred_scores, args):
    conf_matrix = confusion_matrix(pred_list, target_list, args)  # 混淆矩阵
    precision = precision_score(y_true=target_list, y_pred=pred_list, average='micro')
    recall = recall_score(y_true=target_list, y_pred=pred_list, average='micro')  # 召回率
    f1_value = f1_score(y_pred=pred_list, y_true=target_list, average='micro')
    acc_value = accuracy_score(y_pred=pred_list, y_true=target_list)

    # auc_value = roc_auc_score(y_true=target_list, y_score=torch.softmax(torch.tensor(pred_scores), dim=1).tolist(),
    #                           multi_class='ovr')
    target_one_hot = label_binarize(target_list, classes=np.arange(5))
    # fpr, tpr, _ = roc_curve(y_true=np.array(target_one_hot).ravel(), y_score=np.array(pred_scores).ravel())

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
    roc_auc["macro"] = roc_auc_score(y_true=target_list,
                                     y_score=torch.softmax(torch.tensor(pred_scores), dim=1).tolist(),
                                     average='macro',
                                     multi_class='ovr')
    performance_dic = {
        'precision': precision,
        'recall': recall,
        'f1_value': f1_value,
        'acc_value': acc_value,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }
    return conf_matrix, performance_dic


def drawing_confusion_matric(conf_matric, filename):
    # 画混淆矩阵图，配色风格使用cm.Greens
    plt.imshow(conf_matric, interpolation='nearest', cmap=plt.cm.Greens)
    # 显示colorbar
    plt.colorbar()
    # 使用annotate在图中显示混淆矩阵的数据
    for x in range(len(conf_matric)):
        for y in range(len(conf_matric)):
            plt.annotate(conf_matric[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.title('Confusion Matrix')  # 图标title
    plt.xlabel('True label')  # 坐标轴标签
    plt.ylabel('Predicted label')  # 坐标轴标签

    tick_marks = np.arange(5)
    plt.xticks(tick_marks, class_dict)
    plt.yticks(tick_marks, class_dict)
    plt.savefig(filename)
    # plt.show()


def drawing_roc_auc(data, filename):
    # Plot all ROC curves
    fpr = data['fpr']
    tpr = data['tpr']
    roc_auc = data['roc_auc']
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
    plt.title('5-calsses ROC')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    # plt.show()


# 曲线图
def drawing_logs(log_file):
    res = pandas.read_csv(os.path.join('../output/logs', log_file + '.csv'))
    train_loss = np.array(res['train_loss'])
    val_loss = np.array(res['test_loss'])

    train_f1 = np.array(res['train_acc'])
    val_f1 = np.array(res['val_acc'])
    lr = np.array(res['lr'])

    epoch = [i for i in range(0, train_f1.size)]

    # fig, ax = plt.subplots()  # 创建图实例
    plt.plot(epoch, train_loss, label='train_loss')
    plt.plot(epoch, val_loss, label='val_loss')
    plt.plot(epoch, train_f1, label='train_acc')
    plt.plot(epoch, val_f1, label='val_acc')
    plt.plot(epoch, lr, label='lr')
    plt.xlabel('epoch')  # 设置x轴名称 x label
    plt.ylabel('loss and f1')  # 设置y轴名称 y label
    plt.title('Simple Plot')  # 设置图名为Simple Plot
    plt.legend()
    plt.savefig(os.path.join('../output/loss', log_file+'.svg'))
    plt.show()


if __name__ == '__main__':
    drawing_logs('2022-09-23')
