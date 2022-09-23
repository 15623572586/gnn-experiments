from matplotlib import pyplot as plt
import numpy as np
import pandas
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import wfdb


def split_data(seed=42):
    folds = range(1, 11)  # 不包含11
    folds = np.random.RandomState(seed).permutation(folds)  # 根据seec随机排序
    return folds[:9], folds[8:9], folds[9:]


def prepare_input(ecg_file: str):
    if ecg_file.endswith('.mat'):
        ecg_file = ecg_file[:-4]
    ecg_data, _ = wfdb.rdsamp(ecg_file)
    nsteps, nleads = ecg_data.shape
    ecg_data = ecg_data[-15000:, :]
    result = np.zeros((15000, nleads))  # 30 s, 500 Hz
    result[-nsteps:, :] = ecg_data
    return result.transpose()


def cal_scores(y_true, y_pred, y_score):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    return precision, recall, f1, auc, acc


def find_optimal_threshold(y_true, y_score):
    thresholds = np.linspace(0, 1, 100)
    f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
    return thresholds[np.argmax(f1s)]


def cal_f1(y_true, y_score, find_optimal):
    if find_optimal:
        thresholds = np.linspace(0, 1, 100)
    else:
        thresholds = [0.5]
    f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
    return np.max(f1s)


def cal_f1s(y_trues, y_scores, find_optimal=True):
    f1s = []
    for i in range(y_trues.shape[1]):
        f1 = cal_f1(y_trues[:, i], y_scores[:, i], find_optimal)
        f1s.append(f1)
    return np.array(f1s)


def cal_aucs(y_trues, y_scores):
    return roc_auc_score(y_trues, y_scores, average=None)


# 曲线图
def drawing():
    res = pandas.read_csv('../output/logs/2022-08-29-19-56-55.csv')
    train_loss = np.array(res['train_loss'])
    val_loss = np.array(res['val_val_loss'])

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
    plt.show()


if __name__ == '__main__':
    drawing()
