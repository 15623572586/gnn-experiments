import os

from matplotlib import pyplot as plt
import numpy as np
import pandas
import wfdb.processing
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from process.variables import dataset_path

def test_resample(log_file):
    ecg_data, meta_data = wfdb.rdsamp(os.path.join(dataset_path, 'HR00001'))
    all_sig = ecg_data.T
    all_sig_lr = []
    for sig in all_sig:
        data = wfdb.processing.resample_sig(x=sig, fs=500, fs_target=100)
        all_sig_lr.append(data[0])

    res = pandas.read_csv(os.path.join('../output/logs', log_file + '.csv'))
    train_loss = np.array(res['train_loss'])
    epoch = [i for i in range(0, train_loss.size)]
    plt.plot(epoch, train_loss, label='train_loss')
    plt.legend()
    plt.show()

    # 曲线图
def drawing_logs(log_file):
    res = pandas.read_csv(os.path.join('../output/logs', log_file + '.csv'))
    train_loss = np.array(res['train_loss'])
    # val_loss = np.array(res['test_loss'])
    #
    # train_f1 = np.array(res['train_acc'])
    # val_f1 = np.array(res['val_acc'])
    # lr = np.array(res['lr'])

    epoch = [i for i in range(0, train_loss.size)]

    # fig, ax = plt.subplots()  # 创建图实例
    plt.plot(epoch, train_loss, label='train_loss')
    # plt.plot(epoch, val_loss, label='val_loss')
    # plt.plot(epoch, train_f1, label='train_acc')
    # plt.plot(epoch, val_f1, label='val_acc')
    # plt.plot(epoch, lr, label='lr')
    plt.xlabel('epoch')  # 设置x轴名称 x label
    plt.ylabel('loss and f1')  # 设置y轴名称 y label
    plt.title('Simple Plot')  # 设置图名为Simple Plot
    plt.legend()
    plt.show()

if __name__ == '__main__':
    drawing_logs('2022-10-03-19-21-12')