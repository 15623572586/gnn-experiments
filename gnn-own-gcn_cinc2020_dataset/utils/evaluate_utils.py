import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
from evaluation_2020 import evaluate_12ECG_score
from evaluation_2020.evaluate_12ECG_score import load_weights, compute_accuracy, compute_f_measure, \
    compute_beta_measures, compute_challenge_metric

from process.ptbxl_preprocess import classes


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


def get_thresholds(val_loader, net, device):
    print('Finding optimal thresholds...')
    # if os.path.exists(threshold_path):
    #     return pickle.load(open(threshold_path, 'rb'))
    output_list, label_list = [], []
    for (data, label, features) in tqdm(val_loader):
        data, labels, features = data.to(device), label.to(device), features.to(device)
        output = net(data, features)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    thresholds = []
    for i in range(y_trues.shape[1]):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        threshold = find_optimal_threshold(y_true, y_score)
        thresholds.append(threshold)
    # pickle.dump(thresholds, open(threshold_path, 'wb'))
    return thresholds


def apply_thresholds(test_loader, net, criterion, device, thresholds):
    output_list, label_list = [], []
    loss_total = 0
    cnt = 0
    for (data, label, features) in tqdm(test_loader):
        data, labels, features = data.to(device), label.to(device), features.to(device)
        output = net(data, features)
        loss = criterion(output, labels)
        loss_total += float(loss.item())
        cnt += 1
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    y_preds = []
    scores = []
    for i in range(len(thresholds)):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        y_pred = (y_score >= thresholds[i]).astype(int)
        scores.append(cal_scores(y_true, y_pred, y_score))
        y_preds.append(y_pred)
    y_preds = np.array(y_preds).transpose()
    scores = np.array(scores)
    print('Precisions:', scores[:, 0])
    print('Recalls:', scores[:, 1])
    print('F1s:', scores[:, 2])
    print('F1_avg:', np.mean(scores[:, 2]))
    print('AUCs:', scores[:, 3])
    print('Accs:', scores[:, 4])
    print(np.mean(scores, axis=0))
    # plot_cm(y_trues, y_preds)
    return loss_total / cnt


def plot_cm(y_trues, y_preds, normalize=True, cmap=plt.cm.Blues):
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    for i, label in enumerate(classes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=[0, 1], yticklabels=[0, 1],
               title=label,
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), ha="center")

        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        np.set_printoptions(precision=3)
        fig.tight_layout()
        plt.savefig(
            r'D:\projects\python-projects\experiments\own-model\gnn-own-gcn_expert_features\\utils\\results/' + label + '.png')
        plt.close(fig)


def test_evaluate(y_trues, y_preds, thresholds):
    y_preds[y_preds >= thresholds] = 1
    y_preds[y_preds < thresholds] = 0

    weights_file = r'D:\projects\python-projects\experiments\own-model\gnn-own-gcn_expert_features\evaluation_2020\weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    # Load the scored classes and the weights for the Challenge metric.
    print('Loading weights...')
    classes, weights = load_weights(weights_file, equivalent_classes)

    print('- Accuracy...')
    accuracy = compute_accuracy(y_trues, y_preds)
    print("acc:" + str(accuracy))

    print('- F-measure...')
    f_measure, f_measure_classes = compute_f_measure(y_trues, y_preds)
    print(f_measure)
    print(f_measure_classes)

    print('- F-beta and G-beta measures...')
    f_beta_measure, g_beta_measure = compute_beta_measures(y_trues, y_preds, beta=2)
    print(f_beta_measure)
    print(g_beta_measure)

    print('- Challenge metric...')
    challenge_metric = compute_challenge_metric(weights, y_trues, y_preds, classes, normal_class)
    print(challenge_metric)

    print('Done.')
