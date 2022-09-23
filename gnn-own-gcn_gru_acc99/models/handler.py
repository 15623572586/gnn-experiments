import os

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from tqdm import tqdm

from utils.utils import cal_scores


def train(loader, criterion, args, model, epoch, scheduler, optimizer):
    print("\nTraining epoch %d: " % epoch)
    loss_total = 0
    cnt = 0
    model.train()
    for idx, (inputs, target) in enumerate(tqdm(loader)):
        inputs, target = inputs.to(args.device), target.to(args.device)
        output = model(inputs)
        loss = criterion(output, target)
        # print(loss.item())
        cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += float(loss.item())
    scheduler.step()  # 动态更新学习率
    print('Training loss {:.4f}'.format(loss_total / cnt))
    return loss_total / cnt  # 所有batch的平均loss


def validation(loader, model, criterion, args):
    print('Validating...')
    model.eval()
    correct = 0
    loss_total = 0
    cnt = 0
    output_list, labels_list = [], []
    for data, label in tqdm(loader):  # Iterate in batches over the training/test dataset.
        data = data.to(args.device)
        label = label.to(args.device)
        out = model(data)
        # out, _ = model(data)
        loss = criterion(out, label)
        # print(loss.item())
        loss_total += float(loss.item())
        cnt += 1
        # output = torch.sigmoid(out)
        # output_list.append(output.data.cpu().numpy())
        # labels_list.append(label.data.cpu().numpy())
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        correct += int((pred == label).sum())  # Check against ground-truth labels.
    print('Loss: %.4f' % (loss_total / cnt))
    # y_trues = np.vstack(labels_list)
    # y_scores = np.vstack(output_list)
    # f1s = cal_f1s(y_trues, y_scores)
    # avg_f1 = float(np.mean(f1s))
    # print('F1s:', f1s)
    # print('Avg F1: %.4f' % avg_f1)
    # return avg_f1, (loss_total / cnt)
    return correct / len(loader.dataset), (loss_total / cnt)  # Derive ratio of correct predictions.


def cal_f1_my(y_trues, y_scores):
    y_pred = []
    y_true = []
    for scores in y_scores:
        pred = scores.argmax()
        y_pred.append(pred)
    for trues in y_trues:
        label = trues.argmax()
        y_true.append(label)
    acc = accuracy_score(y_true, y_pred)
    print('acc: %.4f' % acc)
    # f1 = f1_score(y_true, y_pred, average='macro')
    return acc


def cal_f1s(y_trues, y_scores, find_optimal=True):
    f1s = []
    for i in range(y_trues.shape[1]):
        f1 = cal_f1(y_trues[:, i], y_scores[:, i], find_optimal)
        f1s.append(f1)
    return np.array(f1s)




def cal_f1(y_true, y_score, find_optimal):
    if find_optimal:
        thresholds = np.linspace(0, 1, 100)
    else:
        thresholds = [0.5]
    f1s = []
    for threshold in thresholds:
        b = y_score > threshold
        f1 = f1_score(y_true, b)
        f1s.append(f1)
    # f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
    return np.max(f1s)


def test(loader, model, args):
    # model = load_model(model_file)
    correct = 0
    for data, label in tqdm(loader):  # Iterate in batches over the training/test dataset.
        data = data.to(args.device)
        label = label.to(args.device).long()
        out, _ = model(data)
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        correct += int((pred == label).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch)
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    # epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model
