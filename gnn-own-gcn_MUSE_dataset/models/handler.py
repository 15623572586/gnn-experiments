import os

import numpy as np
import torch
from tqdm import tqdm

from utils.evaluate_utils import cal_f1s, cal_aucs, get_thresholds, apply_thresholds


def train(loader, criterion, args, model, epoch, scheduler, optimizer):
    print("\nTraining epoch %d: " % epoch)
    loss_total = 0
    cnt = 0
    model.train()
    for idx, (inputs, target, features) in enumerate(tqdm(loader)):
        inputs, target, features = inputs.to(args.device), target.to(args.device), features.to(args.device)
        output = model(inputs, features)
        loss = criterion(output, target)
        # print(loss.item())
        cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += float(loss.item())
        if np.isnan(loss.item()):
            print(idx)
            print(loss.item())
            raise ValueError(
                "loss is nan"
            )
    if scheduler.get_last_lr()[0] >= 1e-6:
        scheduler.step()  # 动态更新学习率
    print('Training loss {:.4f}'.format(loss_total / cnt))
    return loss_total / cnt  # 所有batch的平均loss


# def validation(val_loader, test_loader, model, criterion, args):
#     thresholds = get_thresholds(test_loader, model, args.device)
#
#     print('Results on test data:')
#     test_loss = apply_thresholds(test_loader, model, criterion, args.device, thresholds)
#     return test_loss
#     # output_list, label_list, y_preds = [], [], []
#     # loss_total = 0
#     # cnt = 0
#     #
#     # for (data, label, features) in tqdm(test_loader):
#     #     data, labels, features = data.to(args.device), label.to(args.device), features.to(args.device)
#     #     output = model(data, features)
#     #     loss = criterion(output, labels)
#     #     loss_total += float(loss.item())
#     #     cnt += 1
#     #     output = torch.sigmoid(output)
#     #     output_list.append(output.data.cpu().numpy())
#     #     label_list.append(labels.data.cpu().numpy())
#     # y_trues = np.vstack(label_list)
#     # y_scores = np.vstack(output_list)
#     # y_preds = np.zeros_like(y_scores)
#     # y_preds[np.arange(len(y_scores)), y_scores.argmax(1)] = 1
#     #
#     # classes, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_measure_classes, f_beta_measure, g_beta_measure, challenge_metric \
#     #     = evaluate_12ECG_score(y_trues=y_trues, y_preds=y_preds)
#     # output_string = 'AUROC,AUPRC,Accuracy,F-measure,Fbeta-measure,Gbeta-measure,Challenge metric\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(
#     #     auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric)
#     # class_output_string = 'Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,{}'.format(
#     #     ','.join(classes),
#     #     ','.join('{:.3f}'.format(x) for x in auroc_classes),
#     #     ','.join('{:.3f}'.format(x) for x in auprc_classes),
#     #     ','.join('{:.3f}'.format(x) for x in f_measure_classes))
#     # print(output_string)
#     # print(class_output_string)
#     # return loss_total / cnt


# def validation_bbak(loader, model, criterion, args):
#     print('Validating...')
#     model.eval()
#     loss_total = 0
#     cnt = 0
#     output_list, labels_list = [], []
#     for data, labels, features in tqdm(loader):  # Iterate in batches over the training/test dataset.
#         data, labels, features = data.to(args.device), labels.to(args.device), features.to(args.device)
#         output = model(data, features)
#         loss = criterion(output, labels)
#         loss_total += float(loss.item())
#         cnt += 1
#         output = torch.sigmoid(output)
#         output_list.append(output.data.cpu().numpy())
#         labels_list.append(labels.data.cpu().numpy())
#     y_trues = np.vstack(labels_list)
#     y_scores = np.vstack(output_list)
#     f1s = cal_f1s(y_trues, y_scores)
#     avg_f1 = np.mean(f1s)
#     print('F1s:', f1s)
#     print('Avg F1: %.4f' % avg_f1)
#     # if avg_f1 > args.best_metric:
#     #     args.best_metric = avg_f1
#     aucs = cal_aucs(y_trues, y_scores)
#     avg_auc = np.mean(aucs)
#     print('AUCs:', aucs)
#     print('Avg AUC: %.4f' % avg_auc)
#     return output_list, y_trues, y_scores, (loss_total / cnt)


def validation_bak(loader, model, criterion, args):
    print('Validating...')
    model.eval()
    loss_total = 0
    cnt = 0
    correct = 0
    pred_list, target_list, pred_scores = [], [], []
    for data, label, features in tqdm(loader):  # Iterate in batches over the training/test dataset.
        data = data.to(args.device)
        label = label.to(args.device)
        features = features.to(args.device)
        out = model(data, features)
        loss = criterion(out, label)
        loss_total += float(loss.item())
        cnt += 1
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        labels = label
        # labels = label.argmax(dim=1)
        pred_list += pred.cpu().tolist()
        target_list += labels.cpu().tolist()
        pred_scores += out.cpu().tolist()
        correct += int((pred == label).sum())  # Check against ground-truth labels.
    print(correct / len(loader.dataset))
    return pred_list, target_list, pred_scores, (loss_total / cnt)


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
