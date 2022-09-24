import argparse
import os
import time
from datetime import date

import pandas
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.gcn_gru import EcgGCNGRUModel
from models.handler import train, save_model, validation
from data_loader.ptbxl_dataset import ECGPtbXLDataset
from process.variables import dataset_path, processed_path
from utils.utils import split_data, performance, drawing_confusion_matric, drawing_roc_auc

parser = argparse.ArgumentParser()
parser.add_argument('--leads', type=int, default=12)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--decay_rate', type=float, default=1e-5)
parser.add_argument('--seq_len', type=int, default=1000)
parser.add_argument('--step_len', type=int, default=20)
parser.add_argument('--gru_num_layers', type=int, default=2)
parser.add_argument('--num-workers', type=int, default=1,
                    help='Num of workers to load data')  # 多线程加载数据
parser.add_argument('--log_path', type=str, default='output/logs')
parser.add_argument('--loss_path', type=str, default='output/loss')
parser.add_argument('--roc_path', type=str, default='output/roc_auc')
parser.add_argument('--mat_path', type=str, default='output/conf_matric')
parser.add_argument('--resume', default=False, action='store_true', help='Resume')
parser.add_argument('--model_name', default='52_stemgnn.pt', action='store_true', help='Resume')
parser.add_argument('--start', default=55, action='store_true', help='Resume')

def loadData(args, epoch):
    org_data_dir = os.path.join(dataset_path)  # 源数据目录
    processed_data_dir = os.path.join(processed_path)  # 处理后的数据目录

    label_csv = os.path.join(processed_data_dir, 'labels.csv')
    train_folds, test_folds = split_data(seed=epoch)

    train_dataset = ECGPtbXLDataset('train', org_data_dir, label_csv, train_folds, seq_len=args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=False, drop_last=True)

    test_dataset = ECGPtbXLDataset('test', org_data_dir, label_csv, test_folds, seq_len=args.seq_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=False, drop_last=True)
    return train_loader, test_loader


if __name__ == '__main__':
    args = parser.parse_args()

    # loss
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，常用于多分类任务
    model = EcgGCNGRUModel(seq_len=args.seq_len, step_len=args.step_len, num_classes=args.num_classes, leads=args.leads,
                           batch_size=args.batch_size, gru_num_layers=args.gru_num_layers, device=args.device).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    result_train_file = os.path.join('output', 'train')
    result_test_file = os.path.join('output', 'test')
    log_path = os.path.join(args.log_path, str(date.today()) + '.csv')
    if os.path.exists(log_path):
        log_path = os.path.join(args.log_path,
                                str(date.today()) + '-' + time.strftime("%H-%M-%S", time.localtime()) + '.csv')
    if args.resume:
        model_path = os.path.join(result_train_file, args.model_name)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
    train_time_total = 0
    if args.resume:
        start = args.start
    else:
        start = 0
    lrs = []
    mat_path = os.path.join(args.mat_path, str(date.today()) + '-' + time.strftime("%H-%M-%S", time.localtime()))
    if not os.path.exists(mat_path):
        os.mkdir(mat_path)
    roc_path = os.path.join(args.roc_path, str(date.today()) + '-' + time.strftime("%H-%M-%S", time.localtime()))
    if not os.path.exists(roc_path):
        os.mkdir(roc_path)
    for epoch in range(start, args.epoch):
        train_loader, test_loader = loadData(args, epoch)
        lrs.append(scheduler.get_last_lr()[0])
        epoch_start_time = time.time()
        train_loss = train(loader=train_loader, criterion=criterion, args=args, model=model, epoch=epoch,
                           scheduler=scheduler,
                           optimizer=optimizer)
        train_time = (time.time() - epoch_start_time)
        train_time_total += train_time
        save_model(model=model, model_dir=result_train_file, epoch=epoch)
        pred_list, target_list, pred_scores, test_loss = validation(test_loader, model, criterion, args)
        confusion_matrix, evaluate_res = performance(pred_list, target_list, pred_scores, args)  # 模型评估
        drawing_confusion_matric(confusion_matrix, os.path.join(mat_path, str(epoch)+'.png'))  # 绘制混淆矩阵
        drawing_roc_auc(evaluate_res, os.path.join(roc_path, str(epoch)+'.png'))  # 绘制roc_auc曲线
        log_infos = [['epoch_' + str(epoch) + '_' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),
                      '{:5.2f}s'.format(train_time),
                      '{:.8f}'.format(scheduler.get_last_lr()[0]),
                      '{:.4f}'.format(train_loss),
                      '{:.4f}'.format(test_loss),
                      '{:.4f}'.format(evaluate_res['f1_value']),
                      '{:.4f}'.format(evaluate_res['acc_value']),
                      ]]
        print(log_infos[0])
        df = pandas.DataFrame(log_infos)
        if not os.path.exists(log_path):
            header = ['epoch', 'training_time', 'lr', 'train_loss', 'test_loss', 'f1_value', 'val_acc']
        else:
            header = False
        df.to_csv(log_path, mode='a', header=header, index=False)
        time.sleep(0.5)
    print('Total time of training {:d} epochs: {:.2f}'.format(args.epoch, (train_time_total / 60)))
