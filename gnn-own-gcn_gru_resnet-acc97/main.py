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
from utils.utils import split_data

parser = argparse.ArgumentParser()
parser.add_argument('--leads', type=str, default='all')
parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)
parser.add_argument('--resume', default=False, action='store_true', help='Resume')
parser.add_argument('--seq_len', type=int, default=1000)
parser.add_argument('--step_len', type=int, default=100)
parser.add_argument('--gru_num_layers', type=int, default=2)
parser.add_argument('--num-workers', type=int, default=1,
                    help='Num of workers to load data')  # 多线程加载数据
parser.add_argument('--log_path', type=str, default='output/logs')


def loadData(args, epoch):
    # base_dir = dirname(dirname(abspath(__file__)))
    org_data_dir = os.path.join(dataset_path)  # 源数据目录
    processed_data_dir = os.path.join(processed_path)  # 处理后的数据目录

    label_csv = os.path.join(processed_data_dir, 'labels.csv')

    train_folds, val_folds, test_folds = split_data(seed=epoch)
    train_dataset = ECGPtbXLDataset('train', org_data_dir, label_csv, train_folds, args.leads, seq_len=args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=False, drop_last=True)

    val_dataset = ECGPtbXLDataset('val', org_data_dir, label_csv, test_folds, args.leads, seq_len=args.seq_len)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False, drop_last=True)

    test_dataset = ECGPtbXLDataset('test', org_data_dir, label_csv, test_folds, args.leads, seq_len=args.seq_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=False, drop_last=True)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Training configs: {args}')

    # loss
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，常用于多分类任务

    model = EcgGCNGRUModel(seq_len=args.seq_len, step_len=args.step_len, num_classes=args.num_classes, leads=12,
                           batch_size=args.batch_size, gru_num_layers=args.gru_num_layers, device=args.device).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 16, gamma=0.1)
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
        model_path = os.path.join(result_train_file, '29_stemgnn.pt')
        model.load_state_dict(torch.load(model_path, map_location=args.device))
    train_time_total = 0
    if args.resume:
        start = 30
    else:
        start = 0
    lrs = []
    for epoch in range(start, args.epoch):
        train_loader, val_loader, test_loader = loadData(args, epoch)
        lrs.append(scheduler.get_last_lr()[0])
        epoch_start_time = time.time()
        train_loss = train(loader=train_loader, criterion=criterion, args=args, model=model, epoch=epoch,
                           scheduler=scheduler,
                           optimizer=optimizer)
        train_time = (time.time() - epoch_start_time)
        train_time_total += train_time
        # if epoch > 0 and epoch % 2 == 0:
        save_model(model=model, model_dir=result_train_file, epoch=epoch)

        # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
        # model.load_state_dict(torch.load(os.path.join(result_train_file, '0_stemgnn.pt'), map_location=args.device))
        train_avg_f1, train_val_loss = 0, 0
        # train_avg_f1, train_val_loss = validation(train_loader, model, criterion, args)
        val_avg_f1, val_val_loss = validation(test_loader, model, criterion, args)
        log_infos = [['epoch_' + str(epoch) + '_' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())),
                      '{:5.2f}s'.format(train_time),
                      '{:.8f}'.format(scheduler.get_last_lr()[0]),
                      '{:.4f}'.format(train_loss),
                      '{:.4f}'.format(train_val_loss),
                      '{:.4f}'.format(val_val_loss),
                      '{:.4f}'.format(train_avg_f1),
                      '{:.4f}'.format(val_avg_f1),
                      ]]
        print(log_infos[0])
        df = pandas.DataFrame(log_infos)
        if not os.path.exists(log_path):
            header = ['epoch', 'training_time', 'lr', 'train_loss', 'train_val_loss', 'val_val_loss', 'train_acc',
                      'val_acc']
        else:
            header = False
        df.to_csv(log_path, mode='a', header=header, index=False)
        time.sleep(0.5)
    print('Total time of training {:d} epochs: {:.2f}'.format(args.epoch, (train_time_total / 60)))
