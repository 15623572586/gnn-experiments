import ast
import math
import os.path
from glob import glob

import torch
import numpy as np
import pandas as pd
import pywt
import wfdb
import wfdb.processing
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.signal import medfilt
from tqdm import tqdm

from process.variables import processed_path, processed_data

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# classes = ['IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR', 'PAC', 'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'NSR', 'STach', 'SVPB', 'TAb', 'TInv', 'VPB']
# classes = ['IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR', 'PAC',
#            'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'SA', 'SB', 'NSR', 'STach', 'TAb', 'TInv']
classes = ['IAVB', 'AF', 'CRBBB', 'PAC', 'PVC', 'SB', 'NSR', 'STach', 'TAb']
normal_class = '426783006'
equivalent_classes = {
    '59118001': '713427006',
    '63593006': '284470004',
    '17338001': '427172004'
}

paths = [
    r'D:\learning\科研\数据\PhysioNetChallenge2020\5_PhysioNetChallenge2020_Training_PTB-XL',
    r'D:\learning\科研\数据\PhysioNetChallenge2020\4_PhysioNetChallenge2020_Training_PTB',
    r'D:\learning\科研\数据\PhysioNetChallenge2020\1_PhysioNetChallenge2020_Training_CPSC',
    r'D:\learning\科研\数据\PhysioNetChallenge2020\2_PhysioNetChallenge2020_Training_2_China 12-Lead ECG Challenge Database',
    r'D:\learning\科研\数据\PhysioNetChallenge2020\6_PhysioNetChallenge2020_Training_E'
]


# def get_superclass(scp_codes):
#     super_class = []
#     for scp_code in scp_codes:  # {'NORM': 100.0, ...}
#         scp_code = dict(sorted(scp_code.items(), key=lambda x: x[1], reverse=True))  # 按值排序
#         scps = []
#         max_value = max(scp_code.values())
#         for scp in scp_code:
#             if max_value < 50:
#                 if scp_code[scp] > 0:
#                     scps.append(scp)
#             else:
#                 if scp_code[scp] > 50:
#                     scps.append(scp)
#         super_class

# 获取标签，生成lable索引文件
# 来自CinC2020的数据
def gen_label_cinc_csv(label_csv):
    df = pd.read_csv(os.path.join(
        r'D:\projects\python-projects\experiments\own-model\gnn-own-gcn_cinc2020_dataset\evaluation_2020\dx_mapping_scored_class8.csv'))
    code_map = {}
    for i, row in df.iterrows():
        code_map[str(row['SNOMED CT Code'])] = row['Abbreviation']
    print(code_map)
    recordpaths = []
    for path in paths:
        recordpaths += glob(os.path.join(path, '*.hea'))
    results = []
    for recordpath in tqdm(recordpaths):
        ecg_id = recordpath.split("\\")[-1][:-4]
        _, meta_data = wfdb.rdsamp(recordpath[:-4])
        dx = meta_data['comments'][2]
        dx = dx[4:] if dx.startswith('Dx: ') else ''
        dx = dx.split(',')
        labels = [0] * len(classes)
        for code in dx:
            if code not in code_map:
                continue
            abbreviation = code_map[code]  # 类型缩写
            if abbreviation in classes:
                labels[classes.index(abbreviation)] = 1
                break
        if 1 in labels:
            results.append([ecg_id] + labels)
    df = pd.DataFrame(data=results, columns=['ecg_id'] + classes)
    n = len(df)
    folds = np.zeros(n, dtype=np.int8)
    for i in range(10):
        start = int(n * i / 10)
        end = int(n * (i + 1) / 10)
        folds[start:end] = i + 1
    df['fold'] = np.random.permutation(folds)
    df.to_csv(label_csv, index=None)


# 数据噪声很大：基线漂移、工频干扰、肌电干扰
# 先用中值滤波去基线漂移干扰；再用小波变换去工频干扰和肌电干扰
def denoise():
    filter = int(0.8 * 100)
    recordpaths = []
    for path in paths:
        recordpaths += glob(os.path.join(path, '*.hea'))
    for recordpath in tqdm(recordpaths):
        ecg_id = recordpath.split("\\")[-1][:-4]
        ecg_data, meta_data = wfdb.rdsamp(recordpath[:-4])
        ecg_data = np.nan_to_num(ecg_data)  # 处理非数字的脏数据
        # 0、=====降采样到100Hz=====
        fs = meta_data['fs']
        ecg_data = ecg_data.T
        all_sig_lr = []
        for sig in ecg_data:
            data = wfdb.processing.resample_sig(x=sig, fs=fs, fs_target=100)
            all_sig_lr.append(data[0])
        # 1、======中值滤波======
        baseline = []
        for idx, lead in enumerate(leads):
            baseline.append(medfilt(all_sig_lr[idx], filter + 1))
        filtered_data_1 = np.array(all_sig_lr) - np.array(baseline)

        # plt.figure(figsize=(60, 5))
        # plt.plot(all_sig_lr[2])
        # plt.plot(filtered_data_1[2])
        # plt.show()
        # 2、======小波滤波=======
        w = pywt.Wavelet('db8')
        filtered_data_2 = []
        # 选用Daubechies8小波
        for data in filtered_data_1:
            maxlev = pywt.dwt_max_level(len(data), w.dec_len)
            threshold = 0.1  # Threshold for filtering
            coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解
            # 这里就是对每一层的coffe进行更改
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
            filtered_data_2.append(pywt.waverec(coeffs, 'db8'))  # 将信号进行小波重构
        filtered_data_2 = np.array(filtered_data_2).T
        df = DataFrame(filtered_data_2)
        df.to_csv(os.path.join(processed_data, str(ecg_id) + '.csv'), header=False, index=False)
        # plt.figure(figsize=(60, 5))
        # plt.plot(filtered_data_1[2])
        # plt.plot(filtered_data_2[2])
        # plt.legend(['Before', 'After'])
        # plt.show()


# 统计小类别
def statistic_subclass_byhea():
    df = pd.read_csv(os.path.join(
        r'D:\projects\python-projects\experiments\own-model\gnn-own-gcn_cinc2020_dataset\evaluation_2020\dx_mapping_scored_1.csv'))
    code_map = {}
    for i, row in df.iterrows():
        code_map[str(row['SNOMED CT Code'])] = row['Abbreviation']
    print(code_map)
    paths = [
        r'D:\learning\科研\数据\PhysioNetChallenge2020\5_PhysioNetChallenge2020_Training_PTB-XL',
        r'D:\learning\科研\数据\PhysioNetChallenge2020\4_PhysioNetChallenge2020_Training_PTB',
        r'D:\learning\科研\数据\PhysioNetChallenge2020\1_PhysioNetChallenge2020_Training_CPSC',
        r'D:\learning\科研\数据\PhysioNetChallenge2020\2_PhysioNetChallenge2020_Training_2_China 12-Lead ECG Challenge Database',
        r'D:\learning\科研\数据\PhysioNetChallenge2020\6_PhysioNetChallenge2020_Training_E'
    ]
    recordpaths = []
    for path in paths:
        recordpaths += glob(os.path.join(path, '*.hea'))
    results = {}
    for type in code_map.values():
        results[type] = 0
    for recordpath in tqdm(recordpaths):
        patient_id = recordpath.split("\\")[-1][:-4]
        # print(patient_id)
        _, meta_data = wfdb.rdsamp(recordpath[:-4])
        dx = meta_data['comments'][2]
        dx = dx[4:] if dx.startswith('Dx: ') else ''
        dx = dx.split(',')
        for code in dx:
            if code not in code_map:
                continue
            if normal_class == code:
                if 'NORM' in code_map:
                    code_map['NORM'] += 1
                else:
                    code_map['NORM'] = 1
            if code in equivalent_classes:
                code = equivalent_classes[code]
            abbreviation = code_map[code]
            if abbreviation in results:
                results[abbreviation] += 1
                break
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    results_df = DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(processed_path, 'codes.csv'), header=False)
    print(results)


# header中第一行出现了E00001.mat 要变成 E00001,才能用wfdb读
def process_mat_E():
    path = r'D:\learning\科研\数据\PhysioNetChallenge2020\6_PhysioNetChallenge2020_Training_E'
    files = glob(os.path.join(path, '*.hea'))
    for file_name in files:
        new_content = ''
        with open(file_name, 'r') as f:
            first_row = True
            for line in f:
                if first_row:
                    line = line.replace(".mat", "")
                    first_row = False
                new_content += line
        with open(file_name, 'w', encoding="utf-8") as f:
            f.write(new_content)


if __name__ == '__main__':
    # label_csv = os.path.join(processed_path, 'labels.csv')
    # gen_label_csv(label_csv)
    label_csv = os.path.join(processed_path, 'labels_cinc.csv')
    gen_label_cinc_csv(label_csv)
    # statistic_sub_class()
    # process_mat_E()
    # statistic_subclass_byhea()
    # denoise()
