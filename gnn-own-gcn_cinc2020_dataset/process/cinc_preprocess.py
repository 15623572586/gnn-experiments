import ast
import math
import os.path
from collections import Counter
from glob import glob

import torch
import numpy as np
import pandas as pd
import pywt
import wfdb
import wfdb.processing
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.signal import medfilt
from tqdm import tqdm

from process.variables import processed_path, processed_data

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# classes = ['IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR', 'PAC', 'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'NSR', 'STach', 'SVPB', 'TAb', 'TInv', 'VPB']
# classes = ['IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR', 'PAC',
#            'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'SA', 'SB', 'NSR', 'STach', 'TAb', 'TInv']
classes = ['IAVB', 'AF', 'CRBBB', 'PVC', 'SB', 'NSR', 'STach', 'TInv']
# classes = ['IAVB', 'AF', 'CRBBB', 'PAC', 'PVC', 'SB', 'NSR', 'STach', 'TInv']
class_wight = [1, 1, 1, 1, 1, 1, 1, 1, 1]
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


# 获取标签，生成lable索引文件
# 来自CinC2020的数据
def gen_label_cinc_csv(label_csv):
    df = pd.read_csv(os.path.join(
        r'D:\projects\python-projects\experiments\own-model\gnn-own-gcn_cinc2020_dataset\evaluation_2020\dx_mapping_scored_class9.csv'))
    gen_df = pd.read_csv(
        os.path.join(r'D:\projects\python-projects\experiments\dataset\preprocessed\cinc2020\label_gen.csv'))
    code_map = {}
    for i, row in df.iterrows():
        code_map[str(row['SNOMED CT Code'])] = row['Abbreviation']
    print(code_map)
    recordpaths = []
    for path in paths:
        recordpaths += glob(os.path.join(path, '*.hea'))
    results = []
    nsr_count = 0
    for recordpath in tqdm(recordpaths):
        ecg_id = recordpath.split("\\")[-1][:-4]
        _, meta_data = wfdb.rdsamp(recordpath[:-4])
        dx = meta_data['comments'][2]
        dx = dx[4:] if dx.startswith('Dx: ') else ''
        dx = dx.split(',')
        for code in dx:
            labels = [0] * len(classes)
            if code not in code_map:
                continue
            abbreviation = code_map[code]  # 类型缩写
            if abbreviation in classes:
                labels[classes.index(abbreviation)] = 1
                # break
            if 1 in labels:
                if nsr_count > 10000 and classes[np.argmax(labels)] == 'NSR':
                    continue
                if classes[np.argmax(labels)] == 'NSR':
                    nsr_count += 1
                weights = class_wight[np.argmax(labels)]
                for i in range(0, weights):
                    results.append([ecg_id] + labels)
    df = pd.DataFrame(data=results, columns=['ecg_id'] + classes)
    df = pd.concat([df, gen_df])
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


# 过采样：解决数据不平衡问题
def over_sample():
    # tar_class = ['CRBBB', 'PVC', 'STach', 'TInv']
    ecg_lables = pd.read_csv(os.path.join(processed_path, 'labels_cinc.csv'))

    gen_ecg_id = 1
    gen_labels = []
    gen_dic = {
        'IAVB': 2000,
        'CRBBB': 2500,
        'PVC': 2500,
        'SB': 2500,
        'STach': 2500,
        'TInv': 3500,
    }
    for key in gen_dic.keys():
        gen_cout = 0
        nsr_count = 0
        X_list = []
        Y_list = []
        for i, row in tqdm(ecg_lables.iterrows()):
            if nsr_count >= (gen_dic[key] + 10) and gen_cout >= 10:
                break
            if nsr_count >= (gen_dic[key] + 10) and (row['NSR'] == 1 or row['NSR'] == '1'):
                continue
            y = classes.index(key)
            if row[key] == 1 or row[key] == '1':
                gen_cout += 1
            elif row['NSR'] == 1 or row['NSR'] == '1':
                nsr_count += 1
                y = classes.index('NSR')
            else:
                continue
            ecg_id = row['ecg_id']
            ecg_data = np.array(pd.read_csv(os.path.join(processed_data, str(ecg_id) + '.csv'), header=None)).T

            ecg_data = ecg_data[-12:, -1000:]
            result = np.zeros((12, 1000))
            result[-1000:, :] = ecg_data
            x_list = []
            flag = False
            for idx, lead_data in enumerate(ecg_data):
                if not np.any(ecg_data[idx]):  # 判断是否全0，如果全0，跳过
                    flag = True
                    break
                x_list.extend(ecg_data[idx])
            if not flag:
                X_list.append(x_list)
                Y_list.append(y)
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_list, Y_list)
        print('Resampled dataset shape %s' % Counter(y_res))
        for idx in tqdm(range((gen_dic[key] + 20), len(X_res))):
            if y_res[idx] != classes.index(key):
                continue
            ecg_data_1d = X_res[idx]
            new_ecg_data = []
            for i in range(0, len(ecg_data_1d), 1000):
                new_ecg_data.append(ecg_data_1d[i:i + 1000])
            new_ecg_data = np.array(new_ecg_data).T
            df = DataFrame(new_ecg_data)
            ecg_id = 'G' + key + format(gen_ecg_id, '05d')
            file_name = os.path.join(r'D:\projects\python-projects\experiments\dataset\preprocessed\cinc2020\data',
                                     ecg_id + '.csv')
            df.to_csv(file_name, header=False, index=False)
            gen_ecg_id += 1
            label = [0] * len(classes)
            label[y_res[idx]] = 1
            gen_labels.append([ecg_id] + label)

    label_df = DataFrame(data=gen_labels, columns=['ecg_id'] + classes)
    label_df.to_csv(
        os.path.join(r'D:\projects\python-projects\experiments\dataset\preprocessed\cinc2020', 'label_gen.csv'),
        index=False)
    # plt.figure(figsize=(60, 5))
    # # plt.plot(X_res[2])
    # plt.plot(new_ecg_data[11])
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
    over_sample()
    label_csv = os.path.join(processed_path, 'labels_cinc_new.csv')
    gen_label_cinc_csv(label_csv)
    # statistic_sub_class()
    # process_mat_E()
    # statistic_subclass_byhea()
    # denoise()
