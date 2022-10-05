import ast
import math
import os.path
from glob import glob

import numpy as np
import pandas as pd
import wfdb
from pandas import DataFrame
from tqdm import tqdm

from process.variables import processed_path

class_dict = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
data_dir = r'D:\projects\python-projects\experiments\dataset\ptb-xl-1.0.2'
scp_statements_csv = os.path.join(data_dir, 'scp_statements.csv')
# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(scp_statements_csv, index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


classes = ['IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR', 'PAC', 'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'NSR', 'STach', 'SVPB', 'TAb', 'TInv', 'VPB']


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

def aggregate_diagnostic(y_dic):
    tmp = []
    # max_value = max(y_dic.values())
    for key in y_dic.keys():
        # if max_value > 50:
        #     if y_dic[key] > 50 and key in agg_df.index:
        #         tmp.append(agg_df.loc[key].diagnostic_class)
        # else:
        if y_dic[key] >= 100 and key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    if len(tmp) > 1 and 'NORM' in tmp:
        tmp.remove('NORM')

    return list(set(tmp))


def gen_label_csv(label_csv):
    if not os.path.exists(label_csv):
        results = []
        Y = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'))
        scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        norm_count = 0
        # get_superclass(scp_codes)
        Y['diagnostic_superclass'] = scp_codes.apply(aggregate_diagnostic, norm_count)
        for _, row in tqdm(Y.iterrows()):
            labels = [0] * 5
            if math.isnan(row.age):
                row.age = 62
            if math.isnan(row.weight):
                row.weight = 60
            for superclass in row.diagnostic_superclass:
                if superclass in class_dict:
                    labels[class_dict.index(superclass)] = 1
            if 1 in labels:
                results.append([row.ecg_id] + [row.age] + [row.sex] + [row.weight] + labels + [row.strat_fold] + [
                    row.filename_lr] + [row.filename_hr])
            else:
                print(row.ecg_id)
        df = pd.DataFrame(data=results, columns=['ecg_id'] + ['age'] + ['sex'] + ['weight'] + class_dict + ['fold'] + [
            'filename_lr'] + ['filename_hr'])
        df.to_csv(label_csv, index=None)


# 来自CinC2020的数据
def gen_label_csv_1(label_csv):
    df = pd.read_csv(os.path.join(
        r'D:\projects\python-projects\experiments\own-model\gnn-own-gcn_expert_features\evaluation_2020\dx_mapping_scored.csv'))
    code_map = {}
    # normal_class = '426783006'
    equivalent_classes = {
        '59118001': '713427006',
        '63593006': '284470004',
        '17338001': '427172004'}
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
            # if code in equivalent_classes:
            #     code = equivalent_classes[code]
            abbreviation = code_map[code]  # 类型缩写
            if abbreviation in classes:
                labels[classes.index(abbreviation)] = 1
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


def statistic_sub_class():
    data = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'))
    data_it = data.iterrows()
    scp_codes = data.scp_codes.apply(lambda x: ast.literal_eval(x))
    class_dict = {}
    for scp_code in tqdm(scp_codes):
        for key in scp_code.keys():
            if scp_code[key] >= 100 and key in agg_df.index:
                if key not in class_dict:
                    class_dict[key] = 1
                else:
                    class_dict[key] += 1
    class_dict = dict(sorted(class_dict.items(), key=lambda x: x[1], reverse=True))
    print(class_dict)


def statistic_subclass_byhea():
    df = pd.read_csv(os.path.join(
        r'D:\projects\python-projects\experiments\own-model\gnn-own-gcn_expert_features\evaluation_2020\dx_mapping_scored.csv'))
    code_map = {}
    normal_class = '426783006'
    equivalent_classes = {
        '59118001': '713427006',
        '63593006': '284470004',
        '17338001': '427172004'}
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
            abbreviation = code_map[code]
            if abbreviation in results:
                results[abbreviation] += 1
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
    gen_label_csv_1(label_csv)
    # statistic_sub_class()
    # process_mat_E()
    # statistic_subclass_byhea()
