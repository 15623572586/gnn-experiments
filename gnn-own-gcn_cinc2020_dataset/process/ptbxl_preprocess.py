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




if __name__ == '__main__':
    label_csv = os.path.join(processed_path, 'labels.csv')
    gen_label_csv(label_csv)
