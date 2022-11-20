import os.path

import numpy as np
import pandas as pd

from process.variables import dataset_path

traindata = pd.read_csv(os.path.join(dataset_path, 'Diagnostics.csv'))
target = 'pose'  # pose的值就是分类
feature_columns = ['PatientAge', 'Gender', 'VentricularRate', 'AtrialRate', 'QRSDuration', 'QTInterval', 'QTCorrected',
                   'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'TOffset']
traindata = traindata[feature_columns]


# 数据清洗
def harmonize_data(features):
    # 对数据进行归一化
    # 首先是归一化函数
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    # 我的数据集有38列，前36列为数值，第37列为时间，第38列为字符串类型，因此只对前36列做数值归一
    for title in feature_columns:
        if title == 'Gender':
            continue
        features[title] = features[[title]].apply(max_min_scaler)
    # 把FEMALE定义为0
    features.loc[features['Gender'] == 'FEMALE', 'Gender'] = 0
    features.loc[features['Gender'] == 'MALE', 'Gender'] = 1
    return features


precessed_train_data = harmonize_data(traindata)
print(precessed_train_data)
