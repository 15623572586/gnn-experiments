import os.path

import pandas as pd
import numpy as np
import wfdb
import ast

from tqdm import tqdm


def load_raw_data(df, sampling_rate, path):
    ecg_data = []
    if sampling_rate == 100:
        for f in tqdm(df.filename_lr):
            data, meta = wfdb.rdsamp(os.path.join(path, f))
            ecg_data.append(data)
        # data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    res = np.array([signal for signal, meta in ecg_data])
    return res


path = r'D:\learning\科研\数据\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1'
sampling_rate = 100

# load and convert annotation data
Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
# X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]
print(agg_df.index)


def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


# Apply diagnostic superclass
Y['diagnostic_superclass'] = scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
# Train
# X_train = X[np.where(Y.strat_fold != test_fold)]
# y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# # Test
# X_test = X[np.where(Y.strat_fold == test_fold)]
# y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
