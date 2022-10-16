import os.path

import numpy as np
import pandas as pd
from biosppy import storage
from biosppy.signals import ecg
import torch


from process.variables import processed_data

df = pd.read_csv(os.path.join(processed_data, 'S0549.csv'), header=None)
ecg_data = np.array(df.values).T
out = ecg.ecg(signal=ecg_data[7], sampling_rate=100, show=False)
print(out)