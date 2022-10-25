import pandas
import wfdb
import numpy as np
import pandas as pd

from glob import glob
import os

from tqdm import tqdm

from data_loader.cpsc_dataset import transform
from process.variables import dataset_path, processed_path

org_data_dir = dataset_path
processed_data_dir = processed_path

def gen_reference_csv(data_dir, reference_csv):
    if not os.path.exists(reference_csv):
        recordpaths = glob(os.path.join(data_dir, '*.hea'))
        results = []
        for recordpath in tqdm(recordpaths):
            patient_id = recordpath.split('\\')[-1][:-4]
            _, meta_data = wfdb.rdsamp(recordpath[:-4])
            sample_rate = meta_data['fs']
            signal_len = meta_data['sig_len']
            age = meta_data['comments'][0]
            sex = meta_data['comments'][1]
            dx = meta_data['comments'][2]
            age = age[5:] if age.startswith('Age: ') else np.NaN
            sex = sex[5:] if sex.startswith('Sex: ') else 'Unknown'
            dx = dx[4:] if dx.startswith('Dx: ') else ''
            results.append([patient_id, sample_rate, signal_len, age, sex, dx])
        df = pd.DataFrame(data=results, columns=['patient_id', 'sample_rate', 'signal_len', 'age', 'sex', 'dx'])
        df.sort_values('patient_id').to_csv(reference_csv, index=None)


def gen_label_csv(label_csv, reference_csv, dx_dict, classes):
    if not os.path.exists(label_csv):
        results = []
        df_reference = pd.read_csv(reference_csv)
        for _, row in df_reference.iterrows():
            patient_id = row['patient_id']
            dxs = [dx_dict.get(code, '') for code in row['dx'].split(',')]
            labels = [0] * 9
            for idx, label in enumerate(classes):
                if label in dxs:
                    labels[idx] = 1
            results.append([patient_id] + labels)
        df = pd.DataFrame(data=results, columns=['patient_id'] + classes)
        n = len(df)
        folds = np.zeros(n, dtype=np.int8)
        for i in range(10):
            start = int(n * i / 10)
            end = int(n * (i + 1) / 10)
            folds[start:end] = i + 1
        df['fold'] = np.random.permutation(folds)
        columns = df.columns
        df['keep'] = df[classes].sum(axis=1)
        df = df[df['keep'] > 0]
        df[columns].to_csv(label_csv, index=None)


def gen_data_csv(data_dir, label_csv, seq_len):
    if os.path.exists(label_csv):
        labels = pd.read_csv(label_csv)
        print(labels.shape[0])
        for idx in tqdm(range(labels.shape[0])):
            row = labels.iloc[idx]
            patient_id = row['patient_id']
            data_csv = os.path.join(processed_data_dir, 'spcs-2018', patient_id+'.csv')
            if os.path.exists(data_csv):
                continue
            ecg_data, _ = wfdb.rdsamp(os.path.join(data_dir, patient_id))
            ecg_data = transform(ecg_data, True)
            nsteps, _ = ecg_data.shape
            ecg_data = ecg_data[-seq_len:, -12:]
            result = np.zeros((seq_len, 12))
            result[-nsteps:, :] = ecg_data
            df = pandas.DataFrame(result)
            # print(result)
            df.to_csv(data_csv, header=False, index=False)
    else:
        print(ValueError(f"label_csv not exists"))


# def gen_adj_csv(data_dir, label_csv, seq_len):
#     if os.path.exists(label_csv):
#         labels = pd.read_csv(label_csv)
#         for idx in tqdm(range(labels.shape[0])):
#             row = labels.iloc[idx]
#             patient_id = row['patient_id']
#             data_csv = os.path.join(processed_data_dir, 'spcs-2018', patient_id + '.csv')
#             if os.path.exists(data_csv):
#                 continue
#             ecg_data = np.loadtxt(data_csv, dtype=torch.float)
#
#             df.to_csv(adj_csv, header=False, index=False)
#     else:
#         print(ValueError(f"label_csv not exists"))



if __name__ == "__main__":
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    dx_dict = {
        '426783006': 'SNR', # Normal sinus rhythm
        '164889003': 'AF', # Atrial fibrillation
        '270492004': 'IAVB', # First-degree atrioventricular block
        '164909002': 'LBBB', # Left bundle branch block
        '713427006': 'RBBB', # Complete right bundle branch block
        '59118001': 'RBBB', # Right bundle branch block
        '284470004': 'PAC', # Premature atrial contraction
        '63593006': 'PAC', # Supraventricular premature beats
        '164884008': 'PVC', # Ventricular ectopics
        '429622005': 'STD', # ST-segment depression
        '164931005': 'STE', # ST-segment elevation
    }
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory to dataset')
    # args = parser.parse_args()
    org_data_dir = dataset_path
    processed_data_dir = processed_path
    reference_csv = os.path.join(processed_data_dir, 'reference.csv')
    label_csv = os.path.join(processed_data_dir, 'labels.csv')
    gen_reference_csv(org_data_dir, reference_csv)
    gen_label_csv(label_csv, reference_csv, dx_dict, classes)
    # gen_data_csv(data_dir=org_data_dir, label_csv=label_csv, seq_len=15000)
