from fastparquet import ParquetFile
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from src.data.make_dataset import DataLoader


def load_all():
    df = ParquetFile(os.path.join(project_dir, 'data', 'interim', 'data.parq')).to_pandas().set_index('date')
    data_size = df.groupby(['modality', 'user']).size().unstack()


def parquet_heatmap():
    pf = ParquetFile(os.path.join(project_dir, 'data', 'interim', 'data.parq'))
    df = pf.to_pandas(filters=[('user', '==', 194), ('modality', '==', 'cpm')]).set_index('date')  # .drop(['modality', 'user'], axis=1)
    print(df.shape)

    data = DataLoader.convert_to_npy(df, save=False)
    p = sns.heatmap(np.nan_to_num(data[:, :, 0]))
    plt.show(p)


def npy_heatmap():
    data = np.load(os.path.join(project_dir, 'data', 'interim', 'data.npy')).astype(np.float32)[:, :-1]
    print(data.shape)
    p = sns.heatmap(data[:100, :, 0])
    plt.show(p)


if __name__=='__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    df = ParquetFile(os.path.join(project_dir, 'data', 'interim', 'data.parq')).to_pandas(filters=[('user', '==', 194)]).set_index('date')

    modality_data = list()
    for modality, m_group in df.groupby('modality'):
        modality_data.append(m_group.drop(['modality', 'user'], axis=1))

    # We concatenate on dates to ensure the same dimension across modalities
    fig, ax = plt.subplots(ncols=2, figsize=(10, 30))
    sns.heatmap(pd.concat(modality_data, axis=1).values.reshape(-1, 6, 288)[:, -1, :], ax=ax[0])
    sns.heatmap(modality_data[-1], ax=ax[1])
    plt.show(fig)