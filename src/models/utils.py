import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(random_split=True):
    # Load data
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    data = np.load(os.path.join(project_dir, 'data', 'interim', 'data.npy')).astype(np.float32)

    # Extract user id
    user_list = data[:, -1, 0]
    data = data[:, :-1]

    # Data shapes
    n_samples, n_bins, n_mods = data.shape
    n_features = n_bins * n_mods

    # Require samples for all modalities
    # data[np.isnan(data).sum(axis=2) > 0] = np.nan

    # Remove outliers
    # TODO: Remove outliers
    # data[data > np.nanmean(data, axis=(0, 1), keepdims=True) + 2*np.nanstd(data, axis=(0, 1), keepdims=True)] = np.nan

    # Create mask
    mask = ~np.isnan(data).reshape(n_samples, n_features) * 1.
    print('Ratio of data to nan: {}'.format(np.mean(mask)))

    # Normalize
    data = np.nan_to_num(data)
    data = MinMaxScaler().fit_transform(data.reshape(n_samples * n_bins, n_mods)).reshape(n_samples, n_features)

    if random_split:
        train_data, test_data, train_mask, test_mask = train_test_split(data, mask, train_size=0.9)
    else:
        train_data, test_data = data[:int(n_samples * 0.9)], data[int(n_samples * 0.9):]
        train_mask, test_mask = mask[:int(n_samples * 0.9)], mask[int(n_samples * 0.9):]

    return train_data, test_data, train_mask, test_mask, user_list