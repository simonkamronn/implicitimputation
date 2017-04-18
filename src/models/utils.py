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

    # Limit data
    data = data[:, :, :2]

    # Require samples for all modalities
    data[np.isnan(data).sum(axis=2) > 0, :] = np.nan

    # Require a certain amount of samples during a day
    data[np.isnan(data[:, :, 0]).sum(axis=1) > 250, :, :] = np.nan

    # Select data
    data = data[(~np.isnan(data)).sum(axis=(1, 2)) > 0]

    # Remove outliers
    # TODO: Remove outliers
    # data[data > np.nanmean(data, axis=(0, 1), keepdims=True) + 2*np.nanstd(data, axis=(0, 1), keepdims=True)] = np.nan

    # Data shapes
    n_samples, n_bins, n_mods = data.shape
    n_features = n_bins * n_mods

    # Create mask
    mask = ~np.isnan(data) * 1.
    print('Ratio of data to nan: {}'.format(np.mean(mask)))

    # Clip step count
    data[data[:, :, 1] > 2000, 1] = 2000.0

    # Clip activity level
    data[data[:, :, 0] > 150000, 0] = 1500000

    # Log transform data
    data += 1
    data = np.log(data)

    # Fill nan
    data[np.isnan(data)] = 0
    data[data == -np.inf] = 0

    # Normalize to 0 - 1
    data = MinMaxScaler().fit_transform(data.reshape(n_samples * n_bins, n_mods)).reshape(n_samples, n_bins, n_mods)

    # Replace nans with -1
    data[mask == 0] = -1

    if random_split:
        train_data, test_data, train_mask, test_mask = train_test_split(data, mask, train_size=0.9)
    else:
        train_data, test_data = data[:int(n_samples * 0.9)], data[int(n_samples * 0.9):]
        train_mask, test_mask = mask[:int(n_samples * 0.9)], mask[int(n_samples * 0.9):]

    return train_data, test_data, train_mask, test_mask, user_list


class InfiniteDataLoader(object):
    """docstring for InfiniteDataLoader"""

    def __init__(self, dataloader):
        super(InfiniteDataLoader, self).__init__()
        self.dataloader = dataloader
        self.data_iter = None

    def next(self):
        try:
            data = self.data_iter.next()
        except Exception:
            # Reached end of the dataset
            self.data_iter = iter(self.dataloader)
            data = self.data_iter.next()

        return data

    def __len__(self):
        return len(self.dataloader)


def normalize_(x, dim=1):
    '''
    Projects points to a sphere inplace.
    '''
    x.div_(x.norm(2, dim=dim).expand_as(x))


def normalize(x, dim=1):
    '''
    Projects points to a sphere.
    '''
    return x.div(x.norm(2, dim=dim).expand_as(x))


def var(x, dim=0):
    '''
    Calculates variance.
    '''
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)


def match(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    '''
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)

        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none', 'wtf ?'


def weights_init(m):
    '''
    Custom weights initialization called on netG and netE
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
