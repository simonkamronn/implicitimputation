import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dropout', type=float, default=.8, metavar='dropout', help='input dropout')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
data = np.load(os.path.join(project_dir, 'data', 'interim', 'data.npy')).astype(np.float32)

# Extract user id
user_list = data[:, -1, 0]
data = data[:, :-1]

# Data shapes
n_samples, n_bins, n_mods = data.shape
n_features = n_bins * n_mods

# Create mask
mask = ~np.isnan(data).reshape(n_samples, n_features) * 1.
# print(mask.sum(axis=1))

# Normalize
data = np.nan_to_num(data)
data = MinMaxScaler().fit_transform(data.reshape(n_samples*n_bins, n_mods)).reshape(n_samples, n_features)

# Random split
train_data, test_data, train_mask, test_mask = train_test_split(data, mask, train_size=0.9)

# Sequential split
# train_data, test_data = data[:int(n_samples * 0.9)], data[int(n_samples * 0.9):]
# train_mask, test_mask = mask[:int(n_samples * 0.9)], mask[int(n_samples * 0.9):]

num_train = train_data.shape[0] // args.batch_size
num_test = test_data.shape[0] // args.batch_size

# Convert to torch tensor
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
train_mask = torch.from_numpy(train_mask).float()
test_mask = torch.from_numpy(test_mask).float()


class RA(nn.Module):
    def __init__(self, params):
        super(RA, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, params[0]), nn.ReLU(),
                                     nn.Linear(params[0], params[1]), nn.ReLU())

        self.decoder = nn.Sequential(nn.Linear(params[1], params[0]), nn.ReLU(),
                                     nn.Linear(params[0], n_features),
                                     nn.Softmax())

    def forward(self, x):
        return self.decoder(self.encoder(x)) + x


class CRA(nn.Module):
    def __init__(self, dropout=.0, num_ra=1):
        super(CRA, self).__init__()
        self.dropout = dropout
        self.ras = nn.ModuleList([RA((300, 100, 10)) for _ in range(num_ra)])

    def forward(self, x):
        x_hat = x * Variable(torch.zeros(args.batch_size, n_features).bernoulli_(1 - self.dropout))
        # x_hat = F.dropout(x, self.dropout, training=True)
        # print(x_hat.eq(x).sum())

        for ra in self.ras:
            x_hat = ra(x_hat)
        return x_hat


model = CRA(dropout=args.dropout, num_ra=4)
if args.cuda:
    model.cuda()
print(model)


def get_batch(source, mask, i, evaluation=False):
    data = Variable(source[i * args.batch_size:(i + 1) * args.batch_size], volatile=evaluation)
    _mask = Variable(mask[i * args.batch_size:(i + 1) * args.batch_size], volatile=evaluation)
    return data, _mask

# Mean Squared Error loss for reconstruction
reconstruction_loss = torch.nn.MSELoss()
reconstruction_loss.sizeAverage = False


def loss_function(recon_x, x, mask):
    recon_loss = reconstruction_loss(recon_x * mask, x * mask)
    return recon_loss

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in range(num_train):
        data, mask = get_batch(train_data, train_mask, batch_idx, evaluation=False)

        if args.cuda:
            data = data.cuda()

        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data, mask)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data),
                100. * batch_idx / num_train,
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_data)))


def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx in range(num_test):
        data, mask = get_batch(test_data, test_mask, batch_idx, evaluation=True)
        if args.cuda:
            data = data.cuda()
        recon_batch = model(data)
        test_loss += loss_function(recon_batch, data, mask).data[0]

    test_loss /= len(test_data)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

# Plot result
test_batch, test_mask_batch = get_batch(test_data, test_mask, 1, evaluation=True)
recon_batch = model(test_batch)

fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(60, 20))
for i in range(6):
    sns.heatmap((test_batch * test_mask_batch).data.numpy().reshape(-1, n_bins, n_mods)[:, :, i], ax=ax[0, i])
    sns.heatmap((recon_batch * test_mask_batch).data.numpy().reshape(-1, n_bins, n_mods)[:, :, i], ax=ax[1, i])

plt.savefig('recon_heatmap')