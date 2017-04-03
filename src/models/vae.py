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
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dropout', type=float, default=.1, metavar='dropout', help='input dropout')
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

# Normalize
data = np.nan_to_num(data)
data = MinMaxScaler().fit_transform(data.reshape(n_samples*n_bins, n_mods)).reshape(n_samples, n_features)

# Random split
# train_data, test_data, train_mask, test_mask = train_test_split(data, mask, train_size=0.9)

# Sequential split
train_data, test_data = data[:int(n_samples * 0.9)], data[int(n_samples * 0.9):]
train_mask, test_mask = mask[:int(n_samples * 0.9)], mask[int(n_samples * 0.9):]

num_train = train_data.shape[0] // args.batch_size
num_test = test_data.shape[0] // args.batch_size

# Convert to torch tensor
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
train_mask = torch.from_numpy(train_mask).float()
test_mask = torch.from_numpy(test_mask).float()


class VAE(nn.Module):
    def __init__(self, dropout=.0):
        super(VAE, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(n_features, 400)
        self.fc21 = nn.Linear(400, 10)
        self.fc22 = nn.Linear(400, 10)
        self.fc3 = nn.Linear(10, 400)
        self.fc4 = nn.Linear(400, n_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=True)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(dropout=args.dropout)
if args.cuda:
    model.cuda()


def get_batch(source, mask, i, evaluation=False):
    data = Variable(source[i * args.batch_size:(i + 1) * args.batch_size], volatile=evaluation)
    _mask = Variable(mask[i * args.batch_size:(i + 1) * args.batch_size], volatile=evaluation)
    return data, _mask

# Mean Squared Error loss for reconstruction
reconstruction_loss = torch.nn.MSELoss()
reconstruction_loss.sizeAverage = False


def loss_function(recon_x, x, mu, logvar, mask):
    recon_loss = reconstruction_loss(recon_x * mask, x * mask) / mask.sum()

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return recon_loss + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in range(num_train):
        data, mask = get_batch(train_data, train_mask, batch_idx, evaluation=False)

        if args.cuda:
            data = data.cuda()

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, mask)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data),
                100. * batch_idx / num_train,
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.6f}'.format(
          epoch, train_loss / len(train_data)))


def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx in range(num_test):
        data, mask = get_batch(test_data, test_mask, batch_idx, evaluation=True)
        if args.cuda:
            data = data.cuda()
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar, mask).data[0]

    test_loss /= len(test_data)
    print('====> Test set loss: {:.6f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

# Plot result
test_batch, test_mask_batch = get_batch(test_data, test_mask, 1, evaluation=True)
recon_batch, mu, logvar = model(test_batch)

fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(60, 20))
for i in range(6):
    sns.heatmap((test_batch * test_mask_batch).data.numpy().reshape(-1, n_bins, n_mods)[:, :, i], ax=ax[0, i])
    sns.heatmap((recon_batch * test_mask_batch).data.numpy().reshape(-1, n_bins, n_mods)[:, :, i], ax=ax[1, i])

plt.savefig('recon_heatmap')