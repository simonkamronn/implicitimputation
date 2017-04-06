import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import np

from visdom import Visdom
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(os.path.join(project_dir, 'src'))

from src.models.utils import load_data
from src.models.crae import CRAE
from src.models.vae import VAE
from src.models.dae import SDAE
from src.models.unet import SUnet
from src.models.config import get_config


def run(args):
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    train_data, test_data, train_mask, test_mask, user_list = load_data(random_split=True)

    # Params
    n_bins = 288
    n_samples, n_features = train_data.shape
    n_mods = n_features // n_bins
    num_train = train_data.shape[0] // args.batch_size
    num_test = test_data.shape[0] // args.batch_size

    # Convert to torch tensor
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)
    train_mask = torch.from_numpy(train_mask).float()
    test_mask = torch.from_numpy(test_mask).float()

    # Mean Squared Error loss for reconstruction
    reconstruction_loss = torch.nn.MSELoss(size_average=False)

    if 'vae' in args.model:
        def loss_function(recon_x, x, mu, logvar, mask):
            recon_loss = reconstruction_loss(recon_x * mask, x * mask)

            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD = torch.sum(KLD_element).mul_(-0.5)

            return recon_loss + KLD
    else:
        def loss_function(recon_x, x, mask):
            recon_loss = reconstruction_loss(recon_x * mask, x * mask)
            return recon_loss

    def get_batch(source, mask, i, evaluation=False):
        data = Variable(source[i * args.batch_size:(i + 1) * args.batch_size], volatile=evaluation)
        _mask = Variable(mask[i * args.batch_size:(i + 1) * args.batch_size], volatile=evaluation)
        return data, _mask

    model = nn.Module()
    if args.model.lower() == 'vae':
        model = VAE((400, 200, 100), input_dim=n_features, dropout=args.dropout, args=args)
    elif args.model.lower() == 'rae':
        model = CRAE((400, 200, 50), input_dim=n_features, dropout=args.dropout, num_blocks=2)
    elif args.model.lower() == 'unet':
        model = SUnet((400, 200, 100, 50), input_dim=n_features, dropout=args.dropout, num_blocks=1)
    else:
        model = SDAE((400, 100, 20), input_dim=n_features, dropout=args.dropout, num_blocks=2)

    print(model)

    # Create optimizer over model parameters
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx in range(num_train):
            data, mask = get_batch(train_data, train_mask, batch_idx, evaluation=False)

            if args.cuda:
                data = data.cuda()

            optimizer.zero_grad()
            if 'vae' in args.model:
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar, mask)
            else:
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

        print('====> Epoch: {} Average loss: {:.6f}'.format(
            epoch, train_loss / len(train_data)))
        return train_loss / len(train_data)

    def test(epoch):
        model.eval()
        test_loss = 0
        for batch_idx in range(num_test):
            data, mask = get_batch(test_data, test_mask, batch_idx, evaluation=True)
            if args.cuda:
                data = data.cuda()
            if 'vae' in args.model:
                recon_batch, mu, logvar = model(data)
                test_loss += loss_function(recon_batch, data, mu, logvar, mask).data[0]
            else:
                recon_batch = model(data)
                test_loss += loss_function(recon_batch, data, mask).data[0]

        test_loss /= len(test_data)
        print('====> Test set loss: {:.6f}'.format(test_loss))
        return test_loss

    train_loss = list()
    test_loss = list()
    for epoch in range(1, args.epochs + 1):
        train_loss.append(train(epoch))
        test_loss.append(test(epoch))

    # Plot result
    test_batch, test_mask_batch = get_batch(test_data, test_mask, 0, evaluation=True)

    if 'vae' in args.model:
        recon_batch, mu, logvar = model(test_batch)
    else:
        recon_batch = model(test_batch)

    test_batch = test_batch.data.numpy().reshape(-1, n_bins, n_mods)
    recon_batch = recon_batch.data.numpy().reshape(-1, n_bins, n_mods)

    fig, ax = plt.subplots(nrows=2, ncols=n_mods, figsize=(10 * n_mods, 20))
    for i, mod in enumerate(('cpm', 'steps')):
        vmax = np.max((test_batch[:, :, i].max(), recon_batch[:, :, i].max()))
        sns.heatmap(test_batch[:, :, i], ax=ax[0, i], vmin=0, vmax=vmax)
        sns.heatmap(recon_batch[:, :, i], ax=ax[1, i], vmin=0, vmax=vmax)
    plt.savefig('{}_recon_heatmap'.format(args.model))

    # Plot error curves
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(range(args.epochs - 1), train_loss[1:], label='train')
    ax.plot(range(args.epochs - 1), test_loss[1:], label='test')
    plt.savefig('{}_error'.format(args.model))

    # Create a visdom object
    vis = Visdom(env=args.model)

    # Heatmap
    for i, mod in enumerate(('cpm', 'steps')):
        vmax = np.max((test_batch[:, :, i].max(), recon_batch[:, :, i].max()))
        vis.heatmap(test_batch[:, :, i], opts=dict(colormap='Electric', title='true_' + mod, xmin=0, xmax=float(vmax)))
        vis.heatmap(recon_batch[:, :, i], opts=dict(colormap='Electric', title='recon_' + mod, xmin=0, xmax=float(vmax)))

    # Errors
    vis.line(np.stack((train_loss[1:], test_loss[1:]), axis=1),
             np.tile(np.arange(args.epochs - 1), (2, 1)).transpose(),
             opts=dict(legend=['train', 'test']))


if __name__ == '__main__':
    run(get_config())
