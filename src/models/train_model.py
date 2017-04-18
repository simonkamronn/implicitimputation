import torch
import torch.utils.data
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
from src.models.avb import AVB
from src.models.config import get_config


def run(args):
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    train_data, test_data, train_mask, test_mask, user_list = load_data(random_split=True)

    # Params
    # n_bins = 288
    n_samples, n_bins, n_mods = train_data.shape
    n_features = n_bins * n_mods
    # n_mods = n_features // n_bins
    modalities = ['cpm', 'steps', 'screen', 'location_lat', 'location_lon'][:n_mods]
    num_train = train_data.shape[0] // args.batch_size
    num_test = test_data.shape[0] // args.batch_size

    # Convert to torch tensor
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)
    train_mask = torch.from_numpy(train_mask).float()
    test_mask = torch.from_numpy(test_mask).float()

    def get_batch(source, mask, i, evaluation=False):
        data = Variable(source[i * args.batch_size:(i + 1) * args.batch_size], volatile=evaluation)
        _mask = Variable(mask[i * args.batch_size:(i + 1) * args.batch_size], volatile=evaluation)
        return data, _mask

    if args.model.lower() == 'vae':
        model = VAE(args.layers, input_dim=n_features, args=args)
    elif args.model.lower() == 'rae':
        model = CRAE(args.layers, input_dim=n_features, args=args)
    elif args.model.lower() == 'unet':
        model = SUnet(args.layers, input_dim=n_features, args=args)
    elif args.model.lower() == 'avb':
        model = AVB(args.layers, input_dim=n_features, args=args)
    else:
        model = SDAE(args.layers, input_dim=n_features, args=args)
    print(model)

    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx in range(num_train):
            data, mask = get_batch(train_data, train_mask, batch_idx, evaluation=False)

            if args.cuda:
                data = data.cuda()

            # Run model updates and collect loss
            loss = model.forward(data, mask)
            train_loss += loss

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_data),
                           100. * batch_idx / num_train,
                           loss / len(data)))

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

            # Evaluate batch on model
            test_loss += model.eval_loss(data, mask)

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
        recon_batch, mu, logvar, noise = model(test_batch, test_mask_batch)
    else:
        recon_batch, noise = model(test_batch, test_mask_batch)

    # Mask out known values
    test_batch = test_batch * (1 - noise) * test_mask_batch
    recon_batch = recon_batch * (1 - noise) * test_mask_batch

    test_batch = test_batch.data.numpy().reshape(-1, n_bins, n_mods)
    recon_batch = recon_batch.data.numpy().reshape(-1, n_bins, n_mods)

    # fig, ax = plt.subplots(nrows=2, ncols=n_mods, figsize=(10 * n_mods, 20))
    # for i, mod in enumerate(modalities):
    #     vmax = np.max((test_batch[:, :, i].max(), recon_batch[:, :, i].max()))
    #     sns.heatmap(test_batch[:, :, i], ax=ax[0, i], vmin=0, vmax=vmax)
    #     sns.heatmap(recon_batch[:, :, i], ax=ax[1, i], vmin=0, vmax=vmax)
    # plt.savefig('{}_recon_heatmap'.format(args.model))
    #
    # # Plot error curves
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.plot(range(args.epochs - 1), train_loss[1:], label='train')
    # ax.plot(range(args.epochs - 1), test_loss[1:], label='test')
    # plt.savefig('{}_error'.format(args.model))

    # Create a visdom object
    vis = Visdom(env=args.model)

    # Heatmap
    for i, mod in enumerate(modalities):
        vmax = np.max((test_batch[:, :, i].max(), recon_batch[:, :, i].max()))
        vis.heatmap(test_batch[:, :, i],
                    opts=dict(colormap='Electric', title='true_' + mod, xmin=0, xmax=float(vmax)))  # , xmax=float(vmax)
        vis.heatmap(recon_batch[:, :, i],
                    opts=dict(colormap='Electric', title='recon_' + mod, xmin=0, xmax=float(vmax)))
    vis.heatmap(((1 - noise) * test_mask_batch)[:, :, 0].data.numpy(), opts=dict(title='mask'))

    # Errors
    vis.line(np.stack((train_loss[1:], test_loss[1:]), axis=1),
             np.tile(np.arange(args.epochs - 1), (2, 1)).transpose(),
             opts=dict(legend=['train', 'test']))

    return train_loss[-1], test_loss[-1]


if __name__ == '__main__':
    run(get_config())
