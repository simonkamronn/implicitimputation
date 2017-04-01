import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from fastparquet import ParquetFile
import os
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
data_dir = os.path.join(project_dir, 'data', 'interim', 'data.parq')
pf = ParquetFile(data_dir)
data = pf.to_pandas()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(data.values, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader( data.values, batch_size=args.batch_size, shuffle=True, **kwargs)


# Params
noise_variance_var = 0.1


class ImplicitDenoising(nn.Module):
    def __init__(self, num_z=3):
        super(ImplicitDenoising, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.num_z = num_z

        self.g1 = nn.Linear(self.num_z, 20)
        self.g2 = nn.Linear(20, 10)
        self.g3 = nn.Linear(10, 2)

        self.d1 = nn.Linear(2, 20)
        self.d2 = nn.Linear(20, 10)
        self.d3 = nn.Linear(10, 20)
        self.d4 = nn.Linear(20, 2)

    def generator(self, z):
        z = self.relu(self.g1(z))
        z = self.relu(self.g2(z))
        return self.g3(z)

    def denoiser(self, z):
        z = self.relu(self.d1(z))
        z = self.relu(self.d2(z))
        z = self.relu(self.d3(z))
        return self.d4(z)

    def forward(self, x):

        z_rnd = torch.randn(args.batch_size, self.num_z)
        epsilon_rnd = torch.sqrt(noise_variance_var) * torch.randn(args.batchsize, 2)

        samples_from_generator = self.generator(z_rnd)
        noisy_samples_from_generator = samples_from_generator + epsilon_rnd

        denoised_samples_from_generator = self.denoiser(samples_from_generator)
        denoised_noisy_samples_from_generator = self.denoiser(noisy_samples_from_generator)
        return denoised_samples_from_generator, denoised_noisy_samples_from_generator


model = ImplicitDenoising()
if args.cuda:
    model.cuda()

loss_denoiser = nn.MSELoss()
sigmoid = nn.Sigmoid()

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_denoiser(recon_batch, data)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for data, _ in test_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_denoiser(recon_batch, data).data[0]

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)