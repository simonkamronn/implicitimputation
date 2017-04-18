import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.autograd import Variable
from torchvision import datasets, transforms

z_dim = 10
eps_dim = 4


def log(x):
    return torch.log(x + 1e-8)


class AVB:
    def __init__(self, params, input_dim, args):
        self.dropout = args.dropout

        # Encoder: q(z|x,eps)
        self.Q = torch.nn.Sequential(
            torch.nn.Linear(input_dim + eps_dim, params[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(params[0], z_dim)
        )

        # Decoder: p(x|z)
        self.P = torch.nn.Sequential(
            torch.nn.Linear(z_dim, params[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(params[0], input_dim),
            torch.nn.Sigmoid()
        )

        # Discriminator: T(X, z)
        self.T = torch.nn.Sequential(
            torch.nn.Linear(input_dim + z_dim, params[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(params[0], 1)
        )

        self.Q_solver = optim.Adam(self.Q.parameters(), lr=args.lr)
        self.P_solver = optim.Adam(self.P.parameters(), lr=args.lr)
        self.T_solver = optim.Adam(self.T.parameters(), lr=args.lr)

        self.recon_loss = nn.MSELoss(size_average=False)
        # self.recon_loss = nn.CrossEntropyLoss(size_average=False)
        # self.recon_loss = lambda x1, x2: F.binary_cross_entropy(x1, x2, size_average=False)

    def train(self):
        self.Q.train()
        self.P.train()
        self.T.train()

    def eval(self):
        self.Q.eval()
        self.P.eval()
        self.T.eval()

    def vae_step(self):
        self.Q_solver.step()
        self.P_solver.step()

    def discriminator_step(self):
        self.T_solver.step()

    def reset_grad(self):
        self.Q.zero_grad()
        self.P.zero_grad()
        self.T.zero_grad()

    def discriminator_loss(self, x, eps, z):
        z_sample = self.Q(torch.cat([x, eps], 1))
        t_q = F.sigmoid(self.T(torch.cat([x, z_sample], 1)))
        t_prior = F.sigmoid(self.T(torch.cat([x, z], 1)))
        return -torch.mean(log(t_q) + log(1. - t_prior))

    def loss_function(self, recon_x, x, mask):
        return self.recon_loss(recon_x * mask, x * mask)

    def sample(self, x):
        mb_size = x.size(0)
        eps = Variable(torch.randn(mb_size, eps_dim))
        z = Variable(torch.randn(mb_size, z_dim))

        # Sample from the models
        z_sample = self.Q(torch.cat([x, eps], 1))
        x_sample = self.P(z_sample)
        t_sample = self.T(torch.cat([x, z_sample], 1))
        return z_sample, x_sample, t_sample, eps, z

    def sample_recon(self, x):
        noise = Variable(torch.zeros(x.size(0), x.size(1)).bernoulli_(1 - self.dropout))
        x = x * noise

        z_sample, x_sample, t_sample, eps, z = self.sample(x)
        return x_sample, noise

    def eval_loss(self, x, mask):
        recon_batch, noise = self.sample_recon(x)
        return self.loss_function(recon_batch, x, mask).data[0]

    def forward(self, x, mask=None):
        # Apply noise to the input
        noise = Variable(torch.zeros(x.size(0), x.size(1)).bernoulli_(1 - self.dropout))
        x_noisy = x * noise

        # Sample from the model
        z_sample, x_sample, t_sample, eps, z = self.sample(x_noisy)

        # Get ELBO
        disc = torch.mean(-t_sample)
        loglike = self.loss_function(x_sample, x, mask) / x.size(0)
        elbo = -(disc + loglike)

        # Update VAE part
        elbo.backward()
        self.vae_step()
        self.reset_grad()

        # Discriminator loss
        t_loss = self.discriminator_loss(x_noisy, eps, z)

        # Update discriminator
        t_loss.backward()
        self.discriminator_step()
        self.reset_grad()

        return (elbo + t_loss).data[0]

    def __call__(self, *input, **kwargs):
        return self.sample_recon(*input, **kwargs)

if __name__ == '__main__':
    import os
    import sys

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    sys.path.append(os.path.join(project_dir, 'src'))
    from src.models.config import get_config

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=32, shuffle=True, **{})
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=32, shuffle=True, **{})

    cnt = 0
    mb_size = 32
    model = AVB([10], 28*28, get_config())

    for it in range(10):
        for X, y in iter(train_loader):
            X = Variable(X.view(mb_size, -1))
            elbo, t_loss = model.forward(X)

            # Print and plot every now and then
            if it % 1000 == 0:
                print('Iter-{}; ELBO: {:.4}; T_loss: {:.4}'
                      .format(it, -elbo.data[0], -t_loss.data[0]))

                z = Variable(torch.randn(mb_size, z_dim))
                samples = model.P(z).data.numpy()[:16]

                fig = plt.figure(figsize=(4, 4))
                gs = gridspec.GridSpec(4, 4)
                gs.update(wspace=0.05, hspace=0.05)

                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

                if not os.path.exists('out/'):
                    os.makedirs('out/')

                plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
                cnt += 1
                plt.close(fig)