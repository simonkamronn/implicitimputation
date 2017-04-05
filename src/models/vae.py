import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, params, input_dim, dropout=.0, args=None):
        super(VAE, self).__init__()
        self.args = args
        self.dropout = dropout
        params = (input_dim,) + tuple(params)

        self.encoder = nn.Sequential()
        for idx, (p1, p2) in enumerate(zip(params[:-2], params[1:-1])):
            self.encoder.add_module('encoder_{}_linear'.format(str(idx)), nn.Linear(p1, p2))
            self.encoder.add_module('encoder_{}_relu'.format(str(idx)), nn.ReLU())

        self.mu = nn.Linear(params[-2], params[-1])
        self.logvar = nn.Linear(params[-2], params[-1])

        params = list(reversed(params))
        self.decoder = nn.Sequential()
        for idx, (p1, p2) in enumerate(zip(params[:-1], params[1:])):
            self.decoder.add_module('decoder_{}_linear'.format(str(idx)), nn.Linear(p1, p2))
            if idx < len(params) - 2:
                self.decoder.add_module('decoder_{}_relu'.format(str(idx)), nn.ReLU())
        self.decoder.add_module('decoder_{}_sigmoid'.format(str(idx)), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = x * Variable(torch.zeros(x.size(0), x.size(1)).bernoulli_(1 - self.dropout))
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
