import torch
import torch.nn as nn
from torch.autograd import Variable
from .base import Base
import torch.optim as optim


class DAE_module(nn.Module):
    def __init__(self, params, input_dim):
        super(DAE_module, self).__init__()
        params = (input_dim,) + tuple(params)

        self.encoder = nn.Sequential()
        for idx, (p1, p2) in enumerate(zip(params[:-1], params[1:])):
            self.encoder.add_module('encoder_{}_linear'.format(str(idx)), nn.Linear(p1, p2))
            if idx < len(params) - 2:
                self.encoder.add_module('encoder_{}_relu'.format(str(idx)), nn.ReLU())

        params = list(reversed(params))
        self.decoder = nn.Sequential()
        for idx, (p1, p2) in enumerate(zip(params[:-1], params[1:])):
            self.decoder.add_module('decoder_{}_linear'.format(str(idx)), nn.Linear(p1, p2))
            if idx < len(params) - 2:
                self.decoder.add_module('decoder_{}_relu'.format(str(idx)), nn.ReLU())
        self.decoder.add_module('decoder_{}_sigmoid'.format(str(idx)), nn.Sigmoid())

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class SDAE_module(nn.Module):
    def __init__(self, params, input_dim, dropout=.0, num_blocks=1):
        super(SDAE_module, self).__init__()
        self.dropout = dropout
        self.das = nn.Sequential(*[DAE_module(params, input_dim) for _ in range(num_blocks)])

    def forward(self, x):
        noise = Variable(torch.zeros(x.size(0), x.size(1)).bernoulli_(1 - self.dropout))
        x = x * noise
        return self.das(x), noise


class SDAE(Base):
    def __init__(self, params, input_dim, args):
        super(SDAE, self).__init__()
        self.model = SDAE_module(params, input_dim, args.dropout, args.blocks)
        self.optim = optim.Adam(self.model.parameters(), lr=args.lr)