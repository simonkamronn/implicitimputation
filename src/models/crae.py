import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from .base import Base
from torch import optim


class RAE_module(nn.Module):
    def __init__(self, params, input_dim):
        super(RAE_module, self).__init__()
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

    def forward(self, x):
        return self.decoder(self.encoder(x)) + x


class CRAE_module(nn.Module):
    def __init__(self, params, input_dim, dropout=.0, num_blocks=1):
        super(CRAE_module, self).__init__()
        self.params = params
        self.input_dim = input_dim
        self.dropout = dropout
        self.ras = nn.ModuleList([RAE_module(params, input_dim) for _ in range(num_blocks)])

    def forward(self, x):
        noise = Variable(torch.zeros(x.size(0), self.input_dim).bernoulli_(1 - self.dropout))
        x_hat = x * noise
        for ra in self.ras:
            x_hat = ra(x_hat)
        return x_hat


class CRAE(Base):
    def __init__(self, params, input_dim, args):
        super(CRAE, self).__init__()
        self.model = CRAE_module(params, input_dim, args.dropout, args.blocks)
        self.optim = optim.Adam(self.model.parameters(), lr=args.lr)
