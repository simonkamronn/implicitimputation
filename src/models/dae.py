import torch
import torch.nn as nn
from torch.autograd import Variable
from .base import Base
import torch.optim as optim


class SDAE_module(nn.Module):
    def __init__(self, params, input_dim, args):
        super(SDAE_module, self).__init__()
        self.args = args
        self.input_dim = input_dim
        params = (input_dim*2,) + tuple(params)

        self.encoder = nn.Sequential()
        for idx, (p1, p2) in enumerate(zip(params[:-1], params[1:])):
            self.encoder.add_module('encoder_{}_linear'.format(idx), nn.Linear(p1, p2))
            if idx < len(params) - 2:
                self.encoder.add_module('encoder_{}_relu'.format(idx), nn.ReLU())
                # self.encoder.add_module('encoder_{}_dropout'.format(idx), nn.Dropout(p=args.dropout))

        self.encoder.add_module('encoder_sigmoid', nn.Sigmoid())

        params = list(reversed(params))
        self.decoder = nn.Sequential()
        for idx, (p1, p2) in enumerate(zip(params[:-2], params[1:-1])):
            self.decoder.add_module('decoder_{}_linear'.format(idx), nn.Linear(p1, p2))
            if idx < len(params) - 2:
                self.decoder.add_module('decoder_{}_relu'.format(idx), nn.ReLU())
                # self.decoder.add_module('decoder_{}_dropout'.format(idx), nn.Dropout(p=args.dropout))

        self.decoder.add_module('decoder_out_linear', nn.Linear(params[-2], input_dim))
        self.decoder.add_module('decoder_sigmoid', nn.Sigmoid())

    def encode(self, x, mask):
        return self.encoder(torch.cat((x, mask), 2).view(-1, self.input_dim*2))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, mask):
        noise = Variable(torch.zeros(x.size(0), x.size(1)).bernoulli_(1 - self.args.dropout))\
            .repeat(1, 2).view(x.size(0), x.size(1), -1)
        z = self.encode(x * noise, (1 - noise) * mask)
        return self.decode(z).view(x.size(0), x.size(1), -1), noise


class SDAE(Base):
    def __init__(self, params, input_dim, args):
        super(SDAE, self).__init__()
        self.model = SDAE_module(params, input_dim, args)
        self.optim = optim.Adam(self.model.parameters(), lr=args.lr)