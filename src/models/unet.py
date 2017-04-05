import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Unet(nn.Module):
    def __init__(self, params, input_dim):
        super(Unet, self).__init__()
        params = (input_dim,) + tuple(params)
        self.output_activation = nn.Sigmoid()

        self.encoder = nn.ModuleList()
        for idx, (p1, p2) in enumerate(zip(params[:-1], params[1:])):
            self.encoder.append(nn.Linear(p1, p2))

        params = list(reversed(params))
        self.decoder = nn.ModuleList()
        for idx, (p1, p2) in enumerate(zip(params[:-1], params[1:])):
            self.decoder.append(nn.Linear(p1, p2))

        # Gather layers for decode op
        self.layers = list()

    def encode(self, x):
        for idx, m in enumerate(self.encoder):
            x = m(x)
            if idx < len(self.encoder) - 1:
                x = F.relu(x)
            self.layers.append(x)
        return x

    def decode(self, z):
        for idx, (m, l) in enumerate(zip(self.decoder, reversed(self.layers))):
            z = m(z + l)
            if idx < len(self.decoder) - 1:
                z = F.relu(z)
        return self.output_activation(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z) + x


class SUnet(nn.Module):
    def __init__(self, params, input_dim, dropout=.0, num_blocks=1):
        super(SUnet, self).__init__()
        self.dropout = dropout
        self.das = nn.Sequential(*[Unet(params, input_dim) for _ in range(num_blocks)])

    def forward(self, x):
        x = x * Variable(torch.zeros(x.size(0), x.size(1)).bernoulli_(1 - self.dropout))
        return self.das(x)
