import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DAE(nn.Module):
    def __init__(self, params, input_dim, dropout=.0):
        super(DAE, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(input_dim, params[0])
        self.fc2 = nn.Linear(params[0], params[1])
        self.fc3 = nn.Linear(params[1], params[0])
        self.fc4 = nn.Linear(params[0], input_dim)

        self.encoder = nn.Sequential([nn.Linear(params[i], params[i+1]) for i in range(len(params) - 1)])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        # x = F.dropout(x, p=self.dropout, training=True)
        x = x * Variable(torch.zeros(x.size(0), x.size(1)).bernoulli_(1 - self.dropout))
        z = self.encode(x)
        return self.decode(z)