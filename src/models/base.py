import torch
from torch import nn


class Base:
    def __init__(self):
        self.model = None
        self.optim = None
        # self.recon_loss = nn.MSELoss(size_average=False)
        self.recon_loss = nn.BCELoss(size_average=False)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def reset_grad(self):
        self.model.zero_grad()
        self.optim.zero_grad()

    def step(self):
        self.optim.step()

    def loss_function(self, recon_x, x, mask):
        return self.recon_loss(recon_x * mask, x * mask) / mask.sum()

    def eval_loss(self, x, mask):
        recon_batch, noise = self.model(x, mask)
        return self.loss_function(recon_batch, x, mask * (1 - noise)).data[0]

    def forward(self, x, mask):
        self.reset_grad()
        recon_batch, noise = self.model(x, mask)
        loss = self.loss_function(recon_batch, x, mask * (1 - noise))
        loss.backward()
        self.optim.step()
        return loss.data[0]

    def __repr__(self):
        return self.model.__repr__()

    def __call__(self, *input, **kwargs):
        return self.model(*input, **kwargs)