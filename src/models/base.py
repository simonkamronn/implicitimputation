import torch


class Base:
    def __init__(self):
        self.model = None
        self.optim = None
        self.recon_loss = torch.nn.MSELoss(size_average=False)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def reset_grad(self):
        self.model.zero_grad()

    def step(self):
        self.optim.step()

    def loss_function(self, recon_x, x, mask):
        return self.recon_loss(recon_x * mask, x * mask)

    def eval_loss(self, x, mask):
        recon_batch, noise = self.model(x)
        return self.loss_function(recon_batch, x, mask).data[0]

    def forward(self, x, mask):
        self.reset_grad()
        recon_batch, noise = self.model(x)
        loss = self.loss_function(recon_batch, x, mask)
        loss.backward()
        self.optim.step()
        return loss.data[0]

    def __repr__(self):
        return self.model.__repr__()

    def __call__(self, *input, **kwargs):
        return self.model(*input, **kwargs)