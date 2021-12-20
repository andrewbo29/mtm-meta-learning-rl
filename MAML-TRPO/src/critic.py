import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from src.constants import CRITIC_LR


# Value Estimator
class Critic(nn.Module):
    def __init__(self, state_dim, lr=CRITIC_LR, device='cpu'):
        super().__init__()
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        ).to(device)

        self.lr = lr
        self.optimizer = Adam(self.model.parameters(), lr)

    def __del__(self):
        del self.device, self.lr, self.optimizer
        del self.model

    def forward(self, state):
        return self.model(state)

    def update(self, states, rewards):
        loss = F.mse_loss(self.model(states).flatten(), torch.squeeze(rewards))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
