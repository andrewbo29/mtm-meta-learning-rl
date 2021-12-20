import math

import torch
from torch import nn
from torch.distributions import Normal

from src.constants import EPSILON


# Policy / Learner
class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, device=torch.device('cpu')):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        ).to(self.device)

        self.sigma = nn.Parameter(torch.zeros(output_size))

    def __del__(self):
        del self.input_size, self.output_size, self.hidden_size
        del self.device, self.sigma
        del self.model

    def distribution(self, state):
        state = state.to(self.device, non_blocking=True)
        loc = self.model(state)
        scale = torch.exp(torch.clip(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def compute_proba(self, state, action):
        distribution = self.distribution(state)
        return torch.mean(distribution.log_prob(action), dim=1, keepdim=True)

    def forward(self, state):
        return self.distribution(state).sample()

    def loss(self, log_probs, advantages, new_log_probs=None):
        x = log_probs
        if new_log_probs is not None:
            x = torch.exp(new_log_probs - log_probs)
        return -torch.mean(x * advantages)
