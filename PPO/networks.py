import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=256):
        super(ActorNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions)
        )

        # Learnable Log Std Dev (State-independent)
        self.log_std = nn.Parameter(torch.ones(1, n_actions) * -0.5)

    def forward(self, state):
        mu = self.net(state)
        std = self.log_std.exp().expand_as(mu)

        # Continuous Distribution
        dist = Normal(mu, std)
        return dist


class CriticNetwork(nn.Module):
    def __init__(self, n_states, hidden_size=256):
        super(CriticNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.net(state)