import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from torch.distributions import Normal


class GaussianActor(nn.Module):
    """
    Stochastic Actor that outputs a Gaussian distribution (Mean + Std Dev).
    """

    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 256, max_action: float = 1.0):
        super(GaussianActor, self).__init__()
        self.max_action = max_action

        self.l1 = nn.Linear(n_states, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, n_actions)
        self.log_std = nn.Linear(hidden_size, n_actions)

        # Clip log_std to prevent numerical instability
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        mu = self.mean(x)
        log_std = self.log_std(x)

        # Clamp std dev to maintain stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: Tanh-squashed action (for environment)
            log_prob: Log probability of that action (for loss calc)
            mean: Raw mean (for evaluation)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()

        dist = Normal(mu, std)
        z = dist.rsample()  # Reparameterization trick (allows gradients to flow)

        # Tanh squashing to keeping action in [-1, 1]
        action = torch.tanh(z)

        # Enforce action bounds (if max_action != 1)
        action_env = action * self.max_action

        # Correction formula for Tanh squashing (from original SAC paper)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action_env, log_prob, torch.tanh(mu) * self.max_action


class TwinCritic(nn.Module):
    """
    Twin Q-Networks to mitigate overestimation bias.
    """

    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 256):
        super(TwinCritic, self).__init__()

        # Q1 Architecture
        self.l1 = nn.Linear(n_states + n_actions, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        # Q2 Architecture
        self.l4 = nn.Linear(n_states + n_actions, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2