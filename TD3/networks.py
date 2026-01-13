import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 400):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(n_states, hidden_size)
        self.l2 = nn.Linear(hidden_size, 300)
        self.l3 = nn.Linear(300, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # Tanh maps output to [-1, 1], scaling handled in Agent
        return torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 400):
        super(Critic, self).__init__()

        # Q1 Architecture
        self.l1 = nn.Linear(n_states + n_actions, hidden_size)
        self.l2 = nn.Linear(hidden_size, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 Architecture (Twin Critic)
        self.l4 = nn.Linear(n_states + n_actions, hidden_size)
        self.l5 = nn.Linear(hidden_size, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)

        # Q1 Forward
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Q2 Forward
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Used when we only need one Q-value (e.g., for Actor loss calculation)"""
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1