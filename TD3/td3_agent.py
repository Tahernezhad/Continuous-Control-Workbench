import torch
import numpy as np
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
from networks import Actor, Critic


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent.
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 max_action: float,
                 config):

        self.config = config
        self.device = config.DEVICE
        self.max_action = max_action
        self.n_actions = n_actions

        # --- Initialize Networks ---
        self.actor = Actor(n_states, n_actions).to(self.device)
        self.actor_target = Actor(n_states, n_actions).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE_ACTOR)

        self.critic = Critic(n_states, n_actions).to(self.device)
        self.critic_target = Critic(n_states, n_actions).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE_CRITIC)

        self.total_it = 0

    def select_action(self, state: np.ndarray, noise: float = 0.0) -> np.ndarray:
        """
        Selects an action given the state.
        Args:
            state: The current observation from the environment.
            noise: Magnitude of Gaussian exploration noise (default: 0.0).
        """
        # Ensure state is a float tensor and add batch dimension if needed
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if noise > 0:
            # Correct noise sizing (based on previous bug fix)
            noise_val = np.random.normal(0, noise, size=self.n_actions)
            action = (action + noise_val).clip(-self.max_action, self.max_action)

        return action

    def optimize(self, memory):
        """
        Performs one step of optimization (Critic + Delayed Actor).
        """
        if len(memory) < self.config.BATCH_SIZE:
            return

        self.total_it += 1

        # 1. Sample Replay Buffer
        state, action, next_state, reward, not_done = memory.sample(self.config.BATCH_SIZE)

        # Convert to Tensor (on device)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)

        # 2. Target Policy Smoothing
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.config.POLICY_NOISE).clamp(-self.config.NOISE_CLIP,
                                                                                self.config.NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Twin Critic Target
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.config.GAMMA * target_Q)

        # 3. Critic Update
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 4. Delayed Actor Update
        if self.total_it % self.config.POLICY_DELAY == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft Update
            self._soft_update(self.critic, self.critic_target, self.config.TAU)
            self._soft_update(self.actor, self.actor_target, self.config.TAU)

    def _soft_update(self, local_model, target_model, tau: float):
        for param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_checkpoint(self, path: Path):
        """Saves the actor and critic state dicts."""
        torch.save(self.actor.state_dict(), path / "actor.pth")
        torch.save(self.critic.state_dict(), path / "critic.pth")

    def load_checkpoint(self, path: Path):
        """Loads the actor and critic state dicts."""
        self.actor.load_state_dict(torch.load(path / "actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(path / "critic.pth", map_location=self.device))