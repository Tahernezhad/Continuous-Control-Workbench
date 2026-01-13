import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from networks import GaussianActor, TwinCritic


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent with Automatic Entropy Tuning.
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 max_action: float,
                 config):

        self.config = config
        self.device = config.DEVICE
        self.n_actions = n_actions
        self.max_action = max_action
        self.gamma = config.GAMMA
        self.tau = config.TAU

        #  1. Components: Actor & Critics
        self.actor = GaussianActor(n_states, n_actions, config.HIDDEN_SIZE, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE_ACTOR)

        self.critic = TwinCritic(n_states, n_actions, config.HIDDEN_SIZE).to(self.device)
        self.critic_target = TwinCritic(n_states, n_actions, config.HIDDEN_SIZE).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE_CRITIC)

        #  2. Automatic Entropy Tuning (The Improvement)
        # Target Entropy = -dim(A) (heuristic from original paper)
        self.target_entropy = -float(n_actions) * config.TARGET_ENTROPY_SCALE

        # Log Alpha is the learnable parameter (initialized to 0 -> alpha=1.0)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.LEARNING_RATE_ALPHA)

    @property
    def alpha(self):
        """Returns the current value of alpha."""
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state_tensor)
            else:
                action, _, _ = self.actor.sample(state_tensor)

        return action.cpu().data.numpy().flatten()

    def optimize(self, memory):
        if len(memory) < self.config.BATCH_SIZE:
            return

        # Sample Batch
        state, action, next_state, reward, not_done = memory.sample(self.config.BATCH_SIZE)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)

        #  1. Update Alpha (Entropy Temperature)
        with torch.no_grad():
            _, log_prob, _ = self.actor.sample(state)

        # Alpha Loss: Minimize -alpha * (log_pi + target_entropy)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Get current alpha value to detach it for other updates
        curr_alpha = self.alpha.detach()

        #  2. Update Critic
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # Soft Q Target: r + gamma * (min_Q - alpha * entropy)
            target_Q = target_Q - curr_alpha * next_log_prob
            target_Q = reward + (not_done * self.gamma * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  3. Update Actor
        new_action, log_prob, _ = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        # Maximize: Q - alpha * log_prob  --> Minimize: alpha * log_prob - Q
        actor_loss = (curr_alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #  4. Soft Update Targets
        self._soft_update(self.critic, self.critic_target, self.tau)

    def _soft_update(self, local_model, target_model, tau):
        for param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_checkpoint(self, path: Path):
        torch.save(self.actor.state_dict(), path / "actor.pth")
        torch.save(self.critic.state_dict(), path / "critic.pth")
        torch.save(self.log_alpha, path / "log_alpha.pth")

    def load_checkpoint(self, path: Path):
        self.actor.load_state_dict(torch.load(path / "actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(path / "critic.pth", map_location=self.device))
        try:
            self.log_alpha = torch.load(path / "log_alpha.pth", map_location=self.device)
        except FileNotFoundError:
            print(" No saved alpha found. Using initial value.")