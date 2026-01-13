import torch
import numpy as np
import torch.optim as optim
from networks import ActorNetwork, CriticNetwork


class PPOAgent:
    def __init__(self, n_states, n_actions, config):
        self.gamma = config.GAMMA
        self.policy_clip = config.CLIP_EPSILON
        self.n_epochs = config.K_EPOCHS
        self.gae_lambda = config.GAE_LAMBDA
        self.batch_size = config.BATCH_SIZE
        self.entropy_coef = config.ENTROPY_COEF
        self.device = config.DEVICE

        self.actor = ActorNetwork(n_states, n_actions, config.HIDDEN_SIZE).to(self.device)
        self.critic = CriticNetwork(n_states, config.HIDDEN_SIZE).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE)

    def select_action(self, observation):
        """
        Selects an action for the environment.
        Returns: action (numpy), log_prob (scalar), value (scalar)
        """
        state = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.actor(state)
            value = self.critic(state)

            action = dist.sample()

            # For continuous, log_prob is summed over action dimensions
            probs = dist.log_prob(action).sum(axis=-1).item()

            action = action.cpu().numpy()[0]
            value = value.item()

        return action, probs, value

    def learn(self, memory, next_value=0.0, next_done=False):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Loop backwards once instead of nested loops
            last_advantage = 0
            for t in reversed(range(len(reward_arr))):
                if t == len(reward_arr) - 1:
                    next_non_terminal = 1.0 - next_done
                    #next_value = 0
                else:
                    next_non_terminal = 1.0 - dones_arr[t]
                    next_value = values[t + 1]

                delta = reward_arr[t] + self.gamma * next_value * next_non_terminal - values[t]
                advantage[t] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage

            advantage = torch.tensor(advantage).to(self.device)
            values = torch.tensor(values).to(self.device)

            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                # Re-evaluate actions
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions).sum(axis=-1)
                dist_entropy = dist.entropy().sum(axis=-1).mean()

                # Ratio
                prob_ratio = (new_probs - old_probs).exp()

                # Surrogate Loss
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio,
                                                 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]

                # Actor Loss
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                actor_loss -= self.entropy_coef * dist_entropy

                # Critic Loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # Update
                total_loss = actor_loss + 0.5 * critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

        memory.clear()

    def save_checkpoint(self, path):
        torch.save(self.actor.state_dict(), path / "actor.pth")
        torch.save(self.critic.state_dict(), path / "critic.pth")

    def load_checkpoint(self, path):
        self.actor.load_state_dict(torch.load(path / "actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(path / "critic.pth", map_location=self.device))