import pytest
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

from networks import ActorNetwork, CriticNetwork
from rollout_buffer import RolloutBuffer
from ppo_agent import PPOAgent



# Test Configuration
class TestConfig:
    DEVICE = torch.device("cpu")
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    K_EPOCHS = 2
    BATCH_SIZE = 4
    ROLLOUT_LENGTH = 16
    LEARNING_RATE = 1e-3
    ENTROPY_COEF = 0.01
    HIDDEN_SIZE = 64
    MAX_GAMES = 2
    REPORT_INTERVAL = 1
    SAVE_MODEL = False


@pytest.fixture
def ppo_env_setup():
    """Sets up a continuous environment (BipedalWalker) for PPO validation."""
    env = gym.make("BipedalWalker-v3")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    return env, n_states, n_actions


# Test Networks
def test_ppo_network_outputs(ppo_env_setup):
    """Verifies that the Actor returns a Normal distribution and Critic returns a scalar."""
    _, n_states, n_actions = ppo_env_setup
    actor = ActorNetwork(n_states, n_actions, hidden_size=64)
    critic = CriticNetwork(n_states, hidden_size=64)

    state_tensor = torch.randn(1, n_states)

    dist = actor(state_tensor)
    assert isinstance(dist, torch.distributions.Normal)
    action = dist.sample()
    assert action.shape == (1, n_actions)

    value = critic(state_tensor)
    assert value.shape == (1, 1)


# Test Rollout Buffer
def test_rollout_buffer_logic():
    """Tests the list-based rollout buffer storage and batch generation."""
    batch_size = 4
    buffer = RolloutBuffer(batch_size)

    # Fill buffer with dummy transitions
    for _ in range(8):
        buffer.store(
            state=np.zeros(24),
            action=np.zeros(4),
            probs=-0.5,  # Log probability
            vals=0.1,
            reward=1.0,
            done=False
        )

    assert len(buffer.states) == 8

    # Test batch generation
    states, actions, probs, vals, rewards, dones, batches = buffer.generate_batches()

    assert states.shape == (8, 24)
    assert len(batches) == 2  # 8 total samples / 4 batch size
    assert len(batches[0]) == batch_size


# Test Agent
def test_ppo_agent_selection(ppo_env_setup):
    """Checks the stochastic action selection process."""
    _, n_states, n_actions = ppo_env_setup
    agent = PPOAgent(n_states, n_actions, TestConfig)

    obs = np.random.random(n_states)
    action, prob, val = agent.select_action(obs)

    assert isinstance(action, np.ndarray)
    assert action.shape == (n_actions,)
    assert isinstance(prob, float)
    assert isinstance(val, float)


# Test Pipeline
def test_ppo_learn_cycle(ppo_env_setup):
    """Ensures the learn() function executes Advantage estimation and clears the buffer."""
    _, n_states, n_actions = ppo_env_setup
    agent = PPOAgent(n_states, n_actions, TestConfig)
    buffer = RolloutBuffer(TestConfig.BATCH_SIZE)

    for _ in range(TestConfig.ROLLOUT_LENGTH):
        buffer.store(
            state=np.random.random(n_states),
            action=np.random.random(n_actions),
            probs=-0.5,
            vals=0.0,
            reward=1.0,
            done=False
        )

    agent.learn(buffer, next_value=0.1, next_done=False)

    assert len(buffer.states) == 0


def test_ppo_checkpointing(tmpdir, ppo_env_setup):
    """Tests saving and loading of PPO model weights."""
    _, n_states, n_actions = ppo_env_setup
    agent = PPOAgent(n_states, n_actions, TestConfig)

    save_path = Path(tmpdir.mkdir("ppo_models"))
    agent.save_checkpoint(save_path)

    assert (save_path / "actor.pth").exists()
    assert (save_path / "critic.pth").exists()

    agent.load_checkpoint(save_path)


if __name__ == "__main__":
    pytest.main([__file__])