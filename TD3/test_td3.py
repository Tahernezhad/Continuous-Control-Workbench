import pytest
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

from networks import Actor, Critic
from replay_buffer import ReplayBuffer
from td3_agent import TD3Agent



# Test Configuration
class TestConfig:
    DEVICE = torch.device("cpu")
    GAMMA = 0.99
    TAU = 0.005
    LEARNING_RATE_ACTOR = 1e-3
    LEARNING_RATE_CRITIC = 1e-3
    BATCH_SIZE = 4
    REPLAY_BUFFER_SIZE = 100
    WARMUP_STEPS = 5
    POLICY_NOISE = 0.2
    NOISE_CLIP = 0.5
    EXPLORATION_NOISE = 0.1
    POLICY_DELAY = 2
    MAX_GAMES = 2
    MAX_STEPS = 10
    SAVE_MODEL = False


@pytest.fixture
def td3_env_setup():
    """Sets up a continuous environment (Pendulum) for testing."""
    env = gym.make("Pendulum-v1")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    return env, n_states, n_actions, max_action


# Test Networks
def test_actor_critic_shapes(td3_env_setup):
    """Verifies Actor and Twin Critic network output shapes and ranges."""
    _, n_states, n_actions, _ = td3_env_setup
    actor = Actor(n_states, n_actions)
    critic = Critic(n_states, n_actions)

    test_state = torch.randn(1, n_states)
    test_action = torch.randn(1, n_actions)

    # Actor output should be [-1, 1] due to Tanh
    action_out = actor(test_state)
    assert action_out.shape == (1, n_actions)
    assert torch.all(action_out >= -1.0) and torch.all(action_out <= 1.0)

    # Critic should return two Q-values (Twin Critic trick)
    q1, q2 = critic(test_state, test_action)
    assert q1.shape == (1, 1)
    assert q2.shape == (1, 1)
    assert not torch.equal(q1, q2), "Twin critics should initialize with different weights."


# Test Replay Buffer
def test_replay_buffer(td3_env_setup):
    """Tests the numpy-based ReplayBuffer pushing and sampling."""
    _, n_states, n_actions, _ = td3_env_setup
    buffer = ReplayBuffer(n_states, n_actions, max_size=10)

    s = np.random.random(n_states)
    a = np.random.random(n_actions)
    ns = np.random.random(n_states)
    r = 1.0
    d = False

    buffer.push(s, a, ns, r, d)
    assert len(buffer) == 1
    assert buffer.ptr == 1

    states, actions, next_states, rewards, not_dones = buffer.sample(1)
    assert states.shape == (1, n_states)
    assert actions.shape == (1, n_actions)
    assert np.isclose(not_dones[0], 1.0)  # not done = 1.0


# Test Agent
def test_td3_agent_action_selection(td3_env_setup):
    """Checks if the agent handles noise correctly during selection."""
    _, n_states, n_actions, max_action = td3_env_setup
    agent = TD3Agent(n_states, n_actions, max_action, TestConfig)

    state = np.random.random(n_states)

    # Deterministic selection
    action_pure = agent.select_action(state, noise=0.0)
    assert action_pure.shape == (n_actions,)

    # Exploration noise selection
    action_noisy = agent.select_action(state, noise=0.5)
    assert action_noisy.shape == (n_actions,)
    # Ensure clipping respects max_action bounds
    assert np.all(action_noisy <= max_action) and np.all(action_noisy >= -max_action)


def test_delayed_policy_update(td3_env_setup):
    """Verifies the delayed actor update logic."""
    _, n_states, n_actions, max_action = td3_env_setup
    agent = TD3Agent(n_states, n_actions, max_action, TestConfig)
    memory = ReplayBuffer(n_states, n_actions, TestConfig.REPLAY_BUFFER_SIZE)

    # Fill memory
    for _ in range(TestConfig.BATCH_SIZE):
        memory.push(np.zeros(n_states), np.zeros(n_actions), np.zeros(n_states), 1.0, False)

    agent.optimize(memory)
    assert agent.total_it == 1

    agent.optimize(memory)
    assert agent.total_it == 2


# Test Pipeline
def test_td3_training_loop(tmpdir, td3_env_setup):
    """Simulates a mini training session to ensure no tensor dimension crashes."""
    env, n_states, n_actions, max_action = td3_env_setup
    agent = TD3Agent(n_states, n_actions, max_action, TestConfig)
    memory = ReplayBuffer(n_states, n_actions, TestConfig.REPLAY_BUFFER_SIZE)

    s, _ = env.reset()
    for _ in range(TestConfig.BATCH_SIZE + 1):
        a = env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a)
        memory.push(s, a, ns, r, term or trunc)
        s = ns

    agent.optimize(memory)

    model_path = Path(tmpdir.mkdir("models"))
    agent.save_checkpoint(model_path)
    assert (model_path / "actor.pth").exists()
    assert (model_path / "critic.pth").exists()


if __name__ == "__main__":
    pytest.main([__file__])