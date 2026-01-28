import pytest
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

from networks import GaussianActor, TwinCritic
from replay_buffer import ReplayBuffer
from sac_agent import SACAgent


# Test Configuration
class TestConfig:
    DEVICE = torch.device("cpu")
    GAMMA = 0.99
    TAU = 0.005
    LEARNING_RATE_ACTOR = 1e-3
    LEARNING_RATE_CRITIC = 1e-3
    LEARNING_RATE_ALPHA = 1e-3
    HIDDEN_SIZE = 64
    BATCH_SIZE = 4
    REPLAY_BUFFER_SIZE = 100
    WARMUP_STEPS = 5
    TARGET_ENTROPY_SCALE = 1.0
    MAX_GAMES = 2
    MAX_STEPS = 10
    SAVE_MODEL = False


@pytest.fixture
def sac_env_setup():
    """Sets up a continuous environment (Pendulum) for testing."""
    env = gym.make("Pendulum-v1")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    return env, n_states, n_actions, max_action


# Test Networks
def test_sac_network_shapes(sac_env_setup):
    """Verifies Gaussian Actor sampling and Twin Critic architecture."""
    _, n_states, n_actions, max_action = sac_env_setup
    actor = GaussianActor(n_states, n_actions, hidden_size=64, max_action=max_action)
    critic = TwinCritic(n_states, n_actions, hidden_size=64)

    state_batch = torch.randn(5, n_states)

    action, log_prob, mean = actor.sample(state_batch)
    assert action.shape == (5, n_actions)
    assert log_prob.shape == (5, 1)
    assert torch.all(action <= max_action) and torch.all(action >= -max_action)

    q1, q2 = critic(state_batch, action)
    assert q1.shape == (5, 1)
    assert q2.shape == (5, 1)


# Test Replay Buffer
def test_sac_replay_buffer(sac_env_setup):
    """Tests the SAC ReplayBuffer storage and sampling."""
    _, n_states, n_actions, _ = sac_env_setup
    buffer = ReplayBuffer(n_states, n_actions, max_size=10)

    s = np.random.random(n_states)
    a = np.random.random(n_actions)
    ns = np.random.random(n_states)
    r = 1.0
    d = False

    buffer.push(s, a, ns, r, d)
    assert len(buffer) == 1

    states, actions, next_states, rewards, not_dones = buffer.sample(1)
    assert states.shape == (1, n_states)
    assert actions.shape == (1, n_actions)
    assert np.isclose(not_dones[0], 1.0)  # 1 - done


# Tests Agent
def test_sac_entropy_tuning_init(sac_env_setup):
    """Verifies target entropy and learnable alpha initialization."""
    _, n_states, n_actions, max_action = sac_env_setup
    agent = SACAgent(n_states, n_actions, max_action, TestConfig)

    assert agent.target_entropy == -float(n_actions)
    assert torch.isclose(agent.alpha, torch.tensor(1.0))


def test_sac_action_selection(sac_env_setup):
    """Checks stochastic vs deterministic action selection."""
    _, n_states, n_actions, max_action = sac_env_setup
    agent = SACAgent(n_states, n_actions, max_action, TestConfig)
    state = np.random.random(n_states)

    action_stoch = agent.select_action(state, evaluate=False)
    assert action_stoch.shape == (n_actions,)

    action_eval = agent.select_action(state, evaluate=True)
    assert action_eval.shape == (n_actions,)


# Test Pipeline
def test_sac_training_loop(tmpdir, sac_env_setup):
    """Simulates a full SAC update loop including alpha tuning."""
    env, n_states, n_actions, max_action = sac_env_setup
    agent = SACAgent(n_states, n_actions, max_action, TestConfig)
    memory = ReplayBuffer(n_states, n_actions, TestConfig.REPLAY_BUFFER_SIZE)

    s, _ = env.reset()
    for _ in range(TestConfig.BATCH_SIZE + 1):
        a = env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a)
        memory.push(s, a, ns, r, term or trunc)
        s = ns

    # Run optimization
    initial_log_alpha = agent.log_alpha.clone()
    agent.optimize(memory)

    assert not torch.equal(initial_log_alpha, agent.log_alpha)

    # Test Checkpoint Saving
    model_path = Path(tmpdir.mkdir("sac_models"))
    agent.save_checkpoint(model_path)
    assert (model_path / "actor.pth").exists()
    assert (model_path / "critic.pth").exists()
    assert (model_path / "log_alpha.pth").exists()


if __name__ == "__main__":
    pytest.main([__file__])