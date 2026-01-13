import numpy as np
from typing import Tuple


class ReplayBuffer:
    def __init__(self, n_states: int, n_actions: int, max_size: int = 1_000_000):
        self.state = np.zeros((max_size, n_states))
        self.action = np.zeros((max_size, n_actions))
        self.next_state = np.zeros((max_size, n_states))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def push(self, state, action, next_state, reward, done):
        """Stores a transition in the buffer."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples a batch of experiences."""
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )

    def __len__(self) -> int:
        return self.size