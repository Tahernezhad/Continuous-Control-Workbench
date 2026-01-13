import gymnasium as gym
import torch
import numpy as np
import os
from pathlib import Path

import config
from sac_agent import SACAgent
from utils import check_system_device, set_seed


class SACReplayer:
    def __init__(self, run_path: str, env_name: str = "BipedalWalker-v3", n_episodes: int = 5,
                 save_video: bool = False):
        self.run_path = Path(run_path)
        self.model_path = self.run_path / "models"
        self.video_path = self.run_path / "replay_videos"
        self.env_name = env_name
        self.n_episodes = n_episodes
        self.save_video = save_video
        self.device = check_system_device()
        self.env = None
        self.agent = None

        self._check_paths()

    def _check_paths(self):
        if not (self.model_path / "actor.pth").exists():
            raise FileNotFoundError(f"[Error] Actor checkpoint missing in: {self.model_path}")

    def setup_environment(self):
        render_mode = "rgb_array" if self.save_video else "human"
        self.env = gym.make(self.env_name, render_mode=render_mode)
        set_seed(config.SEED)

        if self.save_video:
            self.video_path.mkdir(exist_ok=True)
            self.env = gym.wrappers.RecordVideo(
                self.env,
                video_folder=str(self.video_path),
                episode_trigger=lambda x: True,
                name_prefix="sac-replay"
            )

    def load_agent(self):
        if not self.env: self.setup_environment()
        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])

        self.agent = SACAgent(n_states, n_actions, max_action, config)
        self.agent.load_checkpoint(self.model_path)
        print("[System] SAC Model loaded successfully.")

    def run_replay_session(self):
        if not self.agent: self.load_agent()
        print(f"\nSTARTING REPLAY ({self.n_episodes} Episodes)...")

        for i in range(1, self.n_episodes + 1):
            state, _ = self.env.reset()
            done = False
            score = 0

            while not done:
                # evaluate=True -> Deterministic action (Mean of Gaussian)
                action = self.agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                score += reward

            print(f"Episode {i} | Score: {score:.2f}")

        self.env.close()


if __name__ == "__main__":
    # UPDATE THIS TO YOUR SAC RESULT FOLDER
    RUN_PATH = "results/2026-01-12_13-41-50_BipedalWalker-v3_SAC"

    if os.path.exists(RUN_PATH):
        replayer = SACReplayer(RUN_PATH, n_episodes=3, save_video=True)
        replayer.run_replay_session()
    else:
        print("Please update RUN_PATH in replay.py")