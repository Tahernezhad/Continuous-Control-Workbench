import config
import gymnasium as gym
from pathlib import Path
from td3_agent import TD3Agent
from utils import check_system_device, set_seed


class TD3Replayer:
    """
    Handles the loading, visualization, and video recording of a trained TD3 agent.
    """

    def __init__(self,
                 run_path: str,
                 env_name: str = "BipedalWalker-v3",
                 n_episodes: int = 5,
                 save_video: bool = False):
        """
        Args:
            run_path: Path to the specific timestamped result folder.
            env_name: Name of the gym environment.
            n_episodes: Number of test episodes to run.
            save_video: If True, saves MP4s instead of rendering to window.
        """
        self.run_path = Path(run_path)
        self.model_path = self.run_path / "models"
        self.video_path = self.run_path / "replay_videos"

        self.env_name = env_name
        self.n_episodes = n_episodes
        self.save_video = save_video

        # Internal State
        self.device = check_system_device()
        self.env = None
        self.agent = None

        # Verify paths
        self._check_paths()

    def _check_paths(self):
        """Ensures the model directory and files exist before starting."""
        if not self.model_path.exists():
            raise FileNotFoundError(f" Model directory not found: {self.model_path}")

        # Based on your screenshot, checking for actor.pth inside 'models' folder
        if not (self.model_path / "actor.pth").exists():
            raise FileNotFoundError(f" Actor checkpoint missing in: {self.model_path}")

        print(f" Run directory verified: {self.run_path}")

    def load_config_override(self):
        """Parses hyperparameters.txt to ensure model architecture matches."""
        param_file = self.run_path / "hyperparameters.txt"
        if not param_file.exists():
            print(" No hyperparameters.txt found. Using default 'config.py'.")
            return

        print(f" Loading config overrides from: {param_file.name}")
        with open(param_file, 'r') as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    if hasattr(config, key):
                        # Simple type inference
                        if value.isdigit():
                            setattr(config, key, int(value))
                        elif value.replace('.', '', 1).isdigit():
                            setattr(config, key, float(value))

    def setup_environment(self):
        """Initializes the environment with the correct render mode."""
        # Select mode: 'rgb_array' is required for video recording
        render_mode = "rgb_array" if self.save_video else "human"

        print(f"Initializing Environment: {self.env_name} (Render Mode: {render_mode})...")
        self.env = gym.make(self.env_name, render_mode=render_mode)
        set_seed(config.SEED)

        # Attach Video Recorder if requested
        if self.save_video:
            self.video_path.mkdir(exist_ok=True)
            print(f" Saving videos to: {self.video_path}")

            self.env = gym.wrappers.RecordVideo(
                self.env,
                video_folder=str(self.video_path),
                episode_trigger=lambda x: True,  # Record every episode
                name_prefix="td3-replay"
            )

    def load_agent(self):
        """Initializes the agent and loads the trained weights."""
        if not self.env:
            self.setup_environment()

        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])

        print(f"\nInitializing Agent (States: {n_states}, Actions: {n_actions})...")
        self.agent = TD3Agent(n_states, n_actions, max_action, config)

        print(f" Loading checkpoints from {self.model_path}...")
        self.agent.load_checkpoint(self.model_path)
        print(" Model weights loaded successfully.")

    def run_replay_session(self):
        """Main loop to visualize the agent's performance."""
        if not self.agent:
            self.load_agent()

        print("\n" + "=" * 40)
        print(f"STARTING REPLAY SESSION ({self.n_episodes} Episodes)")
        print("=" * 40)

        for i in range(1, self.n_episodes + 1):
            state, _ = self.env.reset()
            done = False
            score = 0
            steps = 0

            while not done:
                # noise=0.0 ensures we see the pure learned policy
                action = self.agent.select_action(state, noise=0.0)

                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                score += reward
                steps += 1

            print(f"Episode {i} | Score: {score:.2f} | Steps: {steps}")

        self.env.close()
        print("\n[System] Replay finished.")
        if self.save_video:
            print(f"Videos saved at: {self.video_path}")



if __name__ == "__main__":

    # PASTE THE FOLDER NAME FROM YOUR SCREENSHOT HERE
    RESULTS_FOLDER = "2026-01-11_02-47-20_BipedalWalker-v3_TD3"

    # Construct absolute path relative to this script
    BASE_DIR = Path(__file__).parent / "results"
    RUN_PATH = BASE_DIR / RESULTS_FOLDER

    if not RUN_PATH.exists():
        print(f"Directory not found: {RUN_PATH}")
        print("Please check the folder name in 'replay.py'.")
    else:
        replayer = TD3Replayer(
            run_path=str(RUN_PATH),
            env_name=config.ENV_NAME,
            n_episodes=3,  # How many videos to record
            save_video=True  # Set to False for pop-up window
        )

        replayer.load_config_override()
        replayer.run_replay_session()