import config
import numpy as np
import gymnasium as gym
from ppo_agent import PPOAgent
from rollout_buffer import RolloutBuffer
from utils import check_system_device, get_run_directories, set_seed, plot_learning_curve, save_config


def print_environment_summary(env_name: str, n_states: int, n_actions: int):
    print("\n" + "=" * 40)
    print(f"ENVIRONMENT SUMMARY: {env_name}")
    print("=" * 40)
    print(f"Observation Space: {n_states}")
    print(f"Action Space:      {n_actions}")
    print("=" * 40 + "\n")


def run_training_session():
    #  1. System Setup
    device = check_system_device()
    config.DEVICE = device
    set_seed(config.SEED)

    # Setup Directories
    run_dir, model_dir, plot_dir = get_run_directories(
        base_dir="results",
        env_name=config.ENV_NAME,
        algo_name="PPO"
    )
    save_config(config, run_dir / "hyperparameters.txt")
    print(f"[System] Artifacts will be saved to: {run_dir}")

    #  2. Environment Initialization
    env = gym.make(config.ENV_NAME, render_mode="rgb_array")

    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Normalize Inputs (Observation) AND Outputs (Rewards)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: np.clip(obs, -10, 10),
        env.observation_space)
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))

    if config.RECORD_VIDEO:
        video_dir = run_dir / "videos"
        print(f"[System] Recording videos to: {video_dir}")
        env = gym.wrappers.RecordVideo(
            env,
            str(video_dir),
            episode_trigger=lambda x: x % config.VIDEO_RECORD_INTERVAL == 0
        )

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    print_environment_summary(config.ENV_NAME, n_states, n_actions)

    #  3. Agent & Buffer
    print(f"Initializing PPO Agent on {device}...")
    agent = PPOAgent(n_states, n_actions, config)
    buffer = RolloutBuffer(config.BATCH_SIZE)

    score_history = []
    best_score = float('-inf')

    print("\nStarting training workflow...")
    print("-" * 50)

    #  4. Main Training Loop
    for episode in range(1, config.MAX_GAMES + 1):
        state, _ = env.reset()
        done = False

        while not done:
            # Select action
            action, log_prob, val = agent.select_action(state)

            # Clip action for environment
            action_env = np.clip(action, -1.0, 1.0)

            # Step
            next_state, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated

            # Store NORMALIZED reward for training (stable gradients)
            buffer.store(state, action, log_prob, val, reward, done)

            state = next_state

            # PPO Update
            if len(buffer.states) >= config.ROLLOUT_LENGTH:
                if done:
                    next_val = 0.0
                else:
                    _, _, next_val = agent.select_action(next_state)
                agent.learn(buffer, next_value=next_val, next_done=done)

        #  Logging
        # Extract REAL score from RecordEpisodeStatistics
        if "episode" in info:
            real_score = float(info["episode"]["r"])
            score_history.append(real_score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                if config.SAVE_MODEL:
                    agent.save_checkpoint(model_dir)

            if episode % config.REPORT_INTERVAL == 0:
                print(f"Episode {episode}/{config.MAX_GAMES} | "
                      f"Real Score: {real_score:6.2f} | "
                      f"Avg Score: {avg_score:6.2f}")

    #  5. Post-Training
    print("-" * 50)
    print(f"\nTraining complete.")
    print(f"Generating learning curve at {plot_dir}...")
    plot_learning_curve(score_history, plot_dir / "learning_curve.png")

    env.close()
    print("Done.")


if __name__ == '__main__':
    run_training_session()