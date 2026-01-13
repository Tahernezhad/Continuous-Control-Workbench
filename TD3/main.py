import config
import numpy as np
import gymnasium as gym
from td3_agent import TD3Agent
from replay_buffer import ReplayBuffer
from utils import check_system_device, get_run_directories, set_seed, plot_learning_curve, save_config


def print_environment_summary(env_name: str, n_states: int, n_actions: int, max_action: float):
    """
    Prints a summary similar to 'DATASET SUMMARY' in your LLM code.
    """
    print("\n" + "=" * 40)
    print(f"ENVIRONMENT SUMMARY: {env_name}")
    print("=" * 40)
    print(f"Observation Space: {n_states}")
    print(f"Action Space:      {n_actions}")
    print(f"Action Bounds:     [-{max_action}, {max_action}]")
    print("=" * 40 + "\n")


def run_training_session():
    #  1. System Setup & Configuration
    device = check_system_device()
    config.DEVICE = device  # Update config with detected device
    set_seed(config.SEED)

    # Setup Directories (LLM Style)
    run_dir, model_dir, plot_dir = get_run_directories(
        base_dir="results",
        env_name=config.ENV_NAME,
        algo_name="TD3"
    )
    save_config(config, run_dir / "hyperparameters.txt")
    print(f"[System] Artifacts will be saved to: {run_dir}")

    #  2. Environment Initialization
    env = gym.make(config.ENV_NAME, render_mode="rgb_array")

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
    max_action = float(env.action_space.high[0])

    print_environment_summary(config.ENV_NAME, n_states, n_actions, max_action)

    #  3. Agent & Memory Initialization
    print(f"Initializing TD3 Agent on {device}...")
    agent = TD3Agent(n_states, n_actions, max_action, config)
    memory = ReplayBuffer(n_states, n_actions, config.REPLAY_BUFFER_SIZE)

    score_history = []
    best_score = float('-inf')

    print("\nStarting training workflow...")
    print("-" * 50)

    #  4. Main Training Loop
    for episode in range(1, config.MAX_GAMES + 1):
        state, _ = env.reset()
        done = False
        score = 0
        steps = 0

        while not done and steps < config.MAX_STEPS:
            # Warmup Phase: Random actions for initial exploration
            if len(memory) < config.WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise=config.EXPLORATION_NOISE)

            # Environment Step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store & Optimize
            memory.push(state, action, next_state, reward, done)
            agent.optimize(memory)

            state = next_state
            score += reward
            steps += 1

        #  Logging
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if config.SAVE_MODEL:
                agent.save_checkpoint(model_dir)

        if episode % config.REPORT_INTERVAL == 0:
            print(f"Episode {episode}/{config.MAX_GAMES} | "
                  f"Score: {score:6.2f} | "
                  f"Avg Score: {avg_score:6.2f} | "
                  f"Steps: {steps}")

    #  5. Post-Training Artifacts
    print("-" * 50)
    print(f"\nTraining complete.")
    print(f"Generating learning curve at {plot_dir}...")
    plot_learning_curve(score_history, plot_dir / "learning_curve.png")

    env.close()
    print("Done.")


if __name__ == '__main__':
    run_training_session()