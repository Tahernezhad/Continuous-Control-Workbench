import config
import numpy as np
import gymnasium as gym
from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from utils import check_system_device, get_run_directories, set_seed, plot_learning_curve, save_config


def print_environment_summary(env_name: str, n_states: int, n_actions: int, max_action: float):
    print("\n" + "=" * 40)
    print(f" ENVIRONMENT SUMMARY: {env_name}")
    print("=" * 40)
    print(f" Observation Space: {n_states}")
    print(f" Action Space:      {n_actions}")
    print(f" Action Bounds:     [-{max_action}, {max_action}]")
    print("=" * 40 + "\n")


def run_training_session():
    #  Setup
    device = check_system_device()
    config.DEVICE = device
    set_seed(config.SEED)

    run_dir, model_dir, plot_dir = get_run_directories(
        base_dir="results",
        env_name=config.ENV_NAME,
        algo_name="SAC"
    )
    save_config(config, run_dir / "hyperparameters.txt")
    print(f"Artifacts will be saved to: {run_dir}")

    #  Environment
    env = gym.make(config.ENV_NAME, render_mode="rgb_array")

    if config.RECORD_VIDEO:
        video_dir = run_dir / "videos"
        env = gym.wrappers.RecordVideo(
            env,
            str(video_dir),
            episode_trigger=lambda x: x % config.VIDEO_RECORD_INTERVAL == 0
        )

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print_environment_summary(config.ENV_NAME, n_states, n_actions, max_action)

    #  Initialization
    print(f" Initializing SAC Agent on {device}...")
    agent = SACAgent(n_states, n_actions, max_action, config)
    memory = ReplayBuffer(n_states, n_actions, config.REPLAY_BUFFER_SIZE)

    score_history = []
    best_score = float('-inf')  # Corrected from env.reward_range

    print("\n Starting training workflow...")
    print("-" * 50)

    #  Training Loop
    for episode in range(1, config.MAX_GAMES + 1):
        state, _ = env.reset()
        done = False
        score = 0
        steps = 0

        while not done and steps < config.MAX_STEPS:
            if len(memory) < config.WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                # evaluate=False enables stochastic exploration
                action = agent.select_action(state, evaluate=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, next_state, reward, done)
            agent.optimize(memory)

            state = next_state
            score += reward
            steps += 1

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

    #  Post-Training
    print("-" * 50)
    print(f"Training complete. Generating learning curve...")
    plot_learning_curve(score_history, plot_dir / "learning_curve.png")
    env.close()


if __name__ == '__main__':
    run_training_session()