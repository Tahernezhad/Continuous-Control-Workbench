import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List
import matplotlib.pyplot as plt



def check_system_device() -> torch.device:
    """
    Checks and prints the compute device (GPU/CPU) details.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 40)
    print(f" Using device: {str(device).upper()}")

    if device.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(" WARNING: Using CPU. Training will be slow.")
    print("=" * 40 + "\n")

    return device


def get_run_directories(base_dir: str = "results",
                        env_name: str = "env",
                        algo_name: str = "TD3") -> Tuple[Path, Path, Path]:
    """
    Generates timestamped directories for models, logs, and plots.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{env_name}_{algo_name}"
    run_dir = Path(base_dir) / folder_name

    model_dir = run_dir / "models"
    plot_dir = run_dir / "plots"

    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    return run_dir, model_dir, plot_dir


def set_seed(seed: int = 42):
    """Sets the seed for reproducibility across numpy, torch, and python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config_module, save_path: Path):
    """Saves configuration variables to a text file."""
    with open(save_path, 'w') as f:
        for key, value in config_module.__dict__.items():
            if not key.startswith('__') and not isinstance(value, type(torch)):
                f.write(f"{key}: {value}\n")


def plot_learning_curve(scores: List[float], save_path: Path, window: int = 100):
    """Plots the learning curve with a moving average overlay."""
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Episode Score', alpha=0.3, color='tab:blue')

    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window) / window, mode='valid')
        # Adjust x-axis for moving average to align correctly
        x_axis = np.arange(window - 1, len(scores)) + 1
        plt.plot(x_axis, moving_avg, label=f'{window}-Game Moving Avg', color='tab:red', linewidth=2)

    plt.title('Training Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()