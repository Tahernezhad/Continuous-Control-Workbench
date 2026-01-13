import torch


#  Environment Settings
ENV_NAME = "BipedalWalker-v3"
SEED = 42

#  Training Hyperparameters
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.005                  # Soft update coefficient (Polyak averaging)
BATCH_SIZE = 100
REPLAY_BUFFER_SIZE = 1_000_000
WARMUP_STEPS = 1000          # Steps of pure random exploration

#  TD3 Specifics
POLICY_NOISE = 0.2           # Noise added to target policy during critic update
NOISE_CLIP = 0.5             # Range to clip target policy noise
EXPLORATION_NOISE = 0.1      # Gaussian noise added to action during selection
POLICY_DELAY = 2             # Frequency of delayed policy updates

#  Training Loop Settings
MAX_GAMES = 1500
MAX_STEPS = 1600             # Max steps per episode
REPORT_INTERVAL = 10         # Log progress every N episodes
MOVING_AVG_WINDOW = 100

#  Device Configuration
# (Will be updated dynamically in main.py checks)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  I/O Settings
SAVE_MODEL = True
RECORD_VIDEO = False
VIDEO_RECORD_INTERVAL = 50