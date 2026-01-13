import torch


#  Environment Settings
ENV_NAME = "BipedalWalker-v3"
SEED = 42

#  Training Hyperparameters
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 3e-4
LEARNING_RATE_ALPHA = 3e-4
GAMMA = 0.99
TAU = 0.005                  # Soft update coefficient
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 1_000_000
WARMUP_STEPS = 1000          # Steps of pure random exploration

#  SAC Specifics
#ALPHA = 0.2                 # Entropy regularization coefficient (Temperature)
                             # Higher = more exploration, Lower = more exploitation
HIDDEN_SIZE = 256            # Width of the networks
TARGET_ENTROPY_SCALE = 1.0   # Scale factor for target entropy

#  Training Loop Settings
MAX_GAMES = 1000
MAX_STEPS = 1600             # Max steps per episode
REPORT_INTERVAL = 10         # Log progress every N episodes
MOVING_AVG_WINDOW = 100

#  Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  I/O Settings
SAVE_MODEL = True
RECORD_VIDEO = False
VIDEO_RECORD_INTERVAL = 50