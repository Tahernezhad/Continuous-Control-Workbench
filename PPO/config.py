import torch

# Environment Settings
ENV_NAME = "BipedalWalker-v3"
SEED = 42

#  PPO Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.98            # Generalized Advantage Estimation
CLIP_EPSILON = 0.2           # PPO Clip parameter
K_EPOCHS = 5                 # Number of update epochs per rollout
BATCH_SIZE = 128             # Mini-batch size
ROLLOUT_LENGTH = 2048        # Steps to collect before updating
ENTROPY_COEF = 0.005         # Slight entropy bonus for exploration
MAX_GRAD_NORM = 0.5          # Gradient clipping

#  Model Settings
HIDDEN_SIZE = 256

#  Training Loop
MAX_GAMES = 1500
MAX_STEPS = 1600
REPORT_INTERVAL = 10
MOVING_AVG_WINDOW = 100

#  Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  I/O
SAVE_MODEL = True
RECORD_VIDEO = False
VIDEO_RECORD_INTERVAL = 50