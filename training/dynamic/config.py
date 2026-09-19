"""
LSTM Model Configuration for J/Z Dynamic Letters
"""

import torch
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data_collect" / "ngt_dynamic" / "jz_dynamic_normalized.npz"

# Artifacts (plots, confusion matrix) stay next to the training code.
MODEL_DIR = Path(__file__).parent
# The checkpoint goes where the application loads it from, so there is no
# manual copy step between training and serving.
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "dynamic"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"
CLASSES_PATH = CHECKPOINT_DIR / "classes.npy"

# Model Architecture
INPUT_SIZE = 63  # landmarks per frame (21 points × 3 coords)
SEQUENCE_LENGTH = 30  # frames per sequence
HIDDEN_SIZE = 128  # LSTM hidden units
NUM_LAYERS = 2  # LSTM layers
NUM_CLASSES = 2  # J and Z
DROPOUT = 0.3

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15  # early stopping patience
TEST_SPLIT = 0.2  # held out, evaluated exactly once at the end
VAL_SPLIT = 0.2  # used for early stopping and checkpoint selection
RANDOM_SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mapping
CLASSES = ["J", "Z"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
