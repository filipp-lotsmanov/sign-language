"""
Inference-time configuration for the static sign classifier.

Everything shared with the rest of the application is re-exported from
src.backend.core.config rather than redefined. This file previously carried its
own copies of the letter count, the confidence threshold and a set of training
hyperparameters, none of which were read by anything: the served app used the
core values and the training scripts used training/static/config.py, so the
duplicates silently drifted.
"""

import torch

from src.backend.core.config import (
    INPUT_SIZE,
    NUM_STATIC_CLASSES,
    STATIC_CLASSES_PATH,
    STATIC_LABEL_ENCODER_PATH,
    STATIC_MODEL_DIR,
    STATIC_MODEL_PATH,
)

# Where the static model artifacts live.
BASE_DIR = STATIC_MODEL_DIR
MODEL_SAVE_PATH = STATIC_MODEL_PATH
CLASSES_PATH = STATIC_CLASSES_PATH
LABEL_ENCODER_PATH = STATIC_LABEL_ENCODER_PATH

# Model shape. These must match the trained checkpoint; the checkpoint's own
# metadata takes precedence when present.
NUM_CLASSES = NUM_STATIC_CLASSES

# Architecture defaults, matching training/static/config.py.
HIDDEN_DIM = 256
NUM_BLOCKS = 4
DROPOUT = 0.3

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    "BASE_DIR",
    "CLASSES_PATH",
    "DEVICE",
    "DROPOUT",
    "HIDDEN_DIM",
    "INPUT_SIZE",
    "LABEL_ENCODER_PATH",
    "MODEL_SAVE_PATH",
    "NUM_BLOCKS",
    "NUM_CLASSES",
]
