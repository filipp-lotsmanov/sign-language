"""
Shared dataset locations for the three-stage builder pipeline.

The stages previously each hardcoded their own cwd-relative directory, so the
chain did not actually connect: frankenstein_builder wrote to ./ngt_frankenstein,
augment_landmarks read from ./datasets/ngt_frankenstein, and merge_custom_letters
read from ./data_collect. Anchoring every stage here, relative to the repository
root rather than the current working directory, makes the pipeline runnable from
anywhere.

Pipeline order:
    1. frankenstein_builder.py   ->  RAW_NPZ
    2. augment_landmarks.py      ->  AUGMENTED_NPZ
    3. merge_custom_letters.py   ->  FINAL_NPZ
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Override with DATASETS_DIR to keep large datasets outside the repository.
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", PROJECT_ROOT / "datasets"))

FRANKENSTEIN_DIR = DATASETS_DIR / "ngt_frankenstein"

RAW_NPZ = FRANKENSTEIN_DIR / "ngt_frankenstein.npz"
AUGMENTED_NPZ = FRANKENSTEIN_DIR / "ngt_frankenstein_x10.npz"
FINAL_NPZ = FRANKENSTEIN_DIR / "ngt_frankenstein_final.npz"

# Custom recordings produced by data_collect/record_landmarks.py
CUSTOM_LANDMARKS_DIR = PROJECT_ROOT / "data_collect" / "ngt_custom"

# MediaPipe hand landmarker bundle, shared with the serving path.
HAND_LANDMARKER = PROJECT_ROOT / "models" / "hand_landmarker.task"
