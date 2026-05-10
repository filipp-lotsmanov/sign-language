"""
Configuration for Sign Language Recognition Project
====================================================
Shared settings for training, inference, and data processing.
"""

from pathlib import Path

# ============ PATHS ============
DATA_DIR = Path(".")
PHOTOS_DIR = Path("letters")

INPUT_CSV = DATA_DIR / "ngt_data.csv"
OUTPUT_CSV = DATA_DIR / "ngt_final.csv"
MODEL_PATH = DATA_DIR / "best_model.pth"
ENCODER_PATH = DATA_DIR / "label_encoder.pkl"
MEDIAPIPE_MODEL = str(Path(__file__).parent.parent.parent / "models" / "hand_landmarker.task")

# ============ DATA SETTINGS ============
# Mapping: label -> folder name
LETTER_FOLDERS = {
    'D': 'D_letter',
    'F': 'F_letter',
    'G': 'G_letter',
    'R': 'R_letter',
    'S': 'S_letter',
    'V': 'V_letter',
    'X': 'X_letter',
    'Nonsense': 'Nonsense'
}

# Letters to replace in original dataset
LETTERS_TO_REPLACE = list(LETTER_FOLDERS.keys())

# Augmentation multiplier
AUGMENT_MULTIPLIER = 10

# ============ MODEL SETTINGS ============
INPUT_DIM = 63          # 21 landmarks × 3 coordinates
HIDDEN_DIM = 256        # Hidden layer size
NUM_BLOCKS = 4          # Number of residual blocks
DROPOUT = 0.3           # Dropout probability

# ============ TRAINING SETTINGS ============
BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 15           # Early stopping patience
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

TEST_SIZE = 0.15        # Fraction for test set
VAL_SIZE = 0.15         # Fraction for validation set

# ============ INFERENCE SETTINGS ============
CONFIDENCE_THRESHOLD = 0.5
ENTROPY_THRESHOLD = 1.8

# Smoothing settings (prevents rapid letter switching)
SMOOTHING_WINDOW = 25          # Frames to consider for voting
MIN_AGREEMENT = 0.5            # 50% ratio to confirm first letter
SWITCH_THRESHOLD = 0.6         # 60% ratio needed to switch to new letter
MIN_FRAMES_BEFORE_SWITCH = 20  # Minimum frames before allowing switch

MIN_HAND_FRAMES = 10

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30