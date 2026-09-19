"""
Core application configuration and constants.

Server limits are read from the environment so they can be tuned per deployment
without editing code.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_MODEL_DIR = MODELS_DIR / "static"
DYNAMIC_MODEL_DIR = MODELS_DIR / "dynamic"
ASSETS_DIR = PROJECT_ROOT / "src" / "assets"
LETTER_GIFS_DIR = ASSETS_DIR  # GIFs are directly in assets folder

# Model paths
STATIC_MODEL_PATH = STATIC_MODEL_DIR / "best_model.pth"
DYNAMIC_MODEL_PATH = DYNAMIC_MODEL_DIR / "best_model.pth"
STATIC_CLASSES_PATH = STATIC_MODEL_DIR / "classes.npy"
STATIC_LABEL_ENCODER_PATH = STATIC_MODEL_DIR / "label_encoder.pkl"
DYNAMIC_CLASSES_PATH = DYNAMIC_MODEL_DIR / "classes.npy"
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"

# Letter configuration
STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # All except J, Z
DYNAMIC_LETTERS = ["J", "Z"]
ALL_LETTERS = sorted(STATIC_LETTERS + DYNAMIC_LETTERS)

# The trained classifier adds a "Nonsense" class on top of the static letters.
NONSENSE_LABEL = "Nonsense"
NUM_STATIC_CLASSES = len(STATIC_LETTERS) + 1  # 24 letters + Nonsense = 25

# Landmark feature size: 21 MediaPipe landmarks x 3 coordinates.
INPUT_SIZE = 63

# Detection settings
RECORDING_DURATION = 3.0  # seconds to record for each attempt
# The single confidence gate for accepting a prediction. Previously three
# different values lived in three config files; only this one was ever read.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# Session settings
MAX_ATTEMPT_TIME = 60  # seconds before timeout per letter
HINT_THRESHOLD_ATTEMPTS = 3  # Failed attempts between hints
MAX_HINTS = 2  # Maximum number of hints per letter

# Attempt counts at which a hint appears, derived so the constant above is
# actually authoritative. With 3 and MAX_HINTS=2 this is [3, 6].
HINT_THRESHOLDS = [HINT_THRESHOLD_ATTEMPTS * (i + 1) for i in range(MAX_HINTS)]

# Dynamic letter buffer settings
DYNAMIC_BUFFER_SIZE = 30  # frames needed for LSTM
# Shortest usable dynamic sequence; shorter buffers are interpolated up.
DYNAMIC_MIN_BUFFER = 10

# --- Server limits -----------------------------------------------------------
# Sessions live in process memory, so both of these bound how much a client can
# make the server allocate.
SESSION_MAX_IDLE = int(os.getenv("SESSION_MAX_IDLE", "1800"))  # seconds
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))
SESSION_SWEEP_INTERVAL = int(os.getenv("SESSION_SWEEP_INTERVAL", "300"))  # seconds

# Largest accepted base64 frame payload. A 640x480 JPEG at quality 0.8 is well
# under 100 KB; this leaves headroom while rejecting oversized uploads.
MAX_FRAME_BYTES = int(os.getenv("MAX_FRAME_BYTES", str(2 * 1024 * 1024)))

# Per-connection inbound message ceiling, averaged over a sliding window. The
# browser sends 10 frames/second, so this allows a wide margin before throttling.
MAX_MESSAGES_PER_SECOND = float(os.getenv("MAX_MESSAGES_PER_SECOND", "30"))

# Comma-separated list of allowed browser origins. The previous wildcard combined
# with allow_credentials=True is rejected by browsers and unsafe in production.
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]
