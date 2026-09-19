"""
Deprecated shim.

This module used to carry a second, independent copy of DynamicSignLSTM. Two
definitions of the same architecture drift apart silently: a change to one means
checkpoints saved by the training script stop loading in the served app.

The single definition now lives in src/backend/models/lstm_model.py. This module
re-exports it so existing `from model import DynamicSignLSTM` imports keep
working; prefer importing from src.backend.models.lstm_model directly.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.models.lstm_model import DynamicSignLSTM  # noqa: E402

__all__ = ["DynamicSignLSTM"]
