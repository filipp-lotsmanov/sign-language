"""
Checkpoint loading with a safe default.

torch.load with weights_only=False unpickles arbitrary Python objects, which
executes code from the checkpoint file. Model weights are downloaded from a
GitHub release, so a compromised or swapped release would run as the server
user. These helpers try the safe path first and only fall back with a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def load_checkpoint(path: str | Path, device: torch.device, allow_unsafe: bool = True) -> Any:
    """
    Load a checkpoint, preferring weights_only=True.

    Args:
        path: checkpoint file.
        device: map_location target.
        allow_unsafe: whether to retry with weights_only=False when the safe
            load fails. Set False to make an unsafe checkpoint a hard error.

    Returns:
        The deserialized checkpoint.

    Raises:
        Whatever torch.load raises when the safe load fails and allow_unsafe
        is False.
    """
    path = Path(path)

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as safe_error:
        if not allow_unsafe:
            raise

        logger.warning(
            "Safe load of %s failed (%s). Retrying with weights_only=False, which "
            "executes code embedded in the checkpoint. Only do this for files you "
            "produced or trust.",
            path.name,
            safe_error,
        )
        return torch.load(path, map_location=device, weights_only=False)
