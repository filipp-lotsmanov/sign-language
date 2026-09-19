"""
Canonical hand-landmark normalization.

This is the ONLY definition of the transform. The training pipeline and the
serving path both go through it; if they diverge, every prediction is computed
on inputs the model never saw during training, which is exactly the bug
tests/test_landmarks.py::test_training_and_serving_agree exists to catch.

Contract:
    input : (21, 3) or (63,) raw MediaPipe landmarks
    output: (63,) float64, wrist-centered and scaled by the wrist-to-MCP distance

Why float64
-----------
The arithmetic is done and returned in float64 so the transform is genuinely
translation- and scale-invariant rather than invariant only to ~1e-6. In float32
the residual is around 4e-6, which is large enough to fail the invariance tests
and large enough to matter when comparing two implementations for equality.
The cast to float32 happens at the torch boundary, where the model needs it.

The transform is idempotent: normalizing already-normalized landmarks is a
no-op, because the wrist is already at the origin and the scale reference
already has unit norm. training/static/train.py relies on that when it
normalizes a frame mixing raw and pre-normalized rows.
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
NUM_COORDS = 3
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDS  # 63

WRIST_LANDMARK = 0
# Middle-finger MCP (the knuckle). Landmark 12 is the fingertip; using it here
# is the bug that shipped, and it put every served feature vector on a
# different scale than the training data.
SCALE_LANDMARK = 9

# Below this the hand is degenerate and dividing would amplify noise.
MIN_SCALE = 1e-3


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Center landmarks on the wrist and scale by hand size.

    Args:
        landmarks: raw MediaPipe landmarks, shape (21, 3) or (63,).

    Returns:
        np.ndarray of shape (63,), dtype float64.

    Raises:
        ValueError: if the input is not 21 xyz landmarks.
    """
    points = np.asarray(landmarks, dtype=np.float64)

    if points.size != NUM_FEATURES:
        raise ValueError(f"expected 21 xyz landmarks ({NUM_FEATURES} values), got {points.size}")

    points = points.reshape(NUM_LANDMARKS, NUM_COORDS)

    # Subtraction allocates, so the caller's array is never mutated.
    points = points - points[WRIST_LANDMARK]

    scale = float(np.linalg.norm(points[SCALE_LANDMARK]))
    if scale > MIN_SCALE:
        points = points / scale

    return points.reshape(NUM_FEATURES)


def normalize_samples(samples):
    """
    Normalize a list of (label, coords) tuples, preserving labels and order.

    Args:
        samples: iterable of (label, coords)

    Returns:
        list of (label, normalized_coords)
    """
    return [(label, normalize_landmarks(coords)) for label, coords in samples]


def to_model_input(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize and cast to the dtype the torch models expect.

    Keeping the cast here rather than inside normalize_landmarks means the
    transform stays exact while the model boundary still gets float32.
    """
    return normalize_landmarks(landmarks).astype(np.float32, copy=False)
