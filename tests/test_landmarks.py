"""
Regression tests for the shared landmark normalisation.

The bug these exist to prevent: the same transform was implemented four times
and one copy diverged, so the static model was served feature vectors on a
different scale than it was trained on. Any future copy that drifts should
fail test_training_and_serving_agree.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.backend.detection.landmarks import (
    MIN_SCALE,
    NUM_FEATURES,
    SCALE_LANDMARK,
    WRIST_LANDMARK,
    normalize_landmarks,
    normalize_samples,
)


@pytest.fixture
def hand() -> np.ndarray:
    """A plausible 21-landmark hand, wrist offset from the origin."""
    rng = np.random.default_rng(0)
    lm = rng.uniform(-1, 1, (21, 3)) * np.array([0.15, 0.15, 0.05])
    lm[0] = np.array([0.4, 0.6, 0.0])
    lm[9] = lm[0] + np.array([0.0, -0.08, 0.0])
    lm[12] = lm[0] + np.array([0.0, -0.19, 0.0])
    return lm


class TestPipelineParity:
    """Every consumer of the transform must get identical numbers."""

    def test_training_and_serving_agree(self, hand: np.ndarray) -> None:
        # Loaded by path, not by import name: training/ is a directory of
        # scripts, not an installed package, so this has to work regardless
        # of the working directory pytest was started from.
        script = (
            Path(__file__).resolve().parent.parent / "training" / "static" / "dataset_creation.py"
        )
        spec = importlib.util.spec_from_file_location("_training_dataset_creation", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        np.testing.assert_array_equal(normalize_landmarks(hand), module.normalize_landmarks(hand))

    def test_dynamic_detector_delegates(self, hand: np.ndarray) -> None:
        from src.backend.detection.dynamic_detector import DynamicSignPredictor

        predictor = DynamicSignPredictor.__new__(DynamicSignPredictor)
        np.testing.assert_array_equal(
            predictor.normalize_landmarks(hand), normalize_landmarks(hand)
        )

    def test_scale_landmark_is_the_knuckle_not_the_fingertip(self) -> None:
        """Pins the constant itself. 12 is the bug that shipped."""
        assert SCALE_LANDMARK == 9
        assert WRIST_LANDMARK == 0


class TestNormalization:
    def test_returns_flat_63(self, hand: np.ndarray) -> None:
        assert normalize_landmarks(hand).shape == (NUM_FEATURES,)

    def test_accepts_flat_and_nested_alike(self, hand: np.ndarray) -> None:
        np.testing.assert_array_equal(
            normalize_landmarks(hand), normalize_landmarks(hand.flatten())
        )

    def test_wrist_lands_on_the_origin(self, hand: np.ndarray) -> None:
        out = normalize_landmarks(hand).reshape(21, 3)
        np.testing.assert_allclose(out[WRIST_LANDMARK], 0.0, atol=1e-12)

    def test_scale_landmark_ends_at_unit_distance(self, hand: np.ndarray) -> None:
        out = normalize_landmarks(hand).reshape(21, 3)
        assert np.linalg.norm(out[SCALE_LANDMARK]) == pytest.approx(1.0)

    def test_translation_invariant(self, hand: np.ndarray) -> None:
        shifted = hand + np.array([0.3, -0.2, 0.05])
        np.testing.assert_allclose(
            normalize_landmarks(hand), normalize_landmarks(shifted), atol=1e-12
        )

    def test_scale_invariant(self, hand: np.ndarray) -> None:
        np.testing.assert_allclose(
            normalize_landmarks(hand), normalize_landmarks(hand * 2.5), atol=1e-12
        )

    def test_input_is_not_mutated(self, hand: np.ndarray) -> None:
        before = hand.copy()
        normalize_landmarks(hand)
        np.testing.assert_array_equal(hand, before)

    def test_degenerate_hand_is_not_amplified(self) -> None:
        collapsed = np.zeros((21, 3))
        collapsed[SCALE_LANDMARK] = MIN_SCALE / 10
        out = normalize_landmarks(collapsed)
        assert np.isfinite(out).all()
        assert np.abs(out).max() <= MIN_SCALE

    @pytest.mark.parametrize("bad", [np.zeros((20, 3)), np.zeros(62), np.zeros((21, 2))])
    def test_wrong_shape_is_rejected(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError, match="21 xyz landmarks"):
            normalize_landmarks(bad)


class TestNormalizeSamples:
    def test_preserves_labels_and_order(self, hand: np.ndarray) -> None:
        out = normalize_samples([("A", hand), ("B", hand * 2)])
        assert [label for label, _ in out] == ["A", "B"]
        np.testing.assert_allclose(out[0][1], out[1][1], atol=1e-12)
