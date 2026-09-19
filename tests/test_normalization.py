"""
Properties of the canonical normalization that tests/test_landmarks.py does not
already cover: idempotency, agreement with the historical training reference,
and the dtype boundary between the transform and the model.
"""

import numpy as np
import pytest

from src.backend.detection.landmarks import (
    NUM_FEATURES,
    SCALE_LANDMARK,
    normalize_landmarks,
    to_model_input,
)


def historical_training_reference(coords):
    """
    Verbatim copy of the pre-fix training/static/dataset_creation.py transform.

    The models in models/static were trained on data produced by exactly this
    code, so any divergence from it means the checkpoints are invalidated.
    """
    points = coords.reshape(21, 3)
    wrist = points[0].copy()
    points = points - wrist
    scale = np.linalg.norm(points[9])
    if scale > 0.001:
        points = points / scale
    return points.flatten()


@pytest.fixture
def raw_landmarks():
    rng = np.random.default_rng(1234)
    return rng.random((50, NUM_FEATURES))


class TestHistoricalParity:
    """The transform must still match what the shipped checkpoints were trained on."""

    def test_matches_the_reference_the_models_were_trained_with(self, raw_landmarks) -> None:
        for row in raw_landmarks:
            np.testing.assert_allclose(
                normalize_landmarks(row), historical_training_reference(row.copy()), atol=1e-12
            )


class TestIdempotency:
    """training/static/train.py normalizes a frame mixing raw and normalized rows."""

    def test_normalizing_twice_changes_nothing(self, raw_landmarks) -> None:
        once = normalize_landmarks(raw_landmarks[0])
        np.testing.assert_allclose(once, normalize_landmarks(once), atol=1e-12)

    def test_idempotent_across_many_samples(self, raw_landmarks) -> None:
        for row in raw_landmarks:
            once = normalize_landmarks(row)
            np.testing.assert_allclose(once, normalize_landmarks(once), atol=1e-12)


class TestModelBoundary:
    """
    The transform stays in float64 so it is exactly invariant; the cast to the
    dtype torch needs happens at the model boundary, not inside the transform.
    """

    def test_transform_is_float64(self, raw_landmarks) -> None:
        assert normalize_landmarks(raw_landmarks[0]).dtype == np.float64

    def test_model_input_is_float32(self, raw_landmarks) -> None:
        # A float64 array reaching a float32 Linear layer raises
        # "expected scalar type Float but found Double".
        assert to_model_input(raw_landmarks[0]).dtype == np.float32

    def test_model_input_matches_the_transform(self, raw_landmarks) -> None:
        np.testing.assert_allclose(
            to_model_input(raw_landmarks[0]),
            normalize_landmarks(raw_landmarks[0]),
            rtol=1e-6,
        )

    def test_model_input_is_accepted_by_a_linear_layer(self) -> None:
        import torch

        from src.backend.models.cnn_model import ResidualMLP

        model = ResidualMLP()
        model.eval()
        features = to_model_input(np.random.default_rng(0).random(NUM_FEATURES))
        with torch.no_grad():
            out = model(torch.from_numpy(features).unsqueeze(0))
        assert out.shape[0] == 1


class TestDegenerateInput:
    def test_all_zero_hand_is_finite(self) -> None:
        assert np.all(np.isfinite(normalize_landmarks(np.zeros(NUM_FEATURES))))

    def test_scale_landmark_at_origin_is_not_divided(self) -> None:
        collapsed = np.zeros((21, 3))
        collapsed[SCALE_LANDMARK] = [0.0, 1e-6, 0.0]
        out = normalize_landmarks(collapsed).reshape(21, 3)
        np.testing.assert_allclose(out[SCALE_LANDMARK], [0.0, 1e-6, 0.0], atol=1e-18)
