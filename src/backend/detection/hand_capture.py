"""
Hand landmark detection using the MediaPipe Tasks HandLandmarker.

The training data is extracted with the Tasks API and the hand_landmarker.task
bundle (training/static/data_gathering.py, dataset_builder/frankenstein_builder.py).
Serving uses the same API, the same bundle and the same detection confidence so
that the landmarks reaching the model at inference time match the ones it was
trained on.
"""

from pathlib import Path
from typing import Optional
import logging

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.backend.detection.landmarks import normalize_landmarks

logger = logging.getLogger(__name__)

# src/backend/detection/hand_capture.py -> repository root
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "hand_landmarker.task"

# Must match training/static/data_gathering.py:init_mediapipe
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5

# (start, end) landmark index pairs, used only for debug visualization.
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


class HandCapture:
    """Extracts hand landmarks from BGR frames using MediaPipe Tasks."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
    ) -> None:
        """
        Args:
            model_path: path to hand_landmarker.task. Defaults to models/hand_landmarker.task.
            min_detection_confidence: must match the value used to build the training set.

        Raises:
            FileNotFoundError: if the landmarker bundle is missing.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand landmarker not found at {self.model_path}. "
                "Run scripts/setup.sh, or download it with:\n"
                "  curl -L -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            # IMAGE mode matches how the training set was extracted. VIDEO mode is
            # cheaper but applies temporal tracking, which shifts landmark values.
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)
        self.last_result = None

        logger.info("HandCapture initialized with %s", self.model_path.name)

    def extract_landmarks(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract raw hand landmarks from a frame.

        Args:
            frame: BGR image, or None.

        Returns:
            np.ndarray of shape (21, 3) with raw MediaPipe coordinates, or None
            if the frame was unusable or no hand was detected.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect(mp_image)
        self.last_result = result

        if not result.hand_landmarks:
            return None

        hand = result.hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

    def visualize_landmarks(self, frame: np.ndarray) -> None:
        """Draw the most recent detection on a frame, in place."""
        if not self.last_result or not self.last_result.hand_landmarks:
            return

        height, width = frame.shape[:2]
        points = [
            (int(lm.x * width), int(lm.y * height)) for lm in self.last_result.hand_landmarks[0]
        ]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)

    def close(self) -> None:
        """Release the underlying detector."""
        self.detector.close()

    def __enter__(self) -> "HandCapture":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def run_capture_loop(self) -> None:
        """Open the webcam and display the landmark overlay. Debugging aid."""
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            logger.error("Could not open webcam")
            return

        logger.info("Press 'q' to quit")
        try:
            while True:
                ret, frame = webcam.read()
                if not ret:
                    logger.error("Could not read frame")
                    break

                landmarks = self.extract_landmarks(frame)
                self.visualize_landmarks(frame)

                if landmarks is not None:
                    text, color = f"Hand detected: {landmarks.shape}", (0, 255, 0)
                else:
                    text, color = "No hand detected", (0, 0, 255)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow("Hand Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            webcam.release()
            cv2.destroyAllWindows()


# Backwards-compatible alias. Prefer importing normalize_landmarks directly.
# NOTE: this now returns shape (63,) and scales by landmark 9, matching training.
# The previous implementation returned (21, 3) and scaled by landmark 12.
normalize = normalize_landmarks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with HandCapture() as capture:
        capture.run_capture_loop()
