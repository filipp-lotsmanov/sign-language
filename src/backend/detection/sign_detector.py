"""
Unified sign detector that routes frames to the static or dynamic predictor.

Both predictors take RAW MediaPipe landmarks and normalize internally, so this
class never normalizes anything itself.
"""

import logging
from typing import Optional

import numpy as np

from src.backend.core.config import DYNAMIC_BUFFER_SIZE, DYNAMIC_LETTERS
from src.backend.detection.dynamic_detector import DynamicSignPredictor
from src.backend.detection.hand_capture import HandCapture
from src.backend.detection.static_detector import StaticSignPredictor

logger = logging.getLogger(__name__)


class SignDetector:
    """
    Routes to the appropriate predictor based on letter type.
    Handles both static (ResidualMLP) and dynamic (LSTM) sign detection.
    """

    def __init__(self, static_model_path=None, dynamic_model_path=None, device=None):
        """
        Args:
            static_model_path: path to the ResidualMLP checkpoint
            dynamic_model_path: path to the LSTM checkpoint
            device: torch.device for inference
        """
        self.hand_capture = HandCapture()
        self.buffer: list[np.ndarray] = []

        self.static_predictor = None
        if static_model_path:
            try:
                self.static_predictor = StaticSignPredictor(static_model_path, device=device)
                logger.info("Static detector initialized with model: %s", static_model_path)
            except Exception as e:
                logger.error("Failed to load static model: %s", e)

        self.dynamic_predictor = None
        if dynamic_model_path:
            try:
                self.dynamic_predictor = DynamicSignPredictor(dynamic_model_path, device=device)
                logger.info("Dynamic detector initialized with model: %s", dynamic_model_path)
            except Exception as e:
                logger.error("Failed to load dynamic model: %s", e)

    def is_dynamic(self, letter: Optional[str]) -> bool:
        """Check whether a letter requires movement (dynamic detection)."""
        if not letter:
            return False
        return letter.upper() in DYNAMIC_LETTERS

    def process_frame(self, frame, target_letter):
        """
        Process a single frame and detect a sign language gesture.

        Args:
            frame: BGR image from the camera
            target_letter: the letter the user is trying to sign

        Returns:
            tuple: (landmarks, status, data, prediction_info)
                - landmarks: raw (21, 3) landmarks, or None
                - status: human-readable status message
                - data: the stacked sequence once the dynamic buffer is full, else None
                - prediction_info: prediction dict, or None
        """
        landmarks = self.hand_capture.extract_landmarks(frame)

        status = f"Letter: {target_letter} | Hand: NO"
        data = None
        prediction_info = None

        if landmarks is None:
            # Keep the dynamic buffer intact across dropped frames.
            return landmarks, status, data, prediction_info

        status = f"Letter: {target_letter} | Hand: YES"

        if self.is_dynamic(target_letter):
            # Dynamic letters (J, Z): accumulate a sequence for the LSTM.
            # add_frame normalizes internally and enforces the buffer length.
            if self.dynamic_predictor is None:
                status += " | No dynamic model"
                return landmarks, status, data, prediction_info

            self.dynamic_predictor.is_collecting = True
            is_ready = self.dynamic_predictor.add_frame(landmarks)
            self.buffer = list(self.dynamic_predictor.buffer)

            status += f" | Buffer: {len(self.buffer)}/{DYNAMIC_BUFFER_SIZE}"

            if is_ready:
                data = np.array(self.buffer, dtype=np.float32)
                status += f" | Ready: {data.shape}"
                try:
                    # predict() takes at most one positional argument.
                    prediction_info = self.dynamic_predictor.predict(data)
                except Exception as e:
                    logger.error("Dynamic prediction error: %s", e)
                    status += " | Err"
                else:
                    if prediction_info:
                        status += f" | Pred: {prediction_info['predicted_class']}"
        else:
            # Static letters: single-frame classification.
            self.clear_buffer()

            if self.static_predictor is None:
                status += " | No static model"
                return landmarks, status, data, prediction_info

            try:
                # Raw landmarks: the predictor normalizes internally.
                prediction_info = self.static_predictor.predict(landmarks)
            except Exception as e:
                logger.error("Prediction error: %s", e)
                status += " | Err"
            else:
                predicted_class = prediction_info["predicted_class"]
                status += f" | Pred: {predicted_class} ({prediction_info['confidence']:.2f})"
                if target_letter and predicted_class == target_letter.upper():
                    status += " [MATCH]"

        return landmarks, status, data, prediction_info

    def clear_buffer(self):
        """Clear the frame buffer (use when the target letter changes)."""
        self.buffer = []
        if self.dynamic_predictor is not None:
            self.dynamic_predictor.clear_buffer()
