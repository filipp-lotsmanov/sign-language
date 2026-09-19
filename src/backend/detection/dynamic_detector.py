"""
Dynamic sign language detector for letters requiring movement (J, Z).
LSTM-based detection using trained model.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.backend.core.config import (
    DYNAMIC_BUFFER_SIZE,
    DYNAMIC_CLASSES_PATH,
    DYNAMIC_LETTERS,
    DYNAMIC_MIN_BUFFER,
    DYNAMIC_MODEL_PATH,
    INPUT_SIZE,
)
from src.backend.detection.landmarks import normalize_landmarks as _normalize_landmarks
from src.backend.models.checkpoint import load_checkpoint
from src.backend.models.lstm_model import DynamicSignLSTM

logger = logging.getLogger(__name__)


class DynamicSignPredictor:
    """
    Predictor for dynamic NGT signs (J and Z) that require movement.
    Uses LSTM to analyze sequences of hand landmarks over time.
    """

    def __init__(
        self, model_path: Optional[str] = None, device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize the dynamic sign predictor.

        Args:
            model_path: Path to trained LSTM model weights
            device: torch.device to run inference on
        """
        self.device = (
            device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Set model path - default to models/dynamic/
        if model_path is None:
            model_path = DYNAMIC_MODEL_PATH

        self.model_path = Path(model_path)
        self.model = None
        # models/README.md advertises dynamic/classes.npy, but the class list
        # used to be hardcoded here and the file was never read.
        self.classes = self._load_classes()
        self.sequence_length = DYNAMIC_BUFFER_SIZE
        self.input_size = INPUT_SIZE

        # Frame buffer for collecting sequences
        self.buffer = deque(maxlen=self.sequence_length)
        self.is_collecting = False

        # Load model
        self._load_model()

    def _load_classes(self) -> list:
        """
        Load the class list next to the checkpoint, falling back to the
        built-in dynamic letters when the file is absent.
        """
        candidates = [self.model_path.parent / "classes.npy", DYNAMIC_CLASSES_PATH]
        for path in candidates:
            if not path.exists():
                continue
            try:
                classes = [str(c) for c in np.load(path, allow_pickle=True)]
                if classes:
                    logger.info("Loaded %d dynamic classes from %s", len(classes), path.name)
                    return classes
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

        logger.info("No dynamic classes.npy found; using %s", DYNAMIC_LETTERS)
        return list(DYNAMIC_LETTERS)

    def _load_model(self) -> None:
        """Load trained LSTM model."""
        if not self.model_path.exists():
            logger.warning("Dynamic model not found: %s", self.model_path)
            logger.warning("Dynamic letters (J, Z) will not work")
            return

        try:
            checkpoint = load_checkpoint(self.model_path, self.device)
            model_config = checkpoint.get("config", {})

            self.model = DynamicSignLSTM(
                input_size=model_config.get("input_size", self.input_size),
                hidden_size=model_config.get("hidden_size", 128),
                num_layers=model_config.get("num_layers", 2),
                num_classes=model_config.get("num_classes", len(self.classes)),
            )

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            val_acc = checkpoint.get("val_acc", checkpoint.get("accuracy", "N/A"))
            logger.info("Dynamic model loaded: %s", self.model_path.name)
            if val_acc != "N/A":
                logger.info("Validation accuracy: %.3f", val_acc)

        except Exception as e:
            logger.error("Error loading dynamic model: %s", e)
            self.model = None

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks using the shared canonical transform.

        Args:
            landmarks: raw MediaPipe landmarks, shape (21, 3) or (63,)
        Returns:
            Normalized flat array of shape (63,)
        """
        return _normalize_landmarks(landmarks)

    def start_collecting(self) -> None:
        """Start collecting frames for a dynamic gesture."""
        self.buffer.clear()
        self.is_collecting = True

    def stop_collecting(self) -> None:
        """Stop collecting frames."""
        self.is_collecting = False

    def add_frame(self, landmarks: np.ndarray) -> bool:
        """
        Add a frame to the buffer.

        Args:
            landmarks: np.array of shape (21, 3) or (63,)
        Returns:
            True if buffer is full and ready for prediction
        """
        if not self.is_collecting:
            return False

        normalized = self.normalize_landmarks(landmarks)
        self.buffer.append(normalized)

        return len(self.buffer) == self.sequence_length

    def predict(self, landmark_sequence: Optional[np.ndarray] = None) -> Optional[dict]:
        """
        Predict dynamic sign from sequence of hand landmarks.

        Args:
            landmark_sequence: Optional np.array of shape (seq_len, 63)
                             If None, uses internal buffer

        Returns:
            dict with:
                - predicted_class: str, predicted letter
                - confidence: float, confidence score
                - all_probabilities: dict, probabilities for J and Z
        """
        if self.model is None:
            return {
                "predicted_class": "Unknown",
                "confidence": 0.0,
                "all_probabilities": {c: 0.0 for c in self.classes},
            }

        # Use provided sequence or buffer
        if landmark_sequence is not None:
            seq = landmark_sequence
        else:
            if len(self.buffer) < DYNAMIC_MIN_BUFFER:
                return None
            seq = np.array(list(self.buffer), dtype=np.float32)

        # Interpolate to sequence_length if needed
        if len(seq) != self.sequence_length:
            seq = self._interpolate_sequence(seq)

        # Convert to tensor
        X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = float(probs[pred_idx].item())

        # Guard against a checkpoint with more outputs than the class list.
        predicted_class = self.classes[pred_idx] if pred_idx < len(self.classes) else "Unknown"
        all_probs = {
            self.classes[i]: float(probs[i].item())
            for i in range(min(len(self.classes), probs.shape[0]))
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probs,
        }

    def _interpolate_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Interpolate sequence to target length."""
        try:
            from scipy.interpolate import interp1d

            x_old = np.linspace(0, 1, len(seq))
            x_new = np.linspace(0, 1, self.sequence_length)
            seq_interp = np.zeros((self.sequence_length, INPUT_SIZE))
            for i in range(INPUT_SIZE):
                f = interp1d(x_old, seq[:, i], kind="linear")
                seq_interp[:, i] = f(x_new)
            return seq_interp.astype(np.float32)
        except ImportError:
            if len(seq) > self.sequence_length:
                return seq[: self.sequence_length]
            else:
                padding = np.tile(seq[-1], (self.sequence_length - len(seq), 1))
                return np.vstack([seq, padding]).astype(np.float32)

    def get_buffer_progress(self) -> float:
        """Get current buffer fill percentage (0-1)."""
        return len(self.buffer) / self.sequence_length

    def clear_buffer(self) -> None:
        """Clear the frame buffer."""
        self.buffer.clear()
