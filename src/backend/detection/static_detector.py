"""
Static sign language detector using a ResidualMLP over hand landmarks.

predict() takes RAW MediaPipe landmarks and normalizes them internally, so
there is exactly one place normalization happens and no caller can feed the
model differently-scaled inputs than it was trained on.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.backend.detection.landmarks import to_model_input
from src.backend.models import config
from src.backend.models.checkpoint import load_checkpoint
from src.backend.models.cnn_model import ResidualMLP

logger = logging.getLogger(__name__)


class StaticSignPredictor:
    """
    Predictor for static NGT signs (all letters except J and Z).
    Uses a trained ResidualMLP to classify hand landmarks.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            model_path: path to trained model weights (.pth file)
            device: torch.device to run inference on
        """
        self.device = device if device else config.DEVICE
        self.model_path = Path(model_path) if model_path else config.MODEL_SAVE_PATH

        self.classes = self._load_classes()
        self.model = self._load_model()
        self.model.eval()

    def _load_classes(self) -> np.ndarray:
        """
        Load class labels from .npy or .pkl.

        Always returns an ndarray (possibly empty) so callers can use len()
        rather than truth-testing an array, which raises for len > 1.
        """
        model_dir = self.model_path.parent

        npy_path = model_dir / "classes.npy"
        if not npy_path.exists():
            npy_path = config.CLASSES_PATH
        if npy_path.exists():
            try:
                classes = np.load(npy_path, allow_pickle=True)
                logger.info("Loaded %d classes from classes.npy", len(classes))
                return classes
            except Exception as e:
                logger.warning("Failed to load classes.npy: %s", e)

        pkl_path = model_dir / "label_encoder.pkl"
        if not pkl_path.exists():
            pkl_path = config.LABEL_ENCODER_PATH
        if pkl_path.exists():
            try:
                # pickle.load executes code from the file. classes.npy above is
                # the preferred source precisely because it does not.
                logger.warning(
                    "Falling back to %s; unpickling executes code from that file. "
                    "Prefer classes.npy.",
                    pkl_path.name,
                )
                with open(pkl_path, "rb") as f:
                    label_encoder = pickle.load(f)
                classes = np.asarray(label_encoder.classes_)
                logger.info("Loaded %d classes from label_encoder.pkl", len(classes))
                return classes
            except Exception as e:
                logger.warning("Failed to load label_encoder.pkl: %s", e)

        logger.warning("No class labels found in %s", model_dir)
        return np.empty(0, dtype=object)

    def _load_model(self) -> ResidualMLP:
        """Load the model, supporting both metadata and bare state_dict checkpoints."""
        if not self.model_path.exists():
            logger.warning("Model not found at %s", self.model_path)
            return ResidualMLP(config.INPUT_SIZE, config.NUM_CLASSES).to(self.device)

        try:
            checkpoint = load_checkpoint(self.model_path, self.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                num_classes = checkpoint.get("num_classes", config.NUM_CLASSES)
                input_dim = checkpoint.get("input_dim", config.INPUT_SIZE)
                model_name = checkpoint.get("model_name", "ResidualMLP")

                model = ResidualMLP(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    hidden_dim=256,
                    num_blocks=4,
                    dropout=0.3,
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)

                logger.info("Loaded %s from %s", model_name, self.model_path.name)
                for metric in ("val_acc", "test_acc"):
                    if checkpoint.get(metric) is not None:
                        logger.info("Checkpoint %s: %.2f%%", metric, checkpoint[metric])
            else:
                # Legacy format: a bare state_dict with no metadata.
                num_classes = len(self.classes) if len(self.classes) else config.NUM_CLASSES
                model = ResidualMLP(input_dim=config.INPUT_SIZE, num_classes=num_classes)
                model.load_state_dict(checkpoint)
                model.to(self.device)
                logger.info("Loaded model (legacy format) from %s", self.model_path.name)

            return model

        except Exception as e:
            logger.error("Error loading model: %s", e)
            logger.warning("Creating new model with default parameters")
            return ResidualMLP(config.INPUT_SIZE, config.NUM_CLASSES).to(self.device)

    def predict(self, landmarks: np.ndarray) -> dict:
        """
        Predict a sign from RAW hand landmarks.

        Args:
            landmarks: raw MediaPipe landmarks, shape (21, 3) or (63,).
                       Do NOT pre-normalize; this method normalizes internally.

        Returns:
            dict with predicted_class, confidence and all_probabilities.
        """
        features = to_model_input(landmarks)

        input_tensor = torch.from_numpy(features).to(self.device).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, idx = probs.max(dim=1)

            idx = int(idx.item())
            confidence = float(confidence.item())

            if idx < len(self.classes):
                predicted_class = str(self.classes[idx])
            else:
                predicted_class = "Unknown"

            all_probs = {
                str(self.classes[i]): float(probs[0][i].item())
                for i in range(min(len(self.classes), probs.shape[1]))
            }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probs,
        }
