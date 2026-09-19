"""
NGT Dynamic Letter Recorder (J, Z)
==================================
Records fixed-length landmark sequences for the two letters that require
movement, and writes the .npz that training/dynamic/train.py expects.

Nothing in the repository produced that file before, so the LSTM could not be
retrained from scratch.

Output
------
    data_collect/ngt_dynamic/jz_dynamic_normalized.npz
        X : float32, shape (n_sequences, SEQUENCE_LENGTH, 63)  normalized
        y : str,     shape (n_sequences,)                      'J' or 'Z'

Each sequence is SEQUENCE_LENGTH frames captured back to back, matching the
LSTM's input window. Frames where the hand is not detected are dropped and the
take is retried, so a sequence never contains a gap.

Landmarks are extracted from the UNMIRRORED frame, matching the rest of the
dataset and the served application. Only the preview is mirrored.

Controls
--------
    SPACE  start a take for the current letter
    1 / 2  switch letter (J / Z)
    D      delete the most recent take
    Q      quit and save

Existing takes are loaded on start and appended to.
"""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.core.config import (  # noqa: E402
    DYNAMIC_BUFFER_SIZE,
    DYNAMIC_LETTERS,
    HAND_LANDMARKER_PATH,
    INPUT_SIZE,
)
from src.backend.detection.landmarks import normalize_landmarks  # noqa: E402

SEQUENCE_LENGTH = DYNAMIC_BUFFER_SIZE  # 30 frames
OUTPUT_DIR = PROJECT_ROOT / "data_collect" / "ngt_dynamic"
OUTPUT_PATH = OUTPUT_DIR / "jz_dynamic_normalized.npz"

TARGET_TAKES_PER_LETTER = 50  # guidance only; the recorder does not enforce it
COUNTDOWN_SECONDS = 1.0

HAND_CONNECTIONS = [
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
]


class DynamicRecorder:
    """Captures fixed-length landmark sequences for the dynamic letters."""

    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not HAND_LANDMARKER_PATH.exists():
            raise FileNotFoundError(
                f"MediaPipe hand landmarker not found at {HAND_LANDMARKER_PATH}. "
                "Run scripts/setup.sh first."
            )

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.letters = list(DYNAMIC_LETTERS)
        self.current_letter_idx = 0

        # sequences[letter] -> list of (SEQUENCE_LENGTH, 63) arrays
        self.sequences = {letter: [] for letter in self.letters}
        self.order = []  # letters in capture order, so D can undo the last take

        self._load_existing()

    def _load_existing(self):
        """Append to any previously recorded takes."""
        if not OUTPUT_PATH.exists():
            return
        try:
            data = np.load(OUTPUT_PATH, allow_pickle=True)
            for sequence, label in zip(data["X"], data["y"], strict=True):
                label = str(label)
                if label in self.sequences:
                    self.sequences[label].append(np.asarray(sequence, dtype=np.float32))
                    self.order.append(label)
            for letter in self.letters:
                print(f"Loaded {letter}: {len(self.sequences[letter])} sequences")
        except Exception as e:
            print(f"Could not load {OUTPUT_PATH}: {e}")

    def get_landmarks(self, frame):
        """Extract raw landmarks from an unmirrored BGR frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return None, None
        hand = result.hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
        return coords, hand

    @staticmethod
    def draw_hand(frame, hand_landmarks):
        """Draw the detected hand on an unmirrored frame."""
        height, width = frame.shape[:2]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

    def record_take(self, cap, letter):
        """
        Capture one SEQUENCE_LENGTH-frame sequence.

        Returns:
            (SEQUENCE_LENGTH, 63) float32 array, or None if the hand was lost.
        """
        deadline = time.time() + COUNTDOWN_SECONDS
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                return None
            preview = cv2.flip(frame, 1)
            remaining = deadline - time.time()
            cv2.putText(
                preview,
                f"{letter}  starting in {remaining:.1f}s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.imshow("NGT Dynamic Recorder", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None

        frames = []
        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                return None

            landmarks, hand = self.get_landmarks(frame)
            if landmarks is None:
                print("   hand lost - take discarded")
                return None

            frames.append(normalize_landmarks(landmarks))

            self.draw_hand(frame, hand)
            preview = cv2.flip(frame, 1)
            cv2.putText(
                preview,
                f"RECORDING {letter}  {len(frames)}/{SEQUENCE_LENGTH}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            cv2.imshow("NGT Dynamic Recorder", preview)
            cv2.waitKey(1)

        return np.asarray(frames, dtype=np.float32)

    def save(self):
        """Write the .npz that training/dynamic/train.py loads."""
        X, y = [], []
        for letter in self.letters:
            for sequence in self.sequences[letter]:
                X.append(sequence)
                y.append(letter)

        if not X:
            print("Nothing recorded; not writing a file.")
            return

        np.savez(
            OUTPUT_PATH,
            X=np.asarray(X, dtype=np.float32),
            y=np.asarray(y),
        )
        print(f"\nSaved {len(X)} sequences to {OUTPUT_PATH}")
        for letter in self.letters:
            print(f"   {letter}: {len(self.sequences[letter])}")

    def undo_last(self):
        """Remove the most recently captured take."""
        if not self.order:
            print("Nothing to undo")
            return
        letter = self.order.pop()
        self.sequences[letter].pop()
        print(f"Removed last {letter} take ({len(self.sequences[letter])} left)")

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Could not open the webcam")
            return

        print("\n" + "=" * 50)
        print("NGT DYNAMIC RECORDER (J / Z)")
        print("=" * 50)
        print(f"Each take records {SEQUENCE_LENGTH} consecutive frames.")
        print("SPACE - record a take    1/2 - switch letter")
        print("D     - undo last take   Q   - quit and save")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                letter = self.letters[self.current_letter_idx]
                landmarks, hand = self.get_landmarks(frame)
                hand_detected = landmarks is not None
                if hand_detected:
                    self.draw_hand(frame, hand)

                # Mirror for display only, after the overlay so it stays aligned.
                preview = cv2.flip(frame, 1)

                cv2.rectangle(preview, (0, 0), (640, 100), (40, 40, 40), -1)
                cv2.putText(
                    preview,
                    f"Letter: {letter}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                )
                count = len(self.sequences[letter])
                cv2.putText(
                    preview,
                    f"Takes: {count}/{TARGET_TAKES_PER_LETTER}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    preview,
                    "HAND OK" if hand_detected else "NO HAND",
                    (330, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if hand_detected else (0, 0, 255),
                    2,
                )

                cv2.imshow("NGT Dynamic Recorder", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    if not hand_detected:
                        print("No hand in frame; not starting a take")
                        continue
                    sequence = self.record_take(cap, letter)
                    if sequence is not None:
                        self.sequences[letter].append(sequence)
                        self.order.append(letter)
                        print(f"{letter}: {len(self.sequences[letter])} takes")
                elif key == ord("d"):
                    self.undo_last()
                elif key in (ord("1"), ord("2")):
                    self.current_letter_idx = key - ord("1")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            self.save()


if __name__ == "__main__":
    DynamicRecorder().run()


def _self_check():
    """Shape contract, exercised by tests without a webcam."""
    assert SEQUENCE_LENGTH == DYNAMIC_BUFFER_SIZE
    assert normalize_landmarks(np.zeros(INPUT_SIZE)).shape == (INPUT_SIZE,)
