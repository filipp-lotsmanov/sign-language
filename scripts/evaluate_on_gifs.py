"""
Evaluate the static classifier against the tutorial GIFs in src/assets/.

Why these GIFs: they are labelled by filename, they show a signer who is not
in the training recordings, and they are already in the repo. That makes them
the only held-out material available without rebuilding the dataset.

What this is NOT: a test set. One signer, one camera, one viewpoint, and the
GIFs are third-party footage of NGT reference forms, so a letter can score 0%
because the dataset taught a different (ASL/DGS) hand shape for it rather than
because the model is broken. Read the numbers as a comparison between two
configurations on identical frames, not as an accuracy estimate.

Usage:
    python scripts/evaluate_on_gifs.py
    python scripts/evaluate_on_gifs.py --compare-scale-landmarks
    python scripts/evaluate_on_gifs.py --keep-last 0.5 --per-letter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.core.config import ASSETS_DIR, STATIC_LETTERS, STATIC_MODEL_PATH  # noqa: E402
from src.backend.detection.landmarks import SCALE_LANDMARK, WRIST_LANDMARK  # noqa: E402
from src.backend.detection.static_detector import StaticSignPredictor  # noqa: E402

LANDMARKER_TASK = PROJECT_ROOT / "models" / "hand_landmarker.task"


def build_detector():
    """MediaPipe Tasks landmarker, configured as training/static/data_gathering.py does."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not LANDMARKER_TASK.exists():
        sys.exit(
            f"Missing {LANDMARKER_TASK}.\n"
            "Run scripts/setup.sh, or download it from the v0.1.0 release."
        )
    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_TASK)),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return detector, mp


def normalize_by(coords: np.ndarray, scale_landmark: int) -> np.ndarray:
    """Normalisation with a configurable scale landmark, for A/B comparison only.

    Production code must use detection.landmarks.normalize_landmarks; this
    exists so the shipped choice can be measured against the correct one.
    """
    points = coords.reshape(21, 3) - coords.reshape(21, 3)[WRIST_LANDMARK]
    scale = np.linalg.norm(points[scale_landmark])
    if scale > 1e-3:
        points = points / scale
    return points.flatten()


def collect_landmarks(keep_last: float) -> list[tuple[str, np.ndarray]]:
    """Extract landmarks from the trailing `keep_last` fraction of each GIF.

    Early frames catch the hand moving into position, so they are labelled with
    a letter that is not yet being formed. Trimming them measures the letter
    rather than the transition.
    """
    detector, mp = build_detector()
    samples: list[tuple[str, np.ndarray]] = []
    missed = 0

    for gif in sorted(Path(ASSETS_DIR).glob("*.gif")):
        letter = gif.stem.upper()
        if letter not in STATIC_LETTERS:
            continue

        capture = cv2.VideoCapture(str(gif))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()

        for frame in frames[int(len(frames) * (1.0 - keep_last)) :]:
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = detector.detect(image)
            if not result.hand_landmarks:
                missed += 1
                continue
            hand = result.hand_landmarks[0]
            samples.append(
                (letter, np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float64).flatten())
            )

    if missed:
        print(f"note: no hand found in {missed} frame(s)")
    return samples


def score(predictor: StaticSignPredictor, samples, scale_landmark: int):
    correct = nonsense = 0
    per_letter: dict[str, list[int]] = {}
    for letter, coords in samples:
        predicted = predictor.predict(normalize_by(coords, scale_landmark))["predicted_class"]
        hit = int(predicted == letter)
        correct += hit
        nonsense += int(predicted == "Nonsense")
        per_letter.setdefault(letter, []).append(hit)
    total = len(samples)
    return {
        "top1": 100.0 * correct / total,
        "nonsense": 100.0 * nonsense / total,
        "per_letter": {k: 100.0 * sum(v) / len(v) for k, v in sorted(per_letter.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-last",
        type=float,
        default=0.5,
        help="trailing fraction of each GIF to use (default: 0.5)",
    )
    parser.add_argument("--per-letter", action="store_true", help="print a per-letter breakdown")
    parser.add_argument(
        "--compare-scale-landmarks",
        action="store_true",
        help="also score with the old landmark-12 scaling, for comparison",
    )
    args = parser.parse_args()

    if not STATIC_MODEL_PATH.exists():
        sys.exit(f"Missing {STATIC_MODEL_PATH}. Run scripts/setup.sh first.")

    samples = collect_landmarks(args.keep_last)
    if not samples:
        sys.exit("No landmarks extracted - check src/assets/ contains the tutorial GIFs.")

    predictor = StaticSignPredictor(str(STATIC_MODEL_PATH), device=torch.device("cpu"))

    print(
        f"\n{len(samples)} frames from the last {args.keep_last:.0%} of "
        f"{len({s[0] for s in samples})} static-letter GIFs\n"
    )

    main_result = score(predictor, samples, SCALE_LANDMARK)
    print(
        f"scale landmark {SCALE_LANDMARK} (canonical)   "
        f"top-1 {main_result['top1']:5.1f}%   nonsense {main_result['nonsense']:5.1f}%"
    )

    if args.compare_scale_landmarks:
        old = score(predictor, samples, 12)
        print(
            f"scale landmark 12 (pre-fix bug)   "
            f"top-1 {old['top1']:5.1f}%   nonsense {old['nonsense']:5.1f}%"
        )
        print(f"{'':34}delta {main_result['top1'] - old['top1']:+.1f} pts")

    if args.per_letter:
        print("\nper-letter top-1:")
        for letter, acc in main_result["per_letter"].items():
            bar = "#" * int(acc / 5)
            print(f"  {letter}  {acc:5.1f}%  {bar}")
        zeros = [k for k, v in main_result["per_letter"].items() if v == 0.0]
        if zeros:
            print(f"\n  never recognised: {', '.join(zeros)}")
            print("  check whether the dataset teaches a different hand shape for these")


if __name__ == "__main__":
    main()
