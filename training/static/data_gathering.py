"""
Data Gathering: Extract Hand Landmarks from Images
===================================================
Uses MediaPipe to detect hands and extract 21 landmark points.
Each landmark has (x, y, z) coordinates -> 63 features total.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import MEDIAPIPE_MODEL


def init_mediapipe(model_path=None):
    """
    Initialize MediaPipe Hand Landmarker.

    Args:
        model_path: Path to hand_landmarker.task file. Defaults to config.MEDIAPIPE_MODEL.

    Returns:
        HandLandmarker detector instance
    """
    if model_path is None:
        model_path = MEDIAPIPE_MODEL
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_image(detector, img_path):
    """
    Extract 21 hand landmarks from a single image.

    Args:
        detector: MediaPipe HandLandmarker instance
        img_path: Path to image file

    Returns:
        numpy array of shape (63,) or None if no hand detected
    """
    try:
        mp_image = mp.Image.create_from_file(str(img_path))
        results = detector.detect(mp_image)

        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand = results.hand_landmarks[0]
            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords, dtype=np.float32)
    except Exception as e:
        pass
    return None


def extract_from_folder(detector, folder_path, label):
    """
    Extract landmarks from all images in a folder.

    Args:
        detector: MediaPipe HandLandmarker instance
        folder_path: Path to folder containing images
        label: Class label for these images

    Returns:
        List of (label, landmarks) tuples
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"  Folder not found: {folder_path}")
        return []

    # Collect all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in extensions:
        images.extend(folder_path.glob(ext))

    if not images:
        print(f"  No images found in: {folder_path}")
        return []

    # Extract landmarks
    landmarks_list = []
    failed = 0

    for img_path in tqdm(images, desc=f"  {label}", leave=False):
        lm = extract_landmarks_from_image(detector, img_path)
        if lm is not None:
            landmarks_list.append((label, lm))
        else:
            failed += 1

    print(f"  {label}: {len(landmarks_list)} extracted, {failed} failed")
    return landmarks_list


def gather_data(photos_dir, letter_folders, mediapipe_model=None):
    """
    Main function to gather all training data from photo folders.

    Args:
        photos_dir: Base directory containing letter folders
        letter_folders: Dict mapping labels to folder names
                       e.g., {'A': 'A_letter', 'Nonsense': 'Nonsense'}
        mediapipe_model: Path to MediaPipe model file. Defaults to config.MEDIAPIPE_MODEL.
    
    Returns:
        List of (label, landmarks) tuples
    """
    print("=" * 60)
    print("DATA GATHERING: EXTRACTING LANDMARKS")
    print("=" * 60)
    
    detector = init_mediapipe(mediapipe_model)
    print("MediaPipe initialized\n")
    
    photos_dir = Path(photos_dir)
    all_samples = []
    
    for label, folder_name in letter_folders.items():
        folder_path = photos_dir / folder_name
        samples = extract_from_folder(detector, folder_path, label)
        all_samples.extend(samples)
    
    print(f"\nTotal extracted: {len(all_samples)} samples")
    return all_samples


if __name__ == "__main__":
    # Example usage
    LETTER_FOLDERS = {
        'D': 'D_letter',
        'F': 'F_letter',
        'Nonsense': 'Nonsense'
    }
    
    samples = gather_data("letters", LETTER_FOLDERS)
    print(f"Gathered {len(samples)} samples")