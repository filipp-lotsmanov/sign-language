"""
Replace H, P, T, W in Dataset
=============================
1. Load custom letters
2. Remove old H, P, T, W from main dataset
3. Add new ones
4. Normalize everything
5. Save final dataset

This stage deliberately does NOT augment. Augmentation multiplies each sample,
so doing it here - before training splits the data - scatters near-duplicates of
the same recording across train, validation and test, and the reported accuracy
stops measuring generalization. training/static/train.py augments the training
fold only, after the split.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import AUGMENTED_NPZ, CUSTOM_LANDMARKS_DIR, FINAL_NPZ  # noqa: E402

# Stage 3 of the pipeline: AUGMENTED_NPZ + custom recordings -> FINAL_NPZ.
CUSTOM_DIR = CUSTOM_LANDMARKS_DIR
MAIN_DATASET = AUGMENTED_NPZ
OUTPUT_PATH = FINAL_NPZ

LETTERS_TO_REPLACE = ["H", "P", "T", "W"]


# ============ NORMALIZATION ============


def normalize_landmarks(coords):
    """
    Normalize landmarks:
    1. Center on wrist (point 0)
    2. Scale by hand size (distance to middle finger base)
    """
    points = coords.reshape(21, 3)

    # Center on wrist
    wrist = points[0].copy()
    points = points - wrist

    # Scale by hand size (point 9 = middle finger base)
    scale_factor = np.linalg.norm(points[9])
    if scale_factor > 0.001:
        points = points / scale_factor

    return points.flatten()


# ============ MAIN PIPELINE ============


def main():
    print("=" * 60)
    print("REPLACING H, P, T, W IN DATASET")
    print("=" * 60)

    # 1. Load custom letters
    print("\n1. Loading custom letters...")
    custom_data = {}
    for letter in LETTERS_TO_REPLACE:
        path = CUSTOM_DIR / f"{letter}_landmarks.npy"
        if path.exists():
            data = np.load(path)
            custom_data[letter] = data
            print(f"   {letter}: {len(data)} samples")
        else:
            print(f"   {letter}: file not found!")

    if not custom_data:
        print("\nNo custom data found! Exiting.")
        return

    # 2. Collect the custom letters as-is (augmentation happens after the split)
    print("\n2. Collecting custom recordings...")
    custom_rows = []
    custom_labels = []

    for letter, data in custom_data.items():
        custom_rows.extend(list(data))
        custom_labels.extend([letter] * len(data))
        print(f"   {letter}: {len(data)} samples")

    custom_X = np.array(custom_rows, dtype=np.float32)
    custom_y = np.array(custom_labels)

    # 3. Load main dataset and remove old letters
    print("\n3. Loading main dataset...")
    main_data = np.load(MAIN_DATASET, allow_pickle=True)
    X_main = main_data["X"]
    y_main = main_data["y"]
    print(f"   Total: {len(X_main)} samples")

    # Count how many we'll remove
    for letter in LETTERS_TO_REPLACE:
        count = (y_main == letter).sum()
        print(f"   Removing {letter}: {count} samples")

    # Filter - keep everything except H, P, T, W
    mask = ~np.isin(y_main, LETTERS_TO_REPLACE)
    X_filtered = X_main[mask]
    y_filtered = y_main[mask]
    print(f"   After removal: {len(X_filtered)} samples")

    # 4. Merge
    print("\n4. Merging...")
    X_combined = np.vstack([X_filtered, custom_X])
    y_combined = np.concatenate([y_filtered, custom_y])
    print(f"   Total: {len(X_combined)} samples")

    # 5. Normalize ALL
    print("\n5. Normalizing...")
    X_normalized = np.array([normalize_landmarks(x) for x in X_combined], dtype=np.float32)

    # 6. Shuffle
    print("\n6. Shuffling...")
    indices = np.random.permutation(len(X_normalized))
    X_final = X_normalized[indices]
    y_final = y_combined[indices]

    # 7. Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n7. Saving: {OUTPUT_PATH}")
    np.savez(OUTPUT_PATH, X=X_final, y=y_final)

    # Also save CSV
    csv_path = str(OUTPUT_PATH).replace(".npz", ".csv")
    import pandas as pd

    columns = ["label"] + [f"coord_{i}" for i in range(63)]
    df = pd.DataFrame(np.column_stack([y_final, X_final]), columns=columns)
    df.to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}")

    # Report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"   Total samples: {len(X_final)}")
    print(f"   Classes: {len(set(y_final))}")
    print("\n   Per letter:")
    from collections import Counter

    counts = Counter(y_final)
    for letter in sorted(counts.keys()):
        marker = "[new]" if letter in LETTERS_TO_REPLACE else "  "
        print(f"   {marker} {letter}: {counts[letter]}")

    print(" Done!")


if __name__ == "__main__":
    main()
