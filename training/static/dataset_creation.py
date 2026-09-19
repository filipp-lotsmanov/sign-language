"""
Dataset helpers: splitting, normalization, CSV merging, label encoding.

Split before augment
--------------------
`split_samples` exists so that splitting happens on raw samples, before
augmentation ever runs. The previous flow augmented x10 and split afterwards,
which scattered ten near-copies of every source sample across train, val and
test; the reported accuracy then measured recall of data the model had trained
on. `assert_no_duplicate_rows_across_splits` is the belt-and-braces check that
no identical row survives in a held-out split.

`normalize_landmarks` and `normalize_samples` are re-exported from
src/backend/detection/landmarks.py rather than reimplemented here. The transform
used to exist in four copies and one of them diverged, so the static model was
served feature vectors on a different scale than it was trained on.
"""

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.detection.landmarks import (  # noqa: E402
    normalize_landmarks,
    normalize_samples,
)

__all__ = [
    "LandmarkDataset",
    "assert_no_duplicate_rows_across_splits",
    "encode_splits",
    "load_label_encoder",
    "merge_with_original",
    "normalize_landmarks",
    "normalize_samples",
    "samples_to_dataframe",
    "save_label_encoder",
    "split_samples",
]

# A class needs at least one sample in each of train, val and test.
MIN_SAMPLES_PER_CLASS = 3


def split_samples(samples, test_size, val_size, seed=42):
    """
    Split (label, coords) samples into train/val/test, stratified by label.

    Splits indices rather than data, so the returned tuples hold the *same*
    coordinate objects as the input. Nothing is copied and nothing is augmented.

    Args:
        samples: list of (label, coords)
        test_size: test fraction of the whole dataset
        val_size: validation fraction of the whole dataset
        seed: RNG seed, so the split is reproducible

    Returns:
        (train, val, test), each a list of (label, coords)

    Raises:
        ValueError: if any class is too small to appear in all three splits.
            Rejected loudly rather than silently dropped, because a class that
            vanishes from the test set inflates the reported accuracy.
    """
    labels = [label for label, _ in samples]

    undersized = {
        label: count
        for label, count in sorted(Counter(labels).items())
        if count < MIN_SAMPLES_PER_CLASS
    }
    if undersized:
        raise ValueError(
            f"These classes have fewer than 3 samples and cannot be split three "
            f"ways: {undersized}. Record more, or drop them explicitly."
        )

    indices = np.arange(len(samples))

    idx_temp, idx_test = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=seed
    )
    # Re-scale so val_size stays a fraction of the ORIGINAL dataset.
    val_adjusted = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_temp,
        test_size=val_adjusted,
        stratify=[labels[i] for i in idx_temp],
        random_state=seed,
    )

    def pick(chosen):
        return [samples[i] for i in chosen]

    return pick(idx_train), pick(idx_val), pick(idx_test)


def assert_no_duplicate_rows_across_splits(train_X, held_out_X, split_name):
    """
    Fail if any held-out row is byte-identical to a training row.

    Catches leakage that survives a correct split, for example when the same
    recording was ingested twice under different filenames.

    Args:
        train_X: (n, features) training matrix
        held_out_X: (m, features) validation or test matrix
        split_name: name used in the error message

    Raises:
        RuntimeError: if any exact duplicate is found.
    """
    train_X = np.asarray(train_X)
    held_out_X = np.asarray(held_out_X)

    if train_X.size == 0 or held_out_X.size == 0:
        return

    train_rows = {row.tobytes() for row in np.ascontiguousarray(train_X, dtype=np.float64)}
    duplicates = [
        i
        for i, row in enumerate(np.ascontiguousarray(held_out_X, dtype=np.float64))
        if row.tobytes() in train_rows
    ]

    if duplicates:
        shown = duplicates[:10]
        raise RuntimeError(
            f"DUPLICATE ROWS: {len(duplicates)} row(s) in the '{split_name}' split are "
            f"byte-identical to training rows (indices {shown}"
            f"{'...' if len(duplicates) > len(shown) else ''}). "
            "The reported accuracy for this split would be inflated."
        )


def encode_splits(df_train, df_val, df_test):
    """
    Encode labels consistently across all three splits.

    The encoder is fit on the union of the splits, so a label index means the
    same class everywhere. Fitting per split would silently remap classes.

    Args:
        df_train, df_val, df_test: DataFrames with a 'label' column followed by
            the coordinate columns.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    """
    frames = (df_train, df_val, df_test)

    encoder = LabelEncoder()
    encoder.fit(pd.concat([df["label"] for df in frames], ignore_index=True))

    encoded = []
    for df in frames:
        encoded.append(df.iloc[:, 1:].to_numpy(dtype=np.float32))
        encoded.append(encoder.transform(df["label"].to_numpy()))

    return (*encoded, encoder)


class LandmarkDataset(Dataset):
    """PyTorch Dataset for landmark data."""

    def __init__(self, X, y):
        """
        Args:
            X: numpy array of features, shape (n_samples, 63)
            y: numpy array of encoded labels, shape (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def samples_to_dataframe(samples):
    """
    Convert a list of (label, coords) tuples to a DataFrame.

    Builds the numeric columns directly rather than round-tripping every
    coordinate through a string via np.column_stack.
    """
    labels = [label for label, _ in samples]
    coords = np.asarray([c for _, c in samples], dtype=np.float32)

    df = pd.DataFrame(coords, columns=[f"coord_{i}" for i in range(coords.shape[1])])
    df.insert(0, "label", labels)
    return df


def merge_with_original(new_samples, original_csv, letters_to_replace):
    """
    Merge new samples with the original dataset, replacing the given letters.

    Returns a merged DataFrame. NOT shuffled and NOT split: the caller splits
    before augmenting.
    """
    df_original = pd.read_csv(original_csv)
    print(f"Loaded original: {len(df_original):,} samples")

    for letter in letters_to_replace:
        count = (df_original["label"] == letter).sum()
        print(f"  Removing {letter}: {count:,} samples")

    df_filtered = df_original[~df_original["label"].isin(letters_to_replace)]
    print(f"After removal: {len(df_filtered):,} samples")

    df_merged = pd.concat([df_filtered, samples_to_dataframe(new_samples)], ignore_index=True)
    print(f"Merged total: {len(df_merged):,} samples")
    return df_merged


def save_label_encoder(le, path):
    """Save label encoder to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(le, f)
    print(f"Label encoder saved: {path}")


def load_label_encoder(path):
    """
    Load a label encoder from a pickle file.

    Unpickling executes code from the file. Prefer classes.npy, which the
    training script also writes, wherever only the class list is needed.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
