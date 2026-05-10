"""
Dataset Creation: Normalize, Merge, and Prepare Data for Training
=================================================================
Handles: normalization, CSV merging, train/val/test splits, PyTorch Dataset.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pickle


def normalize_landmarks(coords):
    """
    Normalize hand landmarks for consistent input.
    
    1. Center on wrist (landmark 0)
    2. Scale by hand size (distance to middle finger base - landmark 9)
    
    Args:
        coords: numpy array of shape (63,) or (21, 3)
    
    Returns:
        Normalized coordinates of shape (63,)
    """
    points = coords.reshape(21, 3)
    
    # Center on wrist
    wrist = points[0].copy()
    points = points - wrist
    
    # Scale by hand size
    scale = np.linalg.norm(points[9])
    if scale > 0.001:
        points = points / scale
    
    return points.flatten()


def normalize_samples(samples):
    """
    Normalize all samples in a list.
    
    Args:
        samples: List of (label, coords) tuples
    
    Returns:
        List of (label, normalized_coords) tuples
    """
    normalized = []
    for label, coords in samples:
        norm_coords = normalize_landmarks(coords)
        normalized.append((label, norm_coords))
    return normalized


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
    Convert list of samples to pandas DataFrame.
    
    Args:
        samples: List of (label, coords) tuples
    
    Returns:
        DataFrame with 'label' column and 63 coordinate columns
    """
    labels = [s[0] for s in samples]
    coords = np.array([s[1] for s in samples])
    
    df = pd.DataFrame(
        np.column_stack([labels, coords]),
        columns=['label'] + [f'coord_{i}' for i in range(63)]
    )
    
    # Convert coordinate columns to float
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def merge_with_original(new_samples, original_csv, letters_to_replace):
    """
    Merge new samples with original dataset, replacing specified letters.
    
    Args:
        new_samples: List of (label, coords) tuples
        original_csv: Path to original CSV file
        letters_to_replace: List of letters to remove from original
    
    Returns:
        Merged DataFrame
    """
    # Load original
    df_original = pd.read_csv(original_csv)
    print(f"Loaded original: {len(df_original):,} samples")
    
    # Remove old letters
    for letter in letters_to_replace:
        count = (df_original['label'] == letter).sum()
        print(f"  Removing {letter}: {count:,} samples")
    
    mask = ~df_original['label'].isin(letters_to_replace)
    df_filtered = df_original[mask]
    print(f"After removal: {len(df_filtered):,} samples")
    
    # Convert new samples to DataFrame
    df_new = samples_to_dataframe(new_samples)
    
    # Merge
    df_merged = pd.concat([df_filtered, df_new], ignore_index=True)
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Merged total: {len(df_merged):,} samples")
    return df_merged


def prepare_data_splits(df, test_size=0.15, val_size=0.15):
    """
    Prepare train/val/test splits with label encoding.
    
    Args:
        df: DataFrame with 'label' column and coordinate columns
        test_size: Fraction for test set
        val_size: Fraction for validation set
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
    """
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df['label'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\nClasses ({len(le.classes_)}):")
    for idx, label in enumerate(le.classes_):
        marker = " ← NONSENSE" if label == "Nonsense" else ""
        print(f"  {idx:2d}: {label}{marker}")
    
    # Stratified splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
    )
    
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adjusted, stratify=y_temp, random_state=42
    )
    
    print(f"\nSplits: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, le


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=512):
    """
    Create PyTorch DataLoaders for training.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = LandmarkDataset(X_train, y_train)
    val_ds = LandmarkDataset(X_val, y_val)
    test_ds = LandmarkDataset(X_test, y_test)
    
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def save_label_encoder(le, path):
    """Save label encoder to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(le, f)
    print(f"Label encoder saved: {path}")


def load_label_encoder(path):
    """Load label encoder from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Example usage
    dummy_coords = np.random.randn(63).astype(np.float32)
    normalized = normalize_landmarks(dummy_coords)
    print(f"Normalized shape: {normalized.shape}")
    print(f"Wrist at origin: {normalized[:3]}")