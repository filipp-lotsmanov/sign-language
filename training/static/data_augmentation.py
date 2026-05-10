"""
Data Augmentation for Hand Landmark Data
========================================
Applies geometric transformations to increase dataset size.
Transformations: noise, scale, rotation (2D/3D), translation, mirror, perspective.
"""

import numpy as np


def add_noise(coords, std=0.01):
    """Add random Gaussian noise to coordinates."""
    return coords + np.random.normal(0, std, coords.shape)


def scale_coords(coords, factor_range=(0.85, 1.15)):
    """Scale landmarks around center point."""
    factor = np.random.uniform(*factor_range)
    points = coords.reshape(-1, 3)
    center = points.mean(axis=0)
    return ((points - center) * factor + center).flatten()


def rotate_2d(coords, max_angle=25):
    """Rotate landmarks in XY plane (simulates hand rotation)."""
    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    points = coords.reshape(-1, 3)
    x, y = points[:, 0], points[:, 1]
    cx, cy = x.mean(), y.mean()
    x_new = (x - cx) * cos_a - (y - cy) * sin_a + cx
    y_new = (x - cx) * sin_a + (y - cy) * cos_a + cy
    points[:, 0], points[:, 1] = x_new, y_new
    return points.flatten()


def rotate_3d(coords, max_angle=15):
    """Rotate landmarks around Z axis (simulates camera tilt)."""
    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    points = coords.reshape(-1, 3)
    y, z = points[:, 1], points[:, 2]
    cy, cz = y.mean(), z.mean()
    y_new = (y - cy) * cos_a - (z - cz) * sin_a + cy
    z_new = (y - cy) * sin_a + (z - cz) * cos_a + cz
    points[:, 1], points[:, 2] = y_new, z_new
    return points.flatten()


def translate(coords, max_shift=0.08):
    """Translate landmarks randomly in XY plane."""
    points = coords.reshape(-1, 3)
    points[:, 0] += np.random.uniform(-max_shift, max_shift)
    points[:, 1] += np.random.uniform(-max_shift, max_shift)
    return points.flatten()


def mirror_x(coords):
    """Mirror landmarks along X axis (left-right flip)."""
    points = coords.reshape(-1, 3)
    points[:, 0] = -points[:, 0]
    return points.flatten()


def perspective_transform(coords, strength=0.1):
    """Apply perspective distortion (simulates camera angle)."""
    points = coords.reshape(-1, 3)
    tilt = np.random.uniform(-strength, strength)
    points[:, 2] += points[:, 1] * tilt
    return points.flatten()


def augment_sample(coords):
    """
    Apply random augmentations to a single sample.
    
    Args:
        coords: numpy array of shape (63,)
    
    Returns:
        Augmented coordinates of shape (63,)
    """
    aug = coords.copy()
    
    # Always add noise
    aug = add_noise(aug, std=np.random.uniform(0.005, 0.02))
    
    # Random transformations
    if np.random.random() < 0.7:
        aug = scale_coords(aug)
    if np.random.random() < 0.7:
        aug = rotate_2d(aug, max_angle=25)
    if np.random.random() < 0.5:
        aug = rotate_3d(aug, max_angle=15)
    if np.random.random() < 0.5:
        aug = translate(aug)
    if np.random.random() < 0.3:
        aug = mirror_x(aug)
    if np.random.random() < 0.3:
        aug = perspective_transform(aug)
    
    return aug


def augment_data(samples, multiplier):
    """
    Augment dataset by creating multiple variations of each sample.
    
    Args:
        samples: List of (label, coords) tuples
        multiplier: How many times to multiply the dataset
    
    Returns:
        Augmented list of (label, coords) tuples
    """
    print(f"Augmenting data x{multiplier}...")
    
    augmented = list(samples)  # Keep originals
    
    for _ in range(multiplier - 1):
        for label, coords in samples:
            aug_coords = augment_sample(coords)
            augmented.append((label, aug_coords))
    
    print(f"Done: Augmented: {len(samples)} -> {len(augmented)} samples")
    return augmented


def augment_by_class(samples, multiplier):
    """
    Augment data with per-class statistics.
    
    Args:
        samples: List of (label, coords) tuples
        multiplier: Augmentation multiplier
    
    Returns:
        Augmented list of (label, coords) tuples
    """
    # Group by label
    by_label = {}
    for label, coords in samples:
        if label not in by_label:
            by_label[label] = []
        by_label[label].append((label, coords))
    
    # Augment each class
    augmented = []
    for label, class_samples in by_label.items():
        original_count = len(class_samples)
        aug_samples = augment_data(class_samples, multiplier)
        augmented.extend(aug_samples)
        print(f"  {label}: {original_count} -> {len(aug_samples)}")
    
    return augmented


if __name__ == "__main__":
    # Example usage
    dummy_sample = np.random.randn(63).astype(np.float32)
    augmented = augment_sample(dummy_sample)
    print(f"Original shape: {dummy_sample.shape}")
    print(f"Augmented shape: {augmented.shape}")