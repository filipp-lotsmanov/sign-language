"""
Training Script for the Static Sign Language Classifier
=======================================================
Trains ResidualMLP with early stopping and saves the best checkpoint.

Ordering guarantee
------------------
Augmentation runs AFTER the train/val/test split and only on the training fold.
Augmenting first, as this script previously did, puts ten near-duplicates of the
same source photo across all three folds and turns the reported accuracy into a
memorization score rather than a generalization estimate.

The dataset_builder stages no longer pre-augment either:
merge_custom_letters.py emits raw samples and augment_landmarks.py is deprecated
behind a --force guard, so an INPUT_CSV regenerated with the current pipeline
carries no duplicates across folds.

AUGMENT_CSV_FOLD below controls whether the pre-built CSV rows are augmented too.
It defaults to False because the augmentation magnitudes in data_augmentation.py
are tuned for raw MediaPipe coordinates in [0, 1], while existing CSVs were
written already normalized. Turn it on once INPUT_CSV holds raw landmarks.
"""

import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Allow both `python train.py` from this directory and `python training/static/train.py`
# from the repository root.
for path in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import config  # noqa: E402  training/static/config.py
from data_augmentation import augment_sample  # noqa: E402
from data_gathering import gather_data  # noqa: E402
from dataset_creation import (  # noqa: E402
    assert_no_duplicate_rows_across_splits,
    split_samples,
)

# One model definition, shared with the serving path.
from src.backend.detection.landmarks import normalize_landmarks  # noqa: E402
from src.backend.models.cnn_model import ResidualMLP  # noqa: E402

# Write straight to where the application loads models from, so there is no
# manual copy step between training and serving.
OUTPUT_DIR = PROJECT_ROOT / "models" / "static"
MODEL_OUTPUT = OUTPUT_DIR / "best_model.pth"
ENCODER_OUTPUT = OUTPUT_DIR / "label_encoder.pkl"
CLASSES_OUTPUT = OUTPUT_DIR / "classes.npy"

SEED = 42

# Augment the CSV-sourced training rows as well as the newly recorded photos.
# See the module docstring for the precondition.
AUGMENT_CSV_FOLD = False


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


def set_seed(seed: int = SEED) -> None:
    """Make the run reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_three_way_split(X, y, test_size, val_size, seed=SEED):
    """
    Split into train/val/test, stratified by label.

    Delegates to dataset_creation.split_samples so the split logic used here is
    the one covered by tests/test_splits.py.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    samples = list(zip(y, X, strict=True))
    train, val, test = split_samples(samples, test_size, val_size, seed=seed)

    def unpack(split):
        if not split:
            return np.empty((0, X.shape[1]), dtype=X.dtype), np.empty(0, dtype=object)
        labels, coords = zip(*split, strict=True)
        return np.asarray(coords), np.asarray(labels)

    return unpack(train), unpack(val), unpack(test)


def augment_training_fold(X_train, y_train, multiplier):
    """
    Expand the training fold only.

    Augmentation runs on raw coordinates, which is the space the magnitudes in
    data_augmentation.py were tuned for. Normalization happens afterwards.
    """
    if multiplier <= 1:
        return X_train, y_train

    extra_X, extra_y = [], []
    for _ in range(multiplier - 1):
        for coords, label in zip(X_train, y_train, strict=True):
            extra_X.append(augment_sample(coords))
            extra_y.append(label)

    X_out = np.vstack([X_train, np.asarray(extra_X, dtype=np.float32)])
    y_out = np.concatenate([y_train, np.asarray(extra_y)])
    return X_out, y_out


def normalize_matrix(X):
    """Apply the canonical normalization row-wise."""
    if len(X) == 0:
        return np.empty((0, 63), dtype=np.float32)
    return np.asarray([normalize_landmarks(row) for row in X], dtype=np.float32)


def load_csv_source(csv_path, letters_to_replace):
    """
    Load the pre-built dataset, dropping the letters this run re-records.

    Returns:
        (X, y) with X already normalized upstream. Normalization is idempotent,
        so re-applying it later is safe either way.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded original: {len(df):,} samples")

    for letter in letters_to_replace:
        print(f"  Removing {letter}: {(df['label'] == letter).sum():,} samples")

    df = df[~df["label"].isin(letters_to_replace)]
    print(f"After removal: {len(df):,} samples")

    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df["label"].values
    return X, y


def build_dataloaders(splits, label_encoder, batch_size):
    """Encode labels and wrap each split in a DataLoader."""
    loaders = []
    for index, (X, y) in enumerate(splits):
        dataset = TensorDataset(
            torch.from_numpy(np.asarray(X, dtype=np.float32)),
            torch.from_numpy(label_encoder.transform(y).astype(np.int64)),
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(index == 0),  # shuffle the training split only
                num_workers=4 if torch.cuda.is_available() else 0,
                pin_memory=torch.cuda.is_available(),
            )
        )
    return loaders


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on a loader. Returns (loss, accuracy, preds, targets)."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    return total_loss / len(loader), 100.0 * correct / total, all_preds, all_targets


def print_per_class_accuracy(y_true, y_pred, label_encoder):
    """Print accuracy for each class."""
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    for true, pred in zip(y_true, y_pred, strict=True):
        total_per_class[true] += 1
        if true == pred:
            correct_per_class[true] += 1

    print("\nPer-class accuracy:")
    print("-" * 50)

    weak_classes = []
    for idx in sorted(total_per_class.keys()):
        letter = label_encoder.inverse_transform([idx])[0]
        acc = 100.0 * correct_per_class[idx] / total_per_class[idx]
        marker = " <- NONSENSE" if letter == "Nonsense" else ""
        if acc < 98:
            weak_classes.append((letter, acc))
        print(
            f"  {letter:>10}: {acc:5.1f}% ({correct_per_class[idx]}/{total_per_class[idx]}){marker}"
        )

    if weak_classes:
        summary = ", ".join(f"{c}({a:.1f}%)" for c, a in weak_classes)
        print(f"\nWeak classes (<98%): {summary}")


def main():
    print("=" * 60)
    print("SIGN LANGUAGE CLASSIFIER - TRAINING")
    print("=" * 60)

    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # ---- Step 1: gather raw landmarks from the recorded photos ----
    print("\n" + "=" * 60)
    print("STEP 1: DATA GATHERING")
    print("=" * 60)
    samples = gather_data(config.PHOTOS_DIR, config.LETTER_FOLDERS)

    new_X = np.asarray([coords for _, coords in samples], dtype=np.float32)
    new_y = np.asarray([label for label, _ in samples])

    # ---- Step 2: split BEFORE augmenting ----
    print("\n" + "=" * 60)
    print("STEP 2: SPLITTING (before augmentation)")
    print("=" * 60)

    train_parts, val_parts, test_parts = [], [], []

    if len(new_X):
        counts = pd.Series(new_y).value_counts()
        too_small = counts[counts < 3]
        if len(too_small):
            raise ValueError(
                "These recorded classes have fewer than 3 samples and cannot be "
                f"split three ways: {dict(too_small)}"
            )
        new_train, new_val, new_test = stratified_three_way_split(
            new_X, new_y, config.TEST_SIZE, config.VAL_SIZE
        )
        print(
            f"Recorded photos: train {len(new_train[0])} | "
            f"val {len(new_val[0])} | test {len(new_test[0])}"
        )
    else:
        print("No photos gathered; using the CSV dataset only.")
        empty = (np.empty((0, 63), dtype=np.float32), np.empty(0, dtype=object))
        new_train = new_val = new_test = empty

    # ---- Step 3: augment the training fold only ----
    print("\n" + "=" * 60)
    print(f"STEP 3: AUGMENTATION (train fold only, x{config.AUGMENT_MULTIPLIER})")
    print("=" * 60)
    before = len(new_train[0])
    aug_X, aug_y = augment_training_fold(*new_train, config.AUGMENT_MULTIPLIER)
    print(f"Train fold: {before} -> {len(aug_X)} samples")

    # ---- Step 4: normalize every fold ----
    print("\n" + "=" * 60)
    print("STEP 4: NORMALIZATION")
    print("=" * 60)
    train_parts.append((normalize_matrix(aug_X), aug_y))
    val_parts.append((normalize_matrix(new_val[0]), new_val[1]))
    test_parts.append((normalize_matrix(new_test[0]), new_test[1]))
    print("Normalized all folds")

    # ---- Step 5: split the pre-built CSV separately and merge fold-wise ----
    print("\n" + "=" * 60)
    print("STEP 5: MERGING THE PRE-BUILT DATASET")
    print("=" * 60)

    if config.INPUT_CSV.exists():
        csv_X, csv_y = load_csv_source(config.INPUT_CSV, config.LETTERS_TO_REPLACE)
        csv_train, csv_val, csv_test = stratified_three_way_split(
            csv_X, csv_y, config.TEST_SIZE, config.VAL_SIZE
        )

        if AUGMENT_CSV_FOLD:
            before = len(csv_train[0])
            csv_train = augment_training_fold(*csv_train, config.AUGMENT_MULTIPLIER)
            print(f"CSV train fold augmented: {before} -> {len(csv_train[0])}")

        # Already normalized upstream; normalize_matrix is a no-op on such rows.
        train_parts.append((normalize_matrix(csv_train[0]), csv_train[1]))
        val_parts.append((normalize_matrix(csv_val[0]), csv_val[1]))
        test_parts.append((normalize_matrix(csv_test[0]), csv_test[1]))
        print(f"CSV: train {len(csv_train[0])} | val {len(csv_val[0])} | test {len(csv_test[0])}")
    else:
        print(f"Original CSV not found: {config.INPUT_CSV}")
        print("Training on the gathered photos only.")

    def concat(parts):
        return (
            np.vstack([X for X, _ in parts]),
            np.concatenate([y for _, y in parts]),
        )

    X_train, y_train = concat(train_parts)
    X_val, y_val = concat(val_parts)
    X_test, y_test = concat(test_parts)

    if len(X_train) == 0:
        print("No training data available. Aborting.")
        return

    print(f"\nFinal: train {len(X_train)} | val {len(X_val)} | test {len(X_test)}")

    # Belt and braces: the split happens before augmentation, but verify no
    # identical row survives in a held-out fold before spending a training run.
    assert_no_duplicate_rows_across_splits(X_train, X_val, "val")
    assert_no_duplicate_rows_across_splits(X_train, X_test, "test")
    print("Leakage check passed: no held-out row duplicates a training row")

    # ---- Step 6: encode labels and build loaders ----
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train, y_val, y_test]))
    num_classes = len(label_encoder.classes_)

    print(f"\nClasses ({num_classes}):")
    for idx, label in enumerate(label_encoder.classes_):
        marker = " <- NONSENSE" if label == "Nonsense" else ""
        print(f"  {idx:2d}: {label}{marker}")

    train_loader, val_loader, test_loader = build_dataloaders(
        [(X_train, y_train), (X_val, y_val), (X_test, y_test)],
        label_encoder,
        config.BATCH_SIZE,
    )

    # ---- Step 7: train, selecting on validation ----
    print("\n" + "=" * 60)
    print("STEP 7: TRAINING")
    print("=" * 60)

    model = ResidualMLP(
        input_dim=config.INPUT_DIM,
        num_classes=num_classes,
        hidden_dim=config.HIDDEN_DIM,
        num_blocks=config.NUM_BLOCKS,
        dropout=config.DROPOUT,
    ).to(device)
    print(f"Model: ResidualMLP ({sum(p.numel() for p in model.parameters()):,} parameters)")

    criterion = FocalLoss(gamma=2)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    print("\nTraining started...")
    print("-" * 60)

    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "ResidualMLP",
                    "input_dim": config.INPUT_DIM,
                    "num_classes": num_classes,
                    "val_acc": val_acc,  # selection metric, NOT a test score
                    "epoch": epoch,
                },
                MODEL_OUTPUT,
            )

            print(
                f"Epoch {epoch + 1:3d}/{config.EPOCHS} | "
                f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% NEW BEST"
            )
        else:
            patience_counter += 1
            if epoch % 10 == 0 or patience_counter == config.PATIENCE:
                print(
                    f"Epoch {epoch + 1:3d}/{config.EPOCHS} | "
                    f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%"
                )

        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\nTraining completed in {time.time() - start_time:.1f}s")

    # ---- Step 8: evaluate once on the untouched test fold ----
    print("\n" + "=" * 60)
    print("STEP 8: FINAL EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(MODEL_OUTPUT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    print(f"\nValidation accuracy (model selection): {best_val_acc:.2f}%")
    print(f"Test accuracy (held out, reported once): {test_acc:.2f}%")

    print_per_class_accuracy(test_targets, test_preds, label_encoder)

    # Record the test score alongside the weights so nothing downstream has to guess.
    checkpoint["test_acc"] = test_acc
    torch.save(checkpoint, MODEL_OUTPUT)

    with open(ENCODER_OUTPUT, "wb") as f:
        pickle.dump(label_encoder, f)
    np.save(CLASSES_OUTPUT, label_encoder.classes_)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model:   {MODEL_OUTPUT}")
    print(f"Encoder: {ENCODER_OUTPUT}")
    print(f"Classes: {CLASSES_OUTPUT}")
    print(f"Val accuracy:  {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
