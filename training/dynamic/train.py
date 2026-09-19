"""
Training script for LSTM Dynamic Sign Model (J/Z)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from model import DynamicSignLSTM
import config


def load_data():
    """
    Load the dataset and build train/val/test loaders.

    The validation split drives early stopping and checkpoint selection; the
    test split is touched exactly once, at the end. Selecting on the test set
    and then reporting that same number, as this script previously did, gives
    an optimistically biased accuracy.
    """
    print(f"Loading data from {config.DATA_PATH}")

    data = np.load(config.DATA_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"]

    # Convert labels to indices
    y_idx = np.array([config.CLASS_TO_IDX[label] for label in y])

    print(f"   X shape: {X.shape}")
    labels, counts = np.unique(y, return_counts=True)
    print(f"   Classes: {dict(zip(labels, counts, strict=True))}")

    # Hold out the test set first.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y_idx,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=y_idx,
    )

    # Carve the validation set out of what remains, re-scaling the fraction so
    # VAL_SPLIT stays a fraction of the ORIGINAL dataset.
    val_adjusted = config.VAL_SPLIT / (1 - config.TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_adjusted,
        random_state=config.RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    def make_loader(features, labels, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(labels.astype(np.int64)),
        )
        return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle)

    return (
        make_loader(X_train, y_train, True),
        make_loader(X_val, y_val, False),
        make_loader(X_test, y_test, False),
    )


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


def plot_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss History")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy History")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=20)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def train():
    """Main training function."""
    print("=" * 60)
    print("TRAINING DYNAMIC SIGN MODEL (J/Z)")
    print("=" * 60)

    device = config.DEVICE
    print(f"\nDevice: {device}")

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = load_data()

    # Create model
    print("\nCreating LSTM model...")
    model = DynamicSignLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
    )

    model = model.to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    print(f"\nTraining for {config.EPOCHS} epochs...")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # Selection and early stopping use the VALIDATION split only.
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"   Epoch {epoch + 1:3d}/{config.EPOCHS} | "
            f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
            f"Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "config": {
                        "model_type": "lstm",
                        "input_size": config.INPUT_SIZE,
                        "hidden_size": config.HIDDEN_SIZE,
                        "num_layers": config.NUM_LAYERS,
                        "num_classes": config.NUM_CLASSES,
                    },
                },
                config.CHECKPOINT_PATH,
            )
            patience_counter = 0
            print(f"   Saved best model (acc: {val_acc:.3f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    # Load the checkpoint selected on validation, then score the held-out test set once.
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)

    print(f"\nValidation accuracy (model selection): {best_val_acc:.4f}")
    print(f"Test accuracy (held out, reported once): {test_acc:.4f}")
    print("\nClassification Report (test split):")
    print(classification_report(y_true, y_pred, target_names=config.CLASSES))

    # Record the test score alongside the weights.
    checkpoint["test_acc"] = test_acc
    torch.save(checkpoint, config.CHECKPOINT_PATH)

    # Save plots
    plot_history(history, config.MODEL_DIR / "training_history.png")
    plot_confusion_matrix(y_true, y_pred, config.MODEL_DIR / "confusion_matrix.png")

    # Save classes next to the checkpoint, where inference looks for them.
    np.save(config.CLASSES_PATH, np.array(config.CLASSES))

    print("\nTraining complete!")
    print(f"   Model saved:   {config.CHECKPOINT_PATH}")
    print(f"   Classes saved: {config.CLASSES_PATH}")


if __name__ == "__main__":
    train()
