"""
Training Script for Sign Language Classifier
=============================================
Trains ResidualMLP model with early stopping and saves best checkpoint.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import time

from models import ResidualMLP
from data_gathering import gather_data
from data_augmentation import augment_by_class
from dataset_creation import (
    normalize_samples, merge_with_original, prepare_data_splits,
    create_dataloaders, save_label_encoder, samples_to_dataframe
)

# ============ CONFIGURATION ============
DATA_DIR = Path(".")
PHOTOS_DIR = Path("letters")
INPUT_CSV = DATA_DIR / "ngt_data.csv"
OUTPUT_CSV = DATA_DIR / "ngt_final.csv"
MODEL_OUTPUT = DATA_DIR / "best_model.pth"
ENCODER_OUTPUT = DATA_DIR / "label_encoder.pkl"

LETTER_FOLDERS = {
    'D': 'D_letter', 'F': 'F_letter', 'G': 'G_letter',
    'R': 'R_letter', 'S': 'S_letter', 'V': 'V_letter',
    'X': 'X_letter', 'Nonsense': 'Nonsense'
}
LETTERS_TO_REPLACE = list(LETTER_FOLDERS.keys())
AUGMENT_MULTIPLIER = 10

BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate model."""
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
    
    return total_loss / len(loader), 100. * correct / total, all_preds, all_targets


def print_per_class_accuracy(y_true, y_pred, le):
    """Print accuracy for each class."""
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    
    for true, pred in zip(y_true, y_pred):
        total_per_class[true] += 1
        if true == pred:
            correct_per_class[true] += 1
    
    print("\nPer-class accuracy:")
    print("-" * 50)
    
    weak_classes = []
    for idx in sorted(total_per_class.keys()):
        letter = le.inverse_transform([idx])[0]
        acc = 100. * correct_per_class[idx] / total_per_class[idx]
        marker = " ← NONSENSE" if letter == "Nonsense" else ""
        if acc < 98:
            marker += ""
            weak_classes.append((letter, acc))
        print(f"  {letter:>10}: {acc:5.1f}% ({correct_per_class[idx]}/{total_per_class[idx]}){marker}")
    
    if weak_classes:
        print(f"\nWeak classes (<98%): {', '.join([f'{c}({a:.1f}%)' for c, a in weak_classes])}")


def main():
    print("=" * 60)
    print("SIGN LANGUAGE CLASSIFIER - TRAINING")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    # Step 1: Gather data
    print("\n" + "=" * 60)
    print("STEP 1: DATA GATHERING")
    print("=" * 60)
    samples = gather_data(PHOTOS_DIR, LETTER_FOLDERS)
    
    if not samples:
        print("No data gathered! Check your photo folders.")
        return
    
    # Step 2: Augment
    print("\n" + "=" * 60)
    print(f"STEP 2: DATA AUGMENTATION (x{AUGMENT_MULTIPLIER})")
    print("=" * 60)
    augmented = augment_by_class(samples, AUGMENT_MULTIPLIER)
    
    # Step 3: Normalize
    print("\n" + "=" * 60)
    print("STEP 3: NORMALIZATION")
    print("=" * 60)
    normalized = normalize_samples(augmented)
    print(f"Normalized {len(normalized)} samples")
    
    # Step 4: Merge with original dataset
    print("\n" + "=" * 60)
    print("STEP 4: DATASET CREATION")
    print("=" * 60)
    
    if INPUT_CSV.exists():
        df = merge_with_original(normalized, INPUT_CSV, LETTERS_TO_REPLACE)
    else:
        print(f"Original CSV not found: {INPUT_CSV}")
        print("Creating dataset from gathered data only...")
        df = samples_to_dataframe(normalized)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    
    # Step 5: Prepare splits
    print("\n" + "=" * 60)
    print("STEP 5: PREPARING DATA SPLITS")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test, le = prepare_data_splits(df)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE
    )
    
    # Step 6: Train
    print("\n" + "=" * 60)
    print("STEP 6: TRAINING")
    print("=" * 60)
    
    num_classes = len(le.classes_)
    model = ResidualMLP(input_dim=63, num_classes=num_classes).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResidualMLP ({params:,} parameters)")
    
    criterion = FocalLoss(gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_acc = 0
    patience_counter = 0
    start_time = time.time()
    
    print("\nTraining started...")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': 'ResidualMLP',
                'input_dim': 63,
                'num_classes': num_classes,
                'test_acc': val_acc
            }, MODEL_OUTPUT)
            
            save_label_encoder(le, ENCODER_OUTPUT)
            
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% NEW BEST")
        else:
            patience_counter += 1
            if epoch % 10 == 0 or patience_counter == PATIENCE:
                print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s")
    
    # Step 7: Final evaluation
    print("\n" + "=" * 60)
    print("STEP 7: FINAL EVALUATION")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(MODEL_OUTPUT, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_acc, test_preds, test_targets = validate(model, test_loader, criterion, device)
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    print_per_class_accuracy(test_targets, test_preds, le)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved: {MODEL_OUTPUT}")
    print(f"Encoder saved: {ENCODER_OUTPUT}")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Classes: {num_classes}")


if __name__ == "__main__":
    main()