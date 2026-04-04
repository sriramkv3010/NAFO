"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — Hospital B Local Training (FIXED)
Dataset : MIT-BIH Arrhythmia Database (48 records)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fix applied vs previous version:
    PROBLEM: WeightedRandomSampler + weighted CrossEntropyLoss
             were both correcting for class imbalance simultaneously.
             This over-corrected so aggressively that the model
             learned N (Normal) was worthless → 0% N recall.

    FIX:     Use WeightedRandomSampler ONLY.
             CrossEntropyLoss is now plain (no class weights).
             The sampler makes training batches balanced.
             The loss then treats all classes equally per batch.
             Rule: use sampler OR weighted loss — never both.

Run from project root:
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python phase1_local/train_hospital_b.py

Expected results (20 epochs):
    Val Accuracy (plain)    : 75% – 88%
    Balanced Accuracy       : 65% – 80%
    V (Ventricular) recall  : > 75%
    N (Normal) recall       : > 80%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix,
)

from src.utils.device import DEVICE
from src.utils.logger import RoundLogger
from src.datasets.hospital_b import load_hospital_b, CLASS_NAMES
from src.encoders.signal_encoder import SignalEncoder

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_DIR = "data/hospital_b"
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
CHECKPOINT = "logs/hospital_b_best.pt"
LOG_PATH = "logs/training_history.json"
N_CLASSES = 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5-CLASS LOCAL HEAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LocalFiveClassHead(nn.Module):
    """
    5-class arrhythmia head for Hospital B local training ONLY.
    NOT used in Phase 2 federation.
    Phase 2 switches to binary SharedClassifierHead:
        0 = N (normal), 1 = any arrhythmia (S/V/F/Q)
    """

    def __init__(self, latent_dim: int = 64, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SANITY CHECKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_sanity_checks(train_loader, encoder, head):
    print("\n── Sanity Checks ─────────────────────────────────")
    X_s, y_s = next(iter(train_loader))

    assert X_s.ndim == 3
    assert X_s.shape[1] == 1
    assert X_s.shape[2] == 187
    assert X_s.dtype == torch.float32
    assert (
        y_s.dtype == torch.long
    ), f"Labels must be torch.long for CrossEntropyLoss, got {y_s.dtype}"
    print(f"  ✓ Feature shape  : {X_s.shape}")
    print(f"  ✓ Label shape    : {y_s.shape}")
    print(f"  ✓ Dtypes         : {X_s.dtype} / {y_s.dtype}")
    assert y_s.min() >= 0 and y_s.max() <= 4
    print(f"  ✓ Label range    : [{int(y_s.min())}, {int(y_s.max())}]")

    unique_in_batch = len(torch.unique(y_s))
    print(
        f"  ✓ Classes in batch : {unique_in_batch} " f"(WeightedSampler working if > 1)"
    )

    assert not torch.isnan(X_s).any()
    print(f"  ✓ No NaNs")

    with torch.no_grad():
        dummy = torch.randn(4, 1, 187).to(DEVICE)
        latent = encoder(dummy)
        logits = head(latent)
    assert latent.shape == (4, 64)
    assert logits.shape == (4, 5)
    print(f"  ✓ Encoder output : {latent.shape}")
    print(f"  ✓ Head output    : {logits.shape}")
    print(f"  ✓ 64-dim contract: VERIFIED")
    print("──────────────────────────────────────────────────\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_one_epoch(encoder, head, loader, optimizer, criterion) -> float:
    encoder.train()
    head.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        latent = encoder(X_batch)
        logits = head(latent)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def evaluate(encoder, head, loader, criterion) -> tuple:
    """
    Evaluate on UNBALANCED val set (real-world distribution).
    Sampler only applies to training — not evaluation.
    This gives honest clinical metrics.
    """
    encoder.eval()
    head.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        latent = encoder(X_batch)
        logits = head(latent)
        loss = criterion(logits, y_batch)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = float((all_preds == all_labels).mean())
    bal_accuracy = float(balanced_accuracy_score(all_labels, all_preds))
    return total_loss / len(loader), accuracy, bal_accuracy, all_preds, all_labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("  PHASE 1 — Hospital B Local Training (FIXED)")
    print("  Dataset : MIT-BIH Arrhythmia (48 records)")
    print(f"  Device  : {DEVICE}")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print()
    train_loader, val_loader, meta = load_hospital_b(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )

    # ── Step 2: Build models ───────────────────────────────────────────
    encoder = SignalEncoder().to(DEVICE)
    head = LocalFiveClassHead(n_classes=N_CLASSES).to(DEVICE)

    # ── Step 3: Sanity checks ──────────────────────────────────────────
    run_sanity_checks(train_loader, encoder, head)

    # ── Step 4: Loss ───────────────────────────────────────────────────
    # PLAIN CrossEntropyLoss — no class weights.
    # WeightedRandomSampler already handles class balance in training.
    # Using both together is double-correction → destroys N class recall.
    criterion = nn.CrossEntropyLoss()
    print("[Loss]    CrossEntropyLoss (plain — no class weights)")
    print("[Sampler] WeightedRandomSampler handles balance in training\n")

    # ── Step 5: Optimiser ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    # ── Step 6: Logger ─────────────────────────────────────────────────
    logger = RoundLogger(LOG_PATH)

    # ── Step 7: Training loop ──────────────────────────────────────────
    print(
        f"{'Epoch':<9} {'Train Loss':<13} {'Val Loss':<12} "
        f"{'Acc':<10} {'Bal Acc':<9}"
    )
    print("-" * 56)

    best_val_loss = float("inf")
    best_state = None
    best_preds = None
    best_labels = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(encoder, head, train_loader, optimizer, criterion)
        val_loss, acc, bal_acc, preds, labels = evaluate(
            encoder, head, val_loader, criterion
        )
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = preds.copy()
            best_labels = labels.copy()
            best_state = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": acc,
                "val_bal_acc": bal_acc,
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
            }

        logger.log(
            phase="phase1",
            round=epoch,
            hospital="hospital_b",
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=acc,
            n_samples=meta["n_train"],
        )

        marker = " ◀ best" if val_loss == best_val_loss else ""
        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"Train: {train_loss:.4f}   "
            f"Val: {val_loss:.4f}   "
            f"Acc: {acc:.4f}   "
            f"Bal: {bal_acc:.4f}"
            f"{marker}"
        )

    # ── Step 8: Save checkpoint ────────────────────────────────────────
    torch.save(best_state, CHECKPOINT)

    # ── Step 9: Final report ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Best epoch     : {best_state['epoch']}")
    print(f"  Best val loss  : {best_state['val_loss']:.4f}")
    print(f"  Best val acc   : {best_state['val_acc']:.4f}")
    print(f"  Best bal acc   : {best_state['val_bal_acc']:.4f}")
    print(f"  Checkpoint     : {CHECKPOINT}")

    present = sorted(np.unique(best_labels).tolist())
    print(f"\n[Classification Report — Val Set (DS2)]")
    print(
        classification_report(
            best_labels,
            best_preds,
            labels=present,
            target_names=[CLASS_NAMES[i] for i in present],
            digits=4,
            zero_division=0,
        )
    )

    # ── Key metric callout ─────────────────────────────────────────────
    print(f"[Key Metrics]")
    if 0 in present:
        n_mask = best_labels == 0
        n_recall = float((best_preds[n_mask] == 0).mean())
        print(
            f"  N (Normal) recall     : {n_recall:.4f}  "
            f"{'✓ GOOD' if n_recall > 0.80 else '✗ LOW — sampler issue'}"
        )
    if 2 in present:
        v_mask = best_labels == 2
        v_recall = float((best_preds[v_mask] == 2).mean())
        print(
            f"  V (Ventricular) recall: {v_recall:.4f}  "
            f"{'✓ GOOD' if v_recall > 0.75 else '✗ LOW — clinically risky'}"
        )

    # Confusion matrix
    print(f"\n[Confusion Matrix]")
    cm = confusion_matrix(best_labels, best_preds, labels=present)
    header = "        " + "  ".join(f"{CLASS_NAMES[i][:7]:>7}" for i in present)
    print(header)
    for i, row in zip(present, cm):
        print(f"  {CLASS_NAMES[i][:7]:>7}  " + "  ".join(f"{v:7d}" for v in row))

    # ── Step 10: 64-dim contract check ────────────────────────────────
    print(f"\n[Phase 1 Contract Check]")
    with torch.no_grad():
        X_check, _ = next(iter(val_loader))
        latent_out = encoder(X_check.to(DEVICE))
    assert latent_out.shape[1] == 64
    print(f"  ✓ Encoder 64-dim output : CONFIRMED")
    print(f"  ✓ Hospital B Phase 1    : COMPLETE")
    print(f"  → Next: run Hospital C")
    print("=" * 60)


if __name__ == "__main__":
    main()
