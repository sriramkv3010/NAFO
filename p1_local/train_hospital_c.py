"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — Hospital C Local Training
Dataset : ChestMNIST (MedMNIST) — Cardiomegaly detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pipeline:
    chestmnist.npz
        → ChestXRayDataset     (1, 28, 28) normalised images
        → ImageEncoder         2D CNN → 64-dim latent
        → SharedClassifierHead 64 → 32 → 1
        → BCEWithLogitsLoss    (plain — no pos_weight)

Key design decisions:
    Binary task    : Cardiomegaly (1) vs Normal (0)
    Label column   : index 2 of 14-label array
    Imbalance fix  : WeightedRandomSampler ONLY
                     (no pos_weight — lesson from Hospital B)
    Loss           : BCEWithLogitsLoss plain
    Metric         : AUC-ROC (better than accuracy for imbalanced data)

Run from project root:
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python phase1_local/train_hospital_c.py

Expected results (20 epochs):
    Val Accuracy : 70% – 80%
    Val AUC-ROC  : 0.75 – 0.85
    Cardiomegaly recall > 60%  ← key clinical metric
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.utils.device import DEVICE
from src.utils.logger import RoundLogger
from src.datasets.hospital_c import load_hospital_c
from src.encoders.image_encoder import ImageEncoder
from src.models.shared_head import SharedClassifierHead

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_PATH = "data/hospital_c/chestmnist.npz"
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
CHECKPOINT = "logs/hospital_c_best.pt"
LOG_PATH = "logs/training_history.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SANITY CHECKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_sanity_checks(train_loader, encoder, head):
    print("\n── Sanity Checks ─────────────────────────────────")

    X_s, y_s = next(iter(train_loader))

    # Shape: (batch, 1, 28, 28)
    assert X_s.ndim == 4, f"Expected 4D input (batch,1,28,28), got {X_s.ndim}D"
    assert X_s.shape[1] == 1, f"Expected 1 channel (grayscale), got {X_s.shape[1]}"
    assert (
        X_s.shape[2] == 28 and X_s.shape[3] == 28
    ), f"Expected 28x28 images, got {X_s.shape[2]}x{X_s.shape[3]}"
    assert X_s.dtype == torch.float32
    assert (
        y_s.dtype == torch.float32
    ), f"Labels must be float32 for BCEWithLogitsLoss, got {y_s.dtype}"
    print(f"  ✓ Feature shape  : {X_s.shape}  (batch, channel, H, W)")
    print(f"  ✓ Label shape    : {y_s.shape}")
    print(f"  ✓ Dtypes         : {X_s.dtype} / {y_s.dtype}")

    # Pixel range check
    assert (
        X_s.min() >= 0.0 and X_s.max() <= 1.0
    ), f"Pixels must be in [0,1], got [{X_s.min():.3f},{X_s.max():.3f}]"
    print(f"  ✓ Pixel range    : [{X_s.min():.3f}, {X_s.max():.3f}]")

    # Binary labels
    unique = set(y_s.numpy().astype(int).tolist())
    assert unique.issubset({0, 1}), f"Labels must be binary {{0,1}}, found {unique}"
    print(f"  ✓ Label values   : {unique}")

    # Sampler check — should see mix of both classes in batch
    n_pos_in_batch = int(y_s.sum())
    print(
        f"  ✓ Positive in batch : {n_pos_in_batch}/{len(y_s)} "
        f"(sampler working if ~{len(y_s)//2})"
    )

    # No NaNs
    assert not torch.isnan(X_s).any()
    print(f"  ✓ No NaNs")

    # Full pipeline shape
    with torch.no_grad():
        dummy = torch.randn(4, 1, 28, 28).to(DEVICE)
        latent = encoder(dummy)
        logit = head(latent)
    assert latent.shape == (4, 64), f"Encoder: {latent.shape}"
    assert logit.shape == (4, 1), f"Head: {logit.shape}"
    print(f"  ✓ Encoder output : {latent.shape}")
    print(f"  ✓ Head output    : {logit.shape}")
    print(f"  ✓ 64-dim contract: VERIFIED")
    print("──────────────────────────────────────────────────\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def train_one_epoch(encoder, head, loader, optimizer, criterion) -> float:
    """One full pass over train_loader. Returns mean train loss."""
    encoder.train()
    head.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)  # (batch, 1, 28, 28)
        y_batch = y_batch.to(DEVICE)  # (batch,) float32

        optimizer.zero_grad()
        latent = encoder(X_batch)  # (batch, 64)
        logit = head(latent)  # (batch, 1)

        # BCEWithLogitsLoss needs both tensors shape (batch,)
        loss = criterion(logit.squeeze(1), y_batch)
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
    Evaluation on unbalanced val set (real-world distribution).
    Returns: (val_loss, accuracy, auc, all_probs, all_labels)
    """
    encoder.eval()
    head.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        latent = encoder(X_batch)
        logit = head(latent)

        loss = criterion(logit.squeeze(1), y_batch)
        total_loss += loss.item()

        prob = torch.sigmoid(logit.squeeze(1))
        all_probs.extend(prob.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= 0.5).astype(int)
    accuracy = float((preds == all_labels).mean())

    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = 0.0

    return total_loss / len(loader), accuracy, auc, all_probs, all_labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("  PHASE 1 — Hospital C Local Training")
    print("  Dataset : ChestMNIST — Cardiomegaly (col 2)")
    print(f"  Device  : {DEVICE}")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print()
    train_loader, val_loader, meta = load_hospital_c(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
    )

    # ── Step 2: Build models ───────────────────────────────────────────
    # SharedClassifierHead is reused from Hospital A (binary, 64→32→1)
    encoder = ImageEncoder().to(DEVICE)
    head = SharedClassifierHead().to(DEVICE)

    # ── Step 3: Sanity checks ──────────────────────────────────────────
    run_sanity_checks(train_loader, encoder, head)

    # ── Step 4: Loss ───────────────────────────────────────────────────
    # Plain BCEWithLogitsLoss — no pos_weight.
    # WeightedRandomSampler already handles class imbalance.
    # Using both is double correction → destroys majority class recall.
    criterion = nn.BCEWithLogitsLoss()
    print("[Loss]    BCEWithLogitsLoss (plain — no pos_weight)")
    print("[Sampler] WeightedRandomSampler handles imbalance\n")

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
        f"{'Acc':<10} {'AUC-ROC':<9}"
    )
    print("-" * 56)

    best_val_loss = float("inf")
    best_state = None
    best_probs = None
    best_labels = None

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(encoder, head, train_loader, optimizer, criterion)
        val_loss, acc, auc, probs, labels = evaluate(
            encoder, head, val_loader, criterion
        )
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_probs = probs.copy()
            best_labels = labels.copy()
            best_state = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": acc,
                "val_auc": auc,
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
            }

        logger.log(
            phase="phase1",
            round=epoch,
            hospital="hospital_c",
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=acc,
            val_auc=auc,
            n_samples=meta["n_train"],
        )

        marker = " ◀ best" if val_loss == best_val_loss else ""
        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"Train: {train_loss:.4f}   "
            f"Val: {val_loss:.4f}   "
            f"Acc: {acc:.4f}   "
            f"AUC: {auc:.4f}"
            f"{marker}"
        )

    # ── Step 8: Save checkpoint ────────────────────────────────────────
    torch.save(best_state, CHECKPOINT)

    # ── Step 9: Final report ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Best epoch     : {best_state['epoch']}")
    print(f"  Best val loss  : {best_state['val_loss']:.4f}")
    print(f"  Best val acc   : {best_state['val_acc']:.4f}")
    print(f"  Best val AUC   : {best_state['val_auc']:.4f}")
    print(f"  Checkpoint     : {CHECKPOINT}")

    best_preds = (best_probs >= 0.5).astype(int)
    print(f"\n[Classification Report — Val Set]")
    print(
        classification_report(
            best_labels.astype(int),
            best_preds,
            target_names=["Normal (0)", "Cardiomegaly (1)"],
            digits=4,
            zero_division=0,
        )
    )

    # ── Key metric callout ─────────────────────────────────────────────
    print(f"[Key Metrics]")
    card_mask = best_labels == 1
    card_recall = (
        float((best_preds[card_mask] == 1).mean()) if card_mask.sum() > 0 else 0.0
    )
    norm_mask = best_labels == 0
    norm_recall = (
        float((best_preds[norm_mask] == 0).mean()) if norm_mask.sum() > 0 else 0.0
    )

    print(
        f"  Normal recall       : {norm_recall:.4f}  "
        f"{'✓ GOOD' if norm_recall > 0.70 else '✗ LOW'}"
    )
    print(
        f"  Cardiomegaly recall : {card_recall:.4f}  "
        f"{'✓ GOOD' if card_recall > 0.60 else '✗ LOW'}"
    )
    auc_status = "✓ GOOD" if best_state["val_auc"] > 0.75 else "✗ LOW"
    print(f"  AUC-ROC             : {best_state['val_auc']:.4f}  {auc_status}")

    # Confusion matrix
    print(f"\n[Confusion Matrix]")
    cm = confusion_matrix(best_labels.astype(int), best_preds)
    print(f"                  Pred Normal  Pred Cardio")
    print(f"  True Normal   : {cm[0,0]:10d}  {cm[0,1]:11d}")
    print(f"  True Cardio   : {cm[1,0]:10d}  {cm[1,1]:11d}")

    # ── Step 10: 64-dim contract check ────────────────────────────────
    print(f"\n[Phase 1 Contract Check]")
    with torch.no_grad():
        X_check, _ = next(iter(val_loader))
        latent_out = encoder(X_check.to(DEVICE))
    assert (
        latent_out.shape[1] == 64
    ), f"FAILED — encoder outputs {latent_out.shape[1]} dims"
    print(f"  ✓ Encoder 64-dim output : CONFIRMED")
    print(f"  ✓ Hospital C Phase 1    : COMPLETE")
    print(f"  → Next: run Hospital D")
    print("=" * 60)


if __name__ == "__main__":
    main()
