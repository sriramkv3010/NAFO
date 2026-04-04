"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — Hospital A Local Training
Dataset : UCI Heart Disease (processed.cleveland.data)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pipeline:
    processed.cleveland.data
        → HeartDiseaseDataset  (hospital_a.py)
        → TabularEncoder       13 → 128 → 64
        → SharedClassifierHead 64 → 32 → 1
        → BCEWithLogitsLoss
        → Adam + ReduceLROnPlateau

This script trains Hospital A in complete isolation.
No federation. No 5G. Pure local training.
The val accuracy and AUC numbers printed here become
the Phase 2 baseline — if FedAvg matches them,
federation is working correctly.

Run from project root:
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python phase1_local/train_hospital_a.py

Expected results (30 epochs):
    Val Accuracy : 78% – 85%
    Val AUC-ROC  : 0.82 – 0.90
    Best checkpoint saved to: logs/hospital_a_best.pt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys

# ── Project root on path so src.* imports resolve ─────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

from src.utils.device import DEVICE
from src.utils.logger import RoundLogger
from src.datasets.hospital_a import load_hospital_a
from src.encoders.tabular_encoder import TabularEncoder
from src.models.shared_head import SharedClassifierHead

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_PATH = "data/hospital_a/processed.cleveland.data"
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3  # Adam learning rate
WEIGHT_DECAY = 1e-4  # L2 regularisation — important for 297-sample datasets
CHECKPOINT = "logs/hospital_a_best.pt"
LOG_PATH = "logs/training_history.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SANITY CHECKS
# Run before training. Better to fail here than mid-epoch.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_sanity_checks(
    train_loader,
    encoder: nn.Module,
    head: nn.Module,
) -> None:
    print("\n── Sanity Checks ─────────────────────────────────")

    # Check 1: batch shapes and dtypes
    X_s, y_s = next(iter(train_loader))
    assert X_s.shape[1] == 13, f"Expected 13 features, got {X_s.shape[1]}"
    assert (
        X_s.dtype == torch.float32
    ), f"Features must be float32 for MPS, got {X_s.dtype}"
    assert y_s.dtype == torch.float32, f"Labels must be float32, got {y_s.dtype}"
    print(f"  ✓ Feature batch shape : {X_s.shape}")
    print(f"  ✓ Label batch shape   : {y_s.shape}")
    print(f"  ✓ Dtypes              : {X_s.dtype} / {y_s.dtype}")

    # Check 2: binary labels only
    unique = set(y_s.numpy().astype(int).tolist())
    assert unique.issubset({0, 1}), f"Labels must be binary {{0,1}}, found {unique}"
    print(f"  ✓ Label values        : {unique}")

    # Check 3: no NaNs in batch
    assert not torch.isnan(X_s).any(), "NaN detected in feature batch"
    assert not torch.isnan(y_s).any(), "NaN detected in label batch"
    print(f"  ✓ No NaNs in batch")

    # Check 4: full encoder → head pipeline shapes
    with torch.no_grad():
        dummy = torch.randn(4, 13).to(DEVICE)
        latent = encoder(dummy)  # triggers 64-dim assertion in base
        logit = head(latent)
    assert latent.shape == (4, 64), f"Encoder: expected (4,64), got {latent.shape}"
    assert logit.shape == (4, 1), f"Head: expected (4,1), got {logit.shape}"
    print(f"  ✓ Encoder output      : {latent.shape}")
    print(f"  ✓ Head output         : {logit.shape}")
    print(f"  ✓ 64-dim contract     : VERIFIED")
    print("──────────────────────────────────────────────────\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def train_one_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """
    One full pass over train_loader.
    Encoder and head train jointly in Phase 1.
    In Phase 2, the head gets its own DP-SGD optimizer.

    Returns: mean train loss for the epoch.
    """
    encoder.train()
    head.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)  # (batch, 13)
        y_batch = y_batch.to(DEVICE)  # (batch,)

        optimizer.zero_grad()

        # Forward: features → latent → logit
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
def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    loader,
    criterion: nn.Module,
) -> tuple:
    """
    Full validation pass.
    Returns: (val_loss, accuracy, auc_roc)
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

        # sigmoid converts raw logit → probability [0,1]
        prob = torch.sigmoid(logit.squeeze(1))
        all_probs.extend(prob.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Accuracy at threshold 0.5
    preds = (all_probs >= 0.5).astype(int)
    accuracy = float((preds == all_labels).mean())

    # AUC-ROC — threshold-independent, better metric for cardiac data
    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = 0.0

    return total_loss / len(loader), accuracy, auc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    os.makedirs("logs", exist_ok=True)

    print("=" * 58)
    print("  PHASE 1 — Hospital A Local Training")
    print("  Dataset : UCI Heart Disease (Cleveland)")
    print(f"  Device  : {DEVICE}")
    print("=" * 58)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print()
    train_loader, val_loader, meta = load_hospital_a(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
    )

    # ── Step 2: Build models ───────────────────────────────────────────
    encoder = TabularEncoder(input_dim=13).to(DEVICE)
    head = SharedClassifierHead().to(DEVICE)

    # ── Step 3: Sanity checks ──────────────────────────────────────────
    run_sanity_checks(train_loader, encoder, head)

    # ── Step 4: Loss ───────────────────────────────────────────────────
    # pos_weight = n_healthy / n_disease = 160/137 ≈ 1.168
    # Upweights the disease class in the loss so model doesn't ignore it
    pos_weight = torch.tensor([meta["pos_weight"]], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[Loss] BCEWithLogitsLoss | pos_weight = {meta['pos_weight']:.4f}\n")

    # ── Step 5: Optimiser ─────────────────────────────────────────────
    # Phase 1: single Adam for encoder + head together
    # Phase 2: head gets its own DP-SGD optimizer via Opacus
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Halves LR after 5 epochs with no val_loss improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # ── Step 6: Logger ─────────────────────────────────────────────────
    logger = RoundLogger(LOG_PATH)

    # ── Step 7: Training loop ──────────────────────────────────────────
    print(
        f"{'Epoch':<9} {'Train Loss':<13} {'Val Loss':<12} "
        f"{'Accuracy':<11} {'AUC-ROC':<9}"
    )
    print("-" * 56)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(encoder, head, train_loader, optimizer, criterion)
        val_loss, acc, auc = evaluate(encoder, head, val_loader, criterion)
        scheduler.step(val_loss)

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": acc,
                "val_auc": auc,
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
            }

        # Log to JSON
        logger.log(
            phase="phase1",
            round=epoch,
            hospital="hospital_a",
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

    # ── Step 8: Save best checkpoint ──────────────────────────────────
    torch.save(best_state, CHECKPOINT)

    # ── Step 9: Final report ───────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  Best epoch     : {best_state['epoch']}")
    print(f"  Best val loss  : {best_state['val_loss']:.4f}")
    print(f"  Best val acc   : {best_state['val_acc']:.4f}")
    print(f"  Best val AUC   : {best_state['val_auc']:.4f}")
    print(f"  Checkpoint     : {CHECKPOINT}")
    print(f"  Log            : {LOG_PATH}")

    # Load best weights for classification report
    encoder.load_state_dict(best_state["encoder"])
    head.load_state_dict(best_state["head"])

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            logit = head(encoder(X_b.to(DEVICE)))
            pred = (torch.sigmoid(logit.squeeze(1)) >= 0.5).int()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_b.numpy().astype(int))

    print(f"\n[Classification Report — Val Set]")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["Healthy (0)", "Disease (1)"],
            digits=4,
        )
    )

    # ── Step 10: Final 64-dim contract check ──────────────────────────
    print("[Phase 1 Contract Check]")
    with torch.no_grad():
        X_check, _ = next(iter(val_loader))
        latent_out = encoder(X_check.to(DEVICE))
    assert (
        latent_out.shape[1] == 64
    ), f"FAILED — encoder outputs {latent_out.shape[1]} dims, expected 64"
    print(f"  ✓ Encoder 64-dim output : CONFIRMED")
    print(f"  ✓ Hospital A Phase 1    : COMPLETE")
    print(f"  → Next: run Hospital C")
    print("=" * 58)


if __name__ == "__main__":
    main()
