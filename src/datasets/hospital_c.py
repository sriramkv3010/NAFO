"""
src/datasets/hospital_c.py
--------------------------
Hospital C — ChestMNIST (MedMNIST)
File : data/hospital_c/chestmnist.npz

File structure (MedMNIST standard):
    train_images : (78468, 28, 28)  uint8  pixel values 0-255
    train_labels : (78468, 14)      uint8  multi-label binary flags
    val_images   : (11219, 28, 28)  uint8
    val_labels   : (11219, 14)      uint8
    test_images  : (22433, 28, 28)  uint8
    test_labels  : (22433, 14)      uint8

Label column selected: index 2 — Cardiomegaly
    Cardiomegaly = enlarged heart visible on chest X-ray
    This is the ONLY label in ChestMNIST that is unambiguously
    cardiovascular, which justifies our single-organ cardiac narrative.

    Why not other labels:
        Effusion (col 3)  — can have non-cardiac causes
        Infiltration (4)  — pulmonary, not cardiac
        All others        — clearly non-cardiac

Class imbalance:
    Cardiomegaly is rare in the dataset (~10% positive rate)
    WeightedRandomSampler handles this in training.
    Plain BCEWithLogitsLoss (no pos_weight) — avoids double correction.
    Lesson from Hospital B: sampler OR weighted loss, never both.

Preprocessing:
    1. Load .npz — images already split into train/val/test
    2. Extract cardiomegaly column (index 2)
    3. Normalise pixels: divide by 255.0 → [0.0, 1.0]
    4. Add channel dim: (N, 28, 28) → (N, 1, 28, 28)
    5. WeightedRandomSampler on training set

5G Slice (Phase 3): eMBB
    DICOM chest X-ray — large image payload, high bandwidth required.
    Latency-tolerant: radiologist review is not real-time.
    eMBB slice provides sustained high throughput for imaging data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Dict

# ── Constants ─────────────────────────────────────────────────────────────────
CARDIOMEGALY_COL = 2  # column index in the 14-label array
SLICE_TYPE = "eMBB"  # 5G slice for Phase 3
IMG_SIZE = 28  # ChestMNIST is 28x28 (MedMNIST standard)


class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for ChestMNIST cardiomegaly binary classification.

    Each sample:
        X : float32 Tensor (1, 28, 28)  — normalised chest X-ray
        y : float32 Tensor scalar       — 0=normal, 1=cardiomegaly

    float32 is mandatory — MPS backend does not support float64.
    Channel dim (1) is required by Conv2d in the ImageEncoder.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # images: (N, 28, 28) uint8 → float32 in [0, 1]
        # Add channel dim: (N, 28, 28) → (N, 1, 28, 28)
        imgs_float = images.astype(np.float32) / 255.0
        self.X = torch.tensor(
            imgs_float[:, np.newaxis, :, :],
            dtype=torch.float32,
        )  # (N, 1, 28, 28)
        self.y = torch.tensor(labels, dtype=torch.float32)  # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_hospital_c(
    data_path: str,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Full pipeline: chestmnist.npz → train/val DataLoaders.

    Uses the pre-defined MedMNIST train/val split.
    No random splitting — split is fixed by MedMNIST standard.

    Args:
        data_path  : path to chestmnist.npz
        batch_size : samples per batch

    Returns:
        train_loader : DataLoader with WeightedRandomSampler
        val_loader   : DataLoader (no shuffle)
        meta         : dict with stats for NAFO Phase 4
    """

    # ── 1. Load .npz ───────────────────────────────────────────────────────────
    print(f"[Hospital C] Loading {data_path} ...")
    data = np.load(data_path)

    # Verify expected keys exist
    required_keys = ["train_images", "train_labels", "val_images", "val_labels"]
    for key in required_keys:
        assert key in data, f"Key '{key}' not found in .npz. Found: {list(data.keys())}"

    train_images = data["train_images"]  # (78468, 28, 28)
    train_labels = data["train_labels"]  # (78468, 14)
    val_images = data["val_images"]  # (11219, 28, 28)
    val_labels = data["val_labels"]  # (11219, 14)

    print(
        f"[Hospital C] Train images: {train_images.shape} | "
        f"Val images: {val_images.shape}"
    )

    # ── 2. Extract cardiomegaly label (column 2) ───────────────────────────────
    # Labels are (N, 14) — we take one column for binary classification
    y_train = train_labels[:, CARDIOMEGALY_COL].astype(np.float32)  # (N,)
    y_val = val_labels[:, CARDIOMEGALY_COL].astype(np.float32)

    # ── 3. Class distribution ──────────────────────────────────────────────────
    n_train_pos = int(y_train.sum())
    n_train_neg = len(y_train) - n_train_pos
    n_val_pos = int(y_val.sum())
    n_val_neg = len(y_val) - n_val_pos

    print(
        f"[Hospital C] Train — Normal: {n_train_neg:,} | "
        f"Cardiomegaly: {n_train_pos:,} "
        f"({100*n_train_pos/len(y_train):.1f}% positive)"
    )
    print(
        f"[Hospital C] Val   — Normal: {n_val_neg:,} | "
        f"Cardiomegaly: {n_val_pos:,} "
        f"({100*n_val_pos/len(y_val):.1f}% positive)"
    )

    # ── 4. Build PyTorch Datasets ──────────────────────────────────────────────
    train_ds = ChestXRayDataset(train_images, y_train)
    val_ds = ChestXRayDataset(val_images, y_val)

    # ── 5. WeightedRandomSampler ───────────────────────────────────────────────
    # Cardiomegaly is ~10% of samples — sampler oversamples positive class.
    # Using plain CrossEntropyLoss / BCEWithLogitsLoss (no pos_weight).
    # Lesson from Hospital B: sampler OR weighted loss — never both.
    class_counts = np.bincount(y_train.astype(int))  # [n_neg, n_pos]
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = class_weights[y_train.astype(int)]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,  # mutually exclusive with shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # ── 6. Metadata ────────────────────────────────────────────────────────────
    meta = {
        "hospital": "hospital_c",
        "slice": SLICE_TYPE,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_train_pos": n_train_pos,
        "n_train_neg": n_train_neg,
        "n_val_pos": n_val_pos,
        "n_val_neg": n_val_neg,
        "input_shape": (1, IMG_SIZE, IMG_SIZE),
        "label_col": CARDIOMEGALY_COL,
        "label_name": "Cardiomegaly",
    }

    print(f"[Hospital C] Train: {meta['n_train']:,} | " f"Val: {meta['n_val']:,}")
    print(f"[Hospital C] Input shape: {meta['input_shape']}")
    print(f"[Hospital C] Slice: {SLICE_TYPE}")

    return train_loader, val_loader, meta
