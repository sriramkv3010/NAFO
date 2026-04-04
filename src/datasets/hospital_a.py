"""
src/datasets/hospital_a.py
--------------------------
Hospital A — UCI Heart Disease
File : data/hospital_a/processed.cleveland.data

Verified facts from actual file inspection:
    303 rows total, no header, comma-separated
    14 columns — 13 features + target (last column)
    6 rows contain '?' in column ca (index 11) or thal (index 12)
    Target: 0 = healthy, 1/2/3/4 = disease severity grades
    After binarization: 0 = healthy (160), 1 = disease (137)

Preprocessing pipeline:
    1. Read CSV with na_values='?' to catch missing markers
    2. Drop 6 rows with missing values → 297 clean rows
    3. Binarize target: 0 stays 0, values 1-4 become 1
    4. Stratified 80/20 train/val split
    5. StandardScaler fit on train only (no val leakage)
    6. WeightedRandomSampler to handle class imbalance in training

5G Slice (Phase 3): mMTC
    Small periodic tabular payload from EHR system.
    Delay-tolerant — does not need URLLC guarantees.
    Lowest bandwidth requirement of the four hospitals.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

# ── Column names — file has no header ─────────────────────────────────────────
# Source: https://archive.ics.uci.edu/dataset/45/heart+disease
COLUMNS = [
    "age",  # continuous : age in years
    "sex",  # binary     : 1=male, 0=female
    "cp",  # categorical: chest pain type 1-4
    "trestbps",  # continuous : resting blood pressure (mmHg)
    "chol",  # continuous : serum cholesterol (mg/dl)
    "fbs",  # binary     : fasting blood sugar >120mg/dl
    "restecg",  # categorical: resting ECG results 0-2
    "thalach",  # continuous : maximum heart rate achieved
    "exang",  # binary     : exercise induced angina
    "oldpeak",  # continuous : ST depression induced by exercise
    "slope",  # categorical: slope of peak exercise ST segment 1-3
    "ca",  # integer    : major vessels coloured 0-3 ← HAS MISSING
    "thal",  # categorical: 3=normal, 6=fixed, 7=reversible ← HAS MISSING
    "target",  # integer    : 0=healthy, 1-4=disease → binarized
]


class HeartDiseaseDataset(Dataset):
    """
    PyTorch Dataset for UCI Heart Disease data.

    Returns (features, label) as float32 tensors.
    float32 is mandatory — MPS backend does not support float64.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, 13)
        self.y = torch.tensor(y, dtype=torch.float32)  # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_hospital_a(
    data_path: str,
    batch_size: int = 32,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Full pipeline: raw file → train DataLoader + val DataLoader.

    Args:
        data_path  : path to processed.cleveland.data
        batch_size : samples per batch
        test_size  : fraction for validation (default 20%)
        seed       : random seed for reproducibility

    Returns:
        train_loader : DataLoader with WeightedRandomSampler
        val_loader   : DataLoader (no shuffle)
        meta         : dict with stats — used by NAFO in Phase 4
    """

    # ── 1. Load raw file ───────────────────────────────────────────────────────
    # na_values='?' converts missing markers to NaN so dropna() catches them
    df = pd.read_csv(
        data_path,
        header=None,
        names=COLUMNS,
        na_values="?",
    )
    n_raw = len(df)

    # ── 2. Drop rows with missing values ──────────────────────────────────────
    # Exactly 6 rows have '?' in 'ca' or 'thal' — verified from file
    df = df.dropna()
    n_clean = len(df)
    print(
        f"[Hospital A] {n_raw} rows → {n_raw - n_clean} dropped " f"→ {n_clean} clean"
    )

    # ── 3. Binarize target ────────────────────────────────────────────────────
    # Original severity grades 1/2/3/4 are not used — we do binary prediction
    df["target"] = (df["target"] > 0).astype(int)
    n_pos = int(df["target"].sum())
    n_neg = n_clean - n_pos
    print(f"[Hospital A] Healthy (0): {n_neg} | Disease (1): {n_pos}")

    # ── 4. Separate features and labels ───────────────────────────────────────
    feature_cols = [c for c in COLUMNS if c != "target"]
    X = df[feature_cols].values.astype(np.float32)  # (297, 13)
    y = df["target"].values.astype(np.float32)  # (297,)

    # ── 5. Stratified train / val split ───────────────────────────────────────
    # stratify=y ensures both splits have the same class ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # ── 6. StandardScaler — fit on train ONLY ─────────────────────────────────
    # Fitting on val would leak distribution info → inflated metrics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # ── 7. Verify no NaNs survived scaling ────────────────────────────────────
    assert not np.isnan(X_train).any(), "NaN in train features after scaling"
    assert not np.isnan(X_val).any(), "NaN in val features after scaling"

    # ── 8. Build PyTorch Datasets ─────────────────────────────────────────────
    train_ds = HeartDiseaseDataset(X_train, y_train)
    val_ds = HeartDiseaseDataset(X_val, y_val)

    # ── 9. WeightedRandomSampler ──────────────────────────────────────────────
    # Oversamples minority class so model doesn't just predict "healthy"
    # to get 54% accuracy for free
    class_counts = np.bincount(y_train.astype(int))  # [n_neg, n_pos]
    class_weights = 1.0 / class_counts.astype(float)  # inverse freq
    sample_weights = class_weights[y_train.astype(int)]  # per-sample weight

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(train_ds),
        replacement=True,
    )

    # sampler and shuffle=True are mutually exclusive in DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # ── 10. Metadata ──────────────────────────────────────────────────────────
    # Used by NAFO aggregator (Phase 4): n_train feeds into alpha_i formula
    # Used by loss function: pos_weight addresses class imbalance
    meta = {
        "hospital": "hospital_a",
        "slice": "mMTC",  # 5G slice assignment (Phase 3)
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_features": 13,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "pos_weight": float(n_neg / n_pos),  # for BCEWithLogitsLoss
        "scaler": scaler,  # save for inference
    }

    print(
        f"[Hospital A] Train: {meta['n_train']} | Val: {meta['n_val']} | "
        f"pos_weight: {meta['pos_weight']:.4f}"
    )

    return train_loader, val_loader, meta
