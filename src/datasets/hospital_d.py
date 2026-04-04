"""
src/datasets/hospital_d.py
--------------------------
Hospital D — Kachuee PPG Dataset
File : data/hospital_d/part_1.mat

Confirmed structure from file inspection:
    data['p'] : shape (1, 1000) dtype=object
                1000 patient records

    data['p'][0, i] : shape (3, N) float64
                      3 signal rows per patient, N samples each
        Row 0 : PPG waveform (photoplethysmography)
        Row 1 : ABP waveform (Arterial Blood Pressure, mmHg)
        Row 2 : Third signal (ECG or respiratory — unused)

    N varies per patient record.

Segmentation strategy:
    Each patient record has a long continuous signal.
    We slice it into fixed windows of SEGMENT_LEN samples.
    For each window:
        PPG segment : row 0 window
        SBP estimate: max(row 1 window) — systolic peak of ABP
        DBP estimate: min(row 1 window) — diastolic trough

Task:
    Binary hypertension classification
        SBP >= 140 mmHg → Hypertension (1)   JNC7 Stage 2 threshold
        SBP <  140 mmHg → Normal (0)

Preprocessing:
    1. Load .mat, loop 1000 records
    2. Slice each into SEGMENT_LEN windows with stride
    3. Per-segment PPG normalisation (zero mean, unit variance)
    4. SBP = max(ABP window), binarize at 140 mmHg
    5. Stratified 80/20 train/val split
    6. WeightedRandomSampler (plain BCEWithLogitsLoss — no pos_weight)

5G Slice (Phase 3): URLLC
    Continuous wearable BP monitoring — hard latency deadline.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from typing import Tuple, Dict, List

# ── Constants ─────────────────────────────────────────────────────────────────
SEGMENT_LEN = 625  # samples per window (5 sec @ 125Hz)
STRIDE = 312  # 50% overlap between windows
SBP_THRESHOLD = 140  # mmHg — JNC7 hypertension cutoff
PPG_ROW = 0  # row index of PPG signal
ABP_ROW = 1  # row index of ABP signal
SLICE_TYPE = "URLLC"


class PPGDataset(Dataset):
    """
    PyTorch Dataset for segmented PPG waveforms.

    Each sample:
        X : float32 Tensor (1, 625) — normalised PPG segment
        y : float32 Tensor scalar  — 0=normal, 1=hypertension
    """

    def __init__(self, segments: np.ndarray, labels: np.ndarray):
        # (N, 625) → (N, 1, 625) for Conv1d / LSTM compatibility
        self.X = torch.tensor(
            segments[:, np.newaxis, :],
            dtype=torch.float32,
        )
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def _segment_record(
    signals: np.ndarray,
    segment_len: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Slice one patient record into fixed-length windows.

    Args:
        signals     : (3, N) array — row 0=PPG, row 1=ABP
        segment_len : window length in samples
        stride      : step between window starts

    Returns:
        ppg_segs : list of (625,) PPG arrays
        sbp_vals : list of SBP floats (max of ABP window)
    """
    ppg_signal = signals[PPG_ROW]  # (N,)
    abp_signal = signals[ABP_ROW]  # (N,)
    n_samples = ppg_signal.shape[0]

    ppg_segs = []
    sbp_vals = []

    start = 0
    while start + segment_len <= n_samples:
        end = start + segment_len

        ppg_win = ppg_signal[start:end]  # (625,)
        abp_win = abp_signal[start:end]  # (625,)

        # SBP = peak of ABP waveform within the window
        sbp = float(abp_win.max())

        # Skip physiologically implausible values
        # Valid SBP range: 70-250 mmHg
        if sbp < 70 or sbp > 250:
            start += stride
            continue

        # Per-segment PPG normalisation
        std = ppg_win.std()
        if std < 1e-6:
            start += stride
            continue
        ppg_norm = (ppg_win - ppg_win.mean()) / std

        ppg_segs.append(ppg_norm.astype(np.float32))
        sbp_vals.append(sbp)
        start += stride

    return ppg_segs, sbp_vals


def load_hospital_d(
    data_dir: str,
    batch_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Full pipeline: part_1.mat → train/val DataLoaders.

    Args:
        data_dir   : folder containing the .mat file
        batch_size : samples per batch
        test_size  : fraction for validation (default 20%)
        seed       : random seed for reproducibility

    Returns:
        train_loader : DataLoader with WeightedRandomSampler
        val_loader   : DataLoader (no shuffle)
        meta         : dict with stats for NAFO Phase 4
    """

    # ── 1. Find and load .mat ──────────────────────────────────────────────────
    mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    if not mat_files:
        raise FileNotFoundError(f"No .mat file in {data_dir}")

    mat_path = os.path.join(data_dir, mat_files[0])
    print(f"[Hospital D] Loading: {mat_path}")
    data = loadmat(mat_path)

    records = data["p"]  # shape (1, 1000)
    n_records = records.shape[1]
    print(f"[Hospital D] Records: {n_records}")
    print(
        f"[Hospital D] Segment: {SEGMENT_LEN} samples | "
        f"Stride: {STRIDE} | SBP threshold: {SBP_THRESHOLD} mmHg"
    )

    # ── 2. Extract all segments across all records ─────────────────────────────
    all_ppg = []
    all_sbp = []
    skipped = 0

    for i in range(n_records):
        rec = records[0, i]  # (3, N) float64

        # Skip malformed records
        if rec.ndim != 2 or rec.shape[0] < 2:
            skipped += 1
            continue

        n_samples = rec.shape[1]
        if n_samples < SEGMENT_LEN:
            skipped += 1
            continue

        ppg_segs, sbp_vals = _segment_record(rec, SEGMENT_LEN, STRIDE)

        all_ppg.extend(ppg_segs)
        all_sbp.extend(sbp_vals)

    print(f"[Hospital D] Skipped records : {skipped}")
    print(f"[Hospital D] Total segments  : {len(all_ppg):,}")

    if len(all_ppg) == 0:
        raise RuntimeError(
            "No valid segments extracted. "
            "Check that row 1 contains ABP values in range 70-250 mmHg."
        )

    # ── 3. Stack into arrays ───────────────────────────────────────────────────
    X = np.stack(all_ppg, axis=0)  # (N, 625)
    sbp_arr = np.array(all_sbp)  # (N,)

    print(
        f"[Hospital D] SBP range : {sbp_arr.min():.1f} – "
        f"{sbp_arr.max():.1f} mmHg "
        f"(mean {sbp_arr.mean():.1f})"
    )

    # ── 4. Binarize SBP ────────────────────────────────────────────────────────
    y = (sbp_arr >= SBP_THRESHOLD).astype(np.float32)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(
        f"[Hospital D] Normal (SBP<140)      : {n_neg:,} " f"({100*n_neg/len(y):.1f}%)"
    )
    print(
        f"[Hospital D] Hypertension (SBP≥140): {n_pos:,} " f"({100*n_pos/len(y):.1f}%)"
    )

    if n_pos == 0:
        raise RuntimeError(
            "No hypertension samples found (SBP >= 140). "
            "The ABP signal in row 1 may have different units or scaling. "
            "Print sbp_arr[:20] to inspect raw values."
        )

    # ── 5. Stratified train/val split ─────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    print(f"[Hospital D] Train: {len(X_train):,} | Val: {len(X_val):,}")

    # ── 6. NaN check ──────────────────────────────────────────────────────────
    assert not np.isnan(X_train).any(), "NaN in training segments"
    assert not np.isnan(X_val).any(), "NaN in val segments"

    # ── 7. Build Datasets ─────────────────────────────────────────────────────
    train_ds = PPGDataset(X_train, y_train)
    val_ds = PPGDataset(X_val, y_val)

    # ── 8. WeightedRandomSampler ──────────────────────────────────────────────
    class_counts = np.bincount(y_train.astype(int))
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
        sampler=sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # ── 9. Metadata ────────────────────────────────────────────────────────────
    meta = {
        "hospital": "hospital_d",
        "slice": SLICE_TYPE,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_train_pos": int(y_train.sum()),
        "n_train_neg": int((y_train == 0).sum()),
        "n_val_pos": int(y_val.sum()),
        "n_val_neg": int((y_val == 0).sum()),
        "input_shape": (1, SEGMENT_LEN),
        "sbp_threshold": SBP_THRESHOLD,
    }

    print(f"[Hospital D] Slice: {SLICE_TYPE}")
    return train_loader, val_loader, meta
