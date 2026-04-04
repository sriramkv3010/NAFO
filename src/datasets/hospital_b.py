"""
src/datasets/hospital_b.py
--------------------------
Hospital B — MIT-BIH Arrhythmia Database
Files : data/hospital_b/*.dat  *.atr  *.hea  (48 records)

Database facts:
    48 records total (100–124, 200–234, some gaps)
    Sampled at 360 Hz, 2-channel ECG (we use channel 0 — MLII)
    Each record ~30 minutes of continuous ECG
    Annotations in .atr file: beat type at each R-peak location

Beat segmentation (standard in literature):
    Extract 187 samples centred on each annotated R-peak
    At 360 Hz: 187 samples ≈ 0.52 seconds per beat
    Before R-peak : 90 samples
    After R-peak  : 96 samples  (90 + 1 + 96 = 187)
    This window is the Kachuee et al. (2018) standard,
    used in every major IEEE paper citing this dataset.

AAMI EC57 5-class mapping (mandatory for IEEE citation):
    N — Normal + Left/Right bundle branch + Atrial escape + Nodal escape
    S — Supraventricular ectopic (Atrial, Aberrant atrial, Nodal, Premature)
    V — Ventricular ectopic (Premature ventricular, Ventricular escape)
    F — Fusion (Fusion of ventricular and normal)
    Q — Unclassifiable (Paced, Fusion of paced, Unclassified)

Class imbalance (expected from full 48-record dataset):
    N : ~75,000 beats  (~83%)  ← severe majority
    V :  ~7,100 beats  (~8%)
    S :  ~2,700 beats  (~3%)
    F :    ~800 beats  (~1%)
    Q :    ~200 beats  (<1%)
    → WeightedRandomSampler is critical here

Inter-patient split (standard protocol, prevents data leakage):
    DS1 (train) : 22 records — 101,106,108,109,112,114,115,116,
                               118,119,122,124,201,203,205,207,
                               208,209,215,220,223,230
    DS2 (test)  : 22 records — 100,103,105,111,113,117,121,123,
                               200,202,210,212,213,214,219,221,
                               222,228,231,232,233,234
    4 excluded  : 102,104,107,217 (paced rhythms only — not useful
                  for arrhythmia classification)

5G Slice (Phase 3): URLLC
    Real-time ECG monitoring — mission critical.
    A missed arrhythmia update has a direct clinical consequence.
    Requires hard latency guarantee, not best-effort WiFi delivery.

Dependency:
    pip install wfdb
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Dict, List
import wfdb

# ── Constants ─────────────────────────────────────────────────────────────────

# Beat window: 90 samples before R-peak, 96 after → 187 total
# Standard: Kachuee et al. 2018, IEEE TNSRE
BEAT_BEFORE = 90
BEAT_AFTER = 96
BEAT_LENGTH = BEAT_BEFORE + 1 + BEAT_AFTER  # 187 samples
ECG_CHANNEL = 0  # Lead MLII (channel 0 in MIT-BIH)
FS = 360  # sampling frequency Hz

# ── AAMI EC57 beat-type mapping ───────────────────────────────────────────────
# Source: ANSI/AAMI EC57:1998 standard
# Each raw wfdb annotation symbol maps to one of 5 classes
AAMI_MAP = {
    # N — Normal and bundle branch beats
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    # S — Supraventricular ectopic
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,
    # V — Ventricular ectopic
    "V": 2,
    "E": 2,
    # F — Fusion
    "F": 3,
    # Q — Unclassifiable / paced
    "/": 4,
    "f": 4,
    "Q": 4,
}

CLASS_NAMES = ["N (Normal)", "S (SVE)", "V (Ventricular)", "F (Fusion)", "Q (Unknown)"]

# ── Inter-patient split — DS1/DS2 protocol ───────────────────────────────────
# Standard in arrhythmia literature to prevent patient-level data leakage
# Records 102, 104, 107, 217 excluded (paced rhythms only)
DS1_RECORDS = [
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
]
DS2_RECORDS = [
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
]

# 5G slice assignment for this hospital
SLICE_TYPE = "URLLC"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BEAT EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def extract_beats_from_record(
    record_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one MIT-BIH record and extract all annotated beats.

    Args:
        record_path: full path WITHOUT extension
                     e.g. "data/hospital_b/100"

    Returns:
        beats  : float32 array of shape (n_beats, 187)
                 each row is one normalised beat segment
        labels : int64 array of shape (n_beats,)
                 AAMI class index 0-4

    Beats whose window extends beyond signal boundaries are skipped.
    Beat types not in AAMI_MAP are skipped (noise, artefacts etc.).
    """
    # Load ECG signal — physical units (mV)
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, ECG_CHANNEL].astype(np.float32)  # (N_samples,)
    n_samples = len(signal)

    # Load beat annotations
    annotation = wfdb.rdann(record_path, "atr")
    r_peaks = annotation.sample  # R-peak sample indices
    symbols = annotation.symbol  # beat type symbols

    beats_list = []
    labels_list = []

    for peak, sym in zip(r_peaks, symbols):
        # Skip beat types not in AAMI mapping (noise, non-beat markers)
        if sym not in AAMI_MAP:
            continue

        # Compute window boundaries
        start = peak - BEAT_BEFORE
        end = peak + BEAT_AFTER + 1  # +1 for Python slice exclusivity

        # Skip if window exceeds signal boundaries
        if start < 0 or end > n_samples:
            continue

        # Extract beat segment
        beat = signal[start:end]  # shape (187,)

        # Per-beat normalisation: zero mean, unit variance
        # Removes baseline wander and amplitude differences between patients
        std = beat.std()
        if std < 1e-6:
            continue  # flat line — skip
        beat = (beat - beat.mean()) / std

        beats_list.append(beat)
        labels_list.append(AAMI_MAP[sym])

    if len(beats_list) == 0:
        return np.empty((0, BEAT_LENGTH), dtype=np.float32), np.empty(
            (0,), dtype=np.int64
        )

    beats = np.stack(beats_list, axis=0).astype(np.float32)  # (n_beats, 187)
    labels = np.array(labels_list, dtype=np.int64)  # (n_beats,)

    return beats, labels


def load_split(
    data_dir: str,
    record_ids: List[str],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and concatenate beats from a list of record IDs.

    Args:
        data_dir   : path to folder containing .dat/.atr/.hea files
        record_ids : list of record ID strings e.g. ["100", "101"]
        verbose    : print per-record stats

    Returns:
        all_beats  : float32 array (total_beats, 187)
        all_labels : int64 array  (total_beats,)
    """
    all_beats = []
    all_labels = []
    skipped = []

    for rid in record_ids:
        record_path = os.path.join(data_dir, rid)

        # Skip if record files don't exist (partial dataset)
        if not os.path.exists(record_path + ".dat"):
            skipped.append(rid)
            continue

        beats, labels = extract_beats_from_record(record_path)

        if len(beats) == 0:
            skipped.append(rid)
            continue

        all_beats.append(beats)
        all_labels.append(labels)

        if verbose:
            unique, counts = np.unique(labels, return_counts=True)
            dist = {CLASS_NAMES[u]: int(c) for u, c in zip(unique, counts)}
            print(f"  Record {rid}: {len(beats):5d} beats — {dist}")

    if skipped:
        print(f"  [Skipped records — files not found]: {skipped}")

    if len(all_beats) == 0:
        raise RuntimeError(
            f"No beats extracted from {data_dir}. "
            "Check that .dat and .atr files are present."
        )

    return (
        np.concatenate(all_beats, axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PYTORCH DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ECGDataset(Dataset):
    """
    PyTorch Dataset for MIT-BIH beat segments.

    Each sample is shape (1, 187) — the leading 1 is the channel dim
    required by Conv1d in the SignalEncoder.

    Labels are long (int64) for CrossEntropyLoss (5-class problem).
    """

    def __init__(self, beats: np.ndarray, labels: np.ndarray):
        # Add channel dimension: (N, 187) → (N, 1, 187)
        self.X = torch.tensor(beats[:, np.newaxis, :], dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN LOADER FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_hospital_b(
    data_dir: str,
    batch_size: int = 64,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Full pipeline: raw MIT-BIH files → train/val DataLoaders.

    Uses the standard DS1/DS2 inter-patient split.
    No random splitting — split is deterministic by record ID.

    Args:
        data_dir   : path to folder with all 48 record files
        batch_size : samples per batch (64 works well for ECG)
        verbose    : print per-record extraction stats

    Returns:
        train_loader : DataLoader with WeightedRandomSampler
        val_loader   : DataLoader (no shuffle)
        meta         : dict with stats for NAFO Phase 4
    """

    print(f"[Hospital B] Loading MIT-BIH Arrhythmia (48 records)...")
    print(
        f"[Hospital B] Beat window: {BEAT_LENGTH} samples @ {FS} Hz "
        f"({BEAT_LENGTH/FS*1000:.0f} ms)"
    )
    print(f"[Hospital B] DS1 (train): {len(DS1_RECORDS)} records")

    # ── Load training set (DS1) ────────────────────────────────────────
    train_beats, train_labels = load_split(data_dir, DS1_RECORDS, verbose)

    print(f"\n[Hospital B] DS2 (val): {len(DS2_RECORDS)} records")

    # ── Load validation set (DS2) ──────────────────────────────────────
    val_beats, val_labels = load_split(data_dir, DS2_RECORDS, verbose)

    # ── Class distribution ─────────────────────────────────────────────
    print(f"\n[Hospital B] Training class distribution:")
    train_counts = np.bincount(train_labels, minlength=5)
    for i, (name, count) in enumerate(zip(CLASS_NAMES, train_counts)):
        pct = 100 * count / len(train_labels)
        print(f"  Class {i} {name:<20}: {count:6d} ({pct:.1f}%)")

    print(f"\n[Hospital B] Validation class distribution:")
    val_counts = np.bincount(val_labels, minlength=5)
    for i, (name, count) in enumerate(zip(CLASS_NAMES, val_counts)):
        pct = 100 * count / len(val_labels)
        print(f"  Class {i} {name:<20}: {count:6d} ({pct:.1f}%)")

    # ── Build Datasets ─────────────────────────────────────────────────
    train_ds = ECGDataset(train_beats, train_labels)
    val_ds = ECGDataset(val_beats, val_labels)

    # ── WeightedRandomSampler ──────────────────────────────────────────
    # Critical for MIT-BIH: class N is ~83% of beats
    # Without this, the model predicts N for everything
    # and reports 83% accuracy while missing all arrhythmias
    class_counts = np.bincount(train_labels, minlength=5)
    # Avoid division by zero for classes with 0 samples
    class_weights = np.where(
        class_counts > 0,
        1.0 / class_counts.astype(float),
        0.0,
    )
    sample_weights = class_weights[train_labels]

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

    # ── Metadata for NAFO ─────────────────────────────────────────────
    meta = {
        "hospital": "hospital_b",
        "slice": SLICE_TYPE,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_classes": 5,
        "class_names": CLASS_NAMES,
        "beat_length": BEAT_LENGTH,
        "input_shape": (1, BEAT_LENGTH),  # for Conv1d: (channels, length)
        "train_counts": train_counts.tolist(),
        "val_counts": val_counts.tolist(),
    }

    print(
        f"\n[Hospital B] Train: {meta['n_train']:,} beats | "
        f"Val: {meta['n_val']:,} beats"
    )
    print(f"[Hospital B] Input shape per beat: {meta['input_shape']}")
    print(f"[Hospital B] Slice: {SLICE_TYPE}")

    return train_loader, val_loader, meta
