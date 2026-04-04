"""
src/utils/logger.py
-------------------
Centralised JSON logger used across all four phases.

Every training script writes one entry per epoch/round to
logs/training_history.json. The Phase 4 dashboard reads from
this same file to render live charts.

Log entry schema:
{
    "phase"         : str,    # "phase1", "phase2", "phase3", "phase4"
    "round"         : int,    # epoch number (Phase 1) or FL round (Phase 2+)
    "hospital"      : str,    # "hospital_a" / "hospital_b" / etc.
    "train_loss"    : float,
    "val_loss"      : float,
    "val_acc"       : float,
    "val_auc"       : float | null,
    "n_samples"     : int,
    "sinr_db"       : float | null,   # Phase 3+
    "compression_k" : int   | null,   # Phase 4
    "epsilon_spent" : float | null,   # Phase 2+ (DP)
    "nafo_alpha"    : float | null,   # Phase 4
    "timestamp"     : str             # ISO 8601 UTC
}

Usage:
    from src.utils.logger import RoundLogger
    logger = RoundLogger("logs/training_history.json")
    logger.log(phase="phase1", round=1, hospital="hospital_a",
               train_loss=0.52, val_loss=0.48, val_acc=0.80,
               n_samples=237)
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional


class RoundLogger:
    """
    Appends one JSON entry per training round.
    Creates the log file if it does not exist.
    """

    def __init__(self, log_path: str = "logs/training_history.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # Initialise with empty list if file is new
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                json.dump([], f)

    def log(
        self,
        phase: str,
        round: int,
        hospital: str,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        n_samples: int,
        val_auc: Optional[float] = None,
        sinr_db: Optional[float] = None,
        compression_k: Optional[int] = None,
        epsilon_spent: Optional[float] = None,
        nafo_alpha: Optional[float] = None,
    ) -> None:
        """Append one entry to the JSON log file."""

        entry = {
            "phase": phase,
            "round": round,
            "hospital": hospital,
            "train_loss": _f(train_loss),
            "val_loss": _f(val_loss),
            "val_acc": _f(val_acc),
            "val_auc": _f(val_auc) if val_auc is not None else None,
            "n_samples": n_samples,
            "sinr_db": _f(sinr_db) if sinr_db is not None else None,
            "compression_k": compression_k,
            "epsilon_spent": _f(epsilon_spent) if epsilon_spent is not None else None,
            "nafo_alpha": _f(nafo_alpha) if nafo_alpha is not None else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.log_path, "r") as f:
            history = json.load(f)
        history.append(entry)
        with open(self.log_path, "w") as f:
            json.dump(history, f, indent=2)

    def read_all(self) -> list:
        """Return all logged entries."""
        with open(self.log_path, "r") as f:
            return json.load(f)


def _f(value: float, decimals: int = 6) -> float:
    """Round float to avoid 16-decimal JSON bloat."""
    return round(float(value), decimals)
