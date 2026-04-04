"""
src/fl/client.py
----------------
Flower client — one class handles all four hospitals.

Federation rule:
    Encoder  → stays LOCAL, trains locally each round
    SharedHead → transmitted to server after each local round

Hospital B label note:
    Phase 1 trained Hospital B with 5-class arrhythmia labels.
    Phase 2 federation uses BINARY labels across all hospitals:
        0 = N (normal)
        1 = any arrhythmia (S / V / F / Q)
    Conversion happens inside fit() and evaluate() for Hospital B only.
    This unifies the label space so the shared head has one coherent task.

Usage:
    client = HospitalClient(
        hospital_id  = "hospital_a",
        train_loader = train_loader,
        val_loader   = val_loader,
        encoder      = TabularEncoder(),
        meta         = meta,
    )
"""

import torch
import torch.nn as nn
import numpy as np
import flwr as fl
from typing import Dict, List, Tuple

from src.utils.device import DEVICE
from src.utils.logger import RoundLogger
from src.models.shared_head import SharedClassifierHead
from src.fl.utils import get_parameters, set_parameters

# ── Training config per FL round ──────────────────────────────────────────────
LOCAL_EPOCHS = 3  # epochs of local training per FL round
LR = 5e-4  # slightly lower LR than Phase 1 (global head stabilises)
LOG_PATH = "logs/training_history.json"


class HospitalClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for one hospital node.

    Each hospital has:
        - Its own local encoder (never transmitted)
        - A shared head (received from server, trained locally, returned)

    The client trains both encoder and head locally each round,
    but only returns the head parameters to the server.
    """

    def __init__(
        self,
        hospital_id: str,
        train_loader,
        val_loader,
        encoder: nn.Module,
        meta: dict,
        fl_round_ref: list,  # mutable list so client can read current round
    ):
        self.hospital_id = hospital_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.encoder = encoder.to(DEVICE)
        self.head = SharedClassifierHead().to(DEVICE)
        self.meta = meta
        self.fl_round_ref = fl_round_ref
        self.logger = RoundLogger(LOG_PATH)

        # Loss: BCEWithLogitsLoss for binary task across all hospitals
        self.criterion = nn.BCEWithLogitsLoss()

    # ── Flower API ─────────────────────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current head parameters to the server."""
        return get_parameters(self.head)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        1. Load global head weights from server
        2. Train encoder + head locally for LOCAL_EPOCHS
        3. Return updated head weights to server

        Only head weights go back — encoder stays local.
        """
        # Load global head weights sent by server
        set_parameters(self.head, parameters)

        # Build optimizer: encoder + head trained jointly
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=LR,
        )

        # Local training
        total_loss = 0.0
        for _ in range(LOCAL_EPOCHS):
            self.encoder.train()
            self.head.train()
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = self._binarize(y_batch).to(DEVICE)

                optimizer.zero_grad()
                logit = self.head(self.encoder(X_batch))
                loss = self.criterion(logit.squeeze(1), y_batch)
                loss.backward()

                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / (LOCAL_EPOCHS * len(self.train_loader))

        # Log training loss
        current_round = self.fl_round_ref[0]
        self.logger.log(
            phase="phase2_fedavg",
            round=current_round,
            hospital=self.hospital_id,
            train_loss=avg_loss,
            val_loss=0.0,  # filled in evaluate()
            val_acc=0.0,
            n_samples=self.meta["n_train"],
        )

        # Return head parameters ONLY — encoder stays local
        return (
            get_parameters(self.head),
            self.meta["n_train"],
            {"train_loss": avg_loss, "hospital": self.hospital_id},
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate global head on local val set.
        Returns loss, n_samples, metrics dict.
        """
        set_parameters(self.head, parameters)

        self.encoder.eval()
        self.head.eval()

        total_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = self._binarize(y_batch).to(DEVICE)

                logit = self.head(self.encoder(X_batch))
                loss = self.criterion(logit.squeeze(1), y_batch)
                total_loss += loss.item()

                prob = torch.sigmoid(logit.squeeze(1))
                all_probs.extend(prob.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs >= 0.5).astype(int)
        accuracy = float((preds == all_labels).mean())
        val_loss = total_loss / len(self.val_loader)

        return (
            val_loss,
            self.meta["n_val"],
            {"accuracy": accuracy, "hospital": self.hospital_id},
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _binarize(self, y: torch.Tensor) -> torch.Tensor:
        """
        Convert labels to binary for federation.

        Hospital A, C, D: already binary — no change.
        Hospital B      : 5-class arrhythmia → binary
                          0 (N normal) → 0
                          1/2/3/4 (any arrhythmia) → 1
        """
        if self.hospital_id == "hospital_b":
            return (y > 0).float()
        return y.float()
