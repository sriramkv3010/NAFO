"""
src/encoders/tabular_encoder.py
--------------------------------
Hospital A — Tabular MLP Encoder
13 StandardScaled clinical features → 64-dim latent vector

Architecture:
    Input  : (batch, 13)
    Linear(13 → 128) + BatchNorm1d + ReLU + Dropout(0.3)
    Linear(128 → 64) + BatchNorm1d + ReLU
    Output : (batch, 64)

Design choices:
    BatchNorm1d  — stabilises training on small datasets (297 rows)
    Dropout(0.3) — reduces overfitting on tabular data
    No final activation — latent space stays unconstrained so the
    shared head can learn its own decision boundaries freely

Phase 1 : trained jointly with SharedHead on local Hospital A data
Phase 2+: stays LOCAL, never transmitted over the network
          Only SharedHead weights cross the 5G link
"""

import torch
import torch.nn as nn
from src.encoders.base import BaseEncoder, LATENT_DIM


class TabularEncoder(BaseEncoder):
    """
    MLP encoder for 13-feature cardiac tabular data.
    Input : (batch, 13)
    Output: (batch, 64)  ← verified by BaseEncoder assertion
    """

    def __init__(self, input_dim: int = 13, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            # ── Layer 1: 13 → 128 ────────────────────────────────────────
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            # ── Layer 2: 128 → 64 bottleneck ─────────────────────────────
            nn.Linear(128, LATENT_DIM),
            nn.BatchNorm1d(LATENT_DIM),
            nn.ReLU(),
            # No activation — unconstrained latent space
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args   : x — Tensor(batch, 13)
        Returns: Tensor(batch, 64)
        """
        return self.net(x)


if __name__ == "__main__":
    enc = TabularEncoder()
    out = enc(torch.randn(8, 13))
    assert out.shape == (8, 64)
    print(f"TabularEncoder OK — output: {out.shape}")
