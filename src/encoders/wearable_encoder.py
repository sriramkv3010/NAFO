"""
src/encoders/wearable_encoder.py
---------------------------------
Hospital D — 1D CNN Wearable Encoder
PPG segment (1, 625) → 64-dim latent vector

Why 1D CNN instead of LSTM:
    LSTM has incomplete MPS support on Apple Silicon.
    With 625-step sequences it runs on CPU fallback,
    causing extreme slowness and training instability on M4.
    1D CNN runs natively on MPS, trains 10x faster,
    and performs equally well on PPG signals.
    Hospital B proved 1D CNN works for physiological time-series.

Architecture:
    Input  : (batch, 1, 625)
    Block 1: Conv1d(1→32,  k=7, pad=3) + BN + ReLU + MaxPool(2) → (32, 312)
    Block 2: Conv1d(32→64, k=5, pad=2) + BN + ReLU + MaxPool(2) → (64, 156)
    Block 3: Conv1d(64→128,k=5, pad=2) + BN + ReLU + MaxPool(2) → (128, 78)
    Block 4: Conv1d(128→128,k=3,pad=1) + BN + ReLU              → (128, 78)
    Pool   : AdaptiveAvgPool1d(1)                                → (128, 1)
    Flatten: → (128,)
    FC     : Linear(128→64) + ReLU
    Output : (batch, 64)

Design notes:
    kernel_size=7 in block 1 captures PPG pulse wave morphology
    (a single pulse at 125Hz spans ~60-100 samples)
    Progressively larger channels capture hierarchical features
    AdaptiveAvgPool1d(1) makes encoder length-agnostic

Phase 1 : trained jointly with SharedHead on Hospital D data
Phase 2+: stays LOCAL — never transmitted over the 5G link
"""

import torch
import torch.nn as nn
from src.encoders.base import BaseEncoder, LATENT_DIM


class WearableEncoder(BaseEncoder):
    """
    1D CNN encoder for 625-sample PPG waveform segments.
    Input : (batch, 1, 625)
    Output: (batch, 64)  ← verified by BaseEncoder assertion
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.conv_net = nn.Sequential(
            # Block 1: large kernel captures pulse wave shape
            # (batch, 1, 625) → (batch, 32, 312)
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2: (batch, 32, 312) → (batch, 64, 156)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3: (batch, 64, 156) → (batch, 128, 78)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 4: refine features
            # (batch, 128, 78) → (batch, 128, 78)
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Global average pool → (batch, 128, 1)
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, LATENT_DIM),
            nn.ReLU(),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args   : x — Tensor(batch, 1, 625)
        Returns: Tensor(batch, 64)
        """
        return self.fc(self.conv_net(x))


if __name__ == "__main__":
    enc = WearableEncoder()
    out = enc(torch.randn(8, 1, 625))
    assert out.shape == (8, 64)
    print(f"WearableEncoder OK — output: {out.shape}")
