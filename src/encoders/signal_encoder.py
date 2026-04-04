"""
src/encoders/signal_encoder.py
-------------------------------
Hospital B — 1D CNN Signal Encoder
ECG beat segment (1, 187) → 64-dim latent vector

Architecture:
    Input  : (batch, 1, 187)    ← 1 channel, 187 time steps
    Conv1  : Conv1d(1→32, k=5) + BatchNorm + ReLU + MaxPool(2) → (32, 91)
    Conv2  : Conv1d(32→64, k=5) + BatchNorm + ReLU + MaxPool(2) → (64, 43)
    Conv3  : Conv1d(64→128, k=3) + BatchNorm + ReLU             → (128, 41)
    Pool   : AdaptiveAvgPool1d(1) → (128, 1)
    Flatten: → (128,)
    FC     : Linear(128→64) + ReLU
    Output : (batch, 64)

Design choices:
    kernel_size=5 in first two layers captures local ECG morphology
    (P-wave, QRS complex, T-wave features are 10-50 samples wide)
    kernel_size=3 in third layer captures finer temporal patterns
    AdaptiveAvgPool1d(1) makes the encoder input-length agnostic
    BatchNorm after every conv stabilises training

Phase 1 : trained jointly with SharedHead on Hospital B data
Phase 2+: stays LOCAL — never transmitted over 5G
          Only SharedClassifierHead weights cross the network

5G Slice: URLLC
    Real-time ECG updates, hard latency deadline
"""

import torch
import torch.nn as nn
from src.encoders.base import BaseEncoder, LATENT_DIM


class SignalEncoder(BaseEncoder):
    """
    1D CNN encoder for 187-sample ECG beat segments.
    Input : (batch, 1, 187)  — (batch, channels, time)
    Output: (batch, 64)      — verified by BaseEncoder assertion
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        # ── Convolutional feature extractor ──────────────────────────────
        self.conv_net = nn.Sequential(
            # Block 1: temporal feature extraction
            # Input : (batch, 1, 187)
            # Output: (batch, 32, 91)  — MaxPool halves time dim
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Block 2: deeper feature extraction
            # Input : (batch, 32, 91)
            # Output: (batch, 64, 43)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Block 3: high-level patterns
            # Input : (batch, 64, 43)
            # Output: (batch, 128, 41)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Global average pooling — collapses time dim to 1
            # Output: (batch, 128, 1)
            nn.AdaptiveAvgPool1d(1),
        )

        # ── Fully connected projection to 64-dim ─────────────────────────
        self.fc = nn.Sequential(
            nn.Flatten(),  # (batch, 128)
            nn.Dropout(dropout),
            nn.Linear(128, LATENT_DIM),  # (batch, 64)
            nn.ReLU(),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args   : x — Tensor(batch, 1, 187)
        Returns: Tensor(batch, 64)
        """
        features = self.conv_net(x)  # (batch, 128, 1)
        latent = self.fc(features)  # (batch, 64)
        return latent


if __name__ == "__main__":
    enc = SignalEncoder()
    out = enc(torch.randn(8, 1, 187))
    assert out.shape == (8, 64)
    print(f"SignalEncoder OK — output: {out.shape}")
