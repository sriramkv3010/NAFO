"""
src/encoders/image_encoder.py
------------------------------
Hospital C — 2D CNN Image Encoder
Chest X-ray (1, 28, 28) → 64-dim latent vector

Architecture:
    Input  : (batch, 1, 28, 28)  ← normalised grayscale X-ray
    Block 1: Conv2d(1→16, k=3)  + BatchNorm + ReLU + MaxPool(2) → (16, 13, 13)
    Block 2: Conv2d(16→32, k=3) + BatchNorm + ReLU + MaxPool(2) → (32, 6, 6)
    Block 3: Conv2d(32→64, k=3) + BatchNorm + ReLU              → (64, 4, 4)
    Pool   : AdaptiveAvgPool2d(1)                                → (64, 1, 1)
    Flatten: → (64,)
    FC     : Linear(64→64) + ReLU
    Output : (batch, 64)

Design choices:
    kernel_size=3 throughout — standard for small 28x28 images
    MaxPool(2) after first two blocks to progressively reduce spatial dims
    AdaptiveAvgPool2d(1) collapses any spatial size to a single vector
    BatchNorm after every conv — stabilises training on medical images
    No final sigmoid/tanh — latent space stays unconstrained

Input note:
    28x28 is already small (MedMNIST standard downsampling from full DICOM)
    Do NOT resize further — spatial features would be lost

Phase 1 : trained jointly with SharedHead on Hospital C data
Phase 2+: stays LOCAL — never transmitted over the 5G link
          Only SharedClassifierHead weights cross the network

5G Slice: eMBB
    Large imaging payload — high bandwidth, latency-tolerant
"""

import torch
import torch.nn as nn
from src.encoders.base import BaseEncoder, LATENT_DIM


class ImageEncoder(BaseEncoder):
    """
    2D CNN encoder for 28x28 grayscale chest X-ray images.
    Input : (batch, 1, 28, 28)
    Output: (batch, 64)  ← verified by BaseEncoder assertion
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        # ── Convolutional backbone ────────────────────────────────────────
        self.conv_net = nn.Sequential(
            # Block 1: (1, 28, 28) → (16, 13, 13)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 28 → 13 (floor division)
            # Block 2: (16, 13, 13) → (32, 6, 6)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 13 → 6
            # Block 3: (32, 6, 6) → (64, 4, 4)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Global average pool — collapses spatial dims to 1x1
            # Output: (batch, 64, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )

        # ── Projection to 64-dim latent ───────────────────────────────────
        self.fc = nn.Sequential(
            nn.Flatten(),  # (batch, 64)
            nn.Dropout(dropout),
            nn.Linear(64, LATENT_DIM),  # (batch, 64)
            nn.ReLU(),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args   : x — Tensor(batch, 1, 28, 28)
        Returns: Tensor(batch, 64)
        """
        features = self.conv_net(x)  # (batch, 64, 1, 1)
        latent = self.fc(features)  # (batch, 64)
        return latent


if __name__ == "__main__":
    enc = ImageEncoder()
    out = enc(torch.randn(8, 1, 28, 28))
    assert out.shape == (8, 64)
    print(f"ImageEncoder OK — output: {out.shape}")
