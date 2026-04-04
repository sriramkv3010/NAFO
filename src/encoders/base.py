"""
src/encoders/base.py
--------------------
Abstract base encoder — enforces the 64-dim bottleneck contract.

NAFO architectural rule:
    Any modality → Encoder → torch.Tensor(batch, 64) → SharedHead

All four hospital encoders inherit from BaseEncoder.
The forward() method asserts output shape == (batch, 64).
A wrong shape raises immediately with a clear message rather than
silently corrupting the shared head downstream.

Usage:
    class MyEncoder(BaseEncoder):
        def _encode(self, x):
            return self.net(x)    # must output (batch, 64)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# ── Single source of truth for bottleneck size ────────────────────────────────
# Imported by: all encoders, shared_head, fl/client.py, nafo/compression.py
LATENT_DIM = 64


class BaseEncoder(ABC, nn.Module):
    """
    Abstract base. Subclasses implement _encode(x) only.
    Never override forward().
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Modality-specific encoding.
        Must return Tensor of shape (batch_size, LATENT_DIM).
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calls _encode() and verifies 64-dim contract."""
        latent = self._encode(x)
        assert latent.shape[-1] == LATENT_DIM, (
            f"{self.__class__.__name__} output dim = {latent.shape[-1]}, "
            f"expected {LATENT_DIM}. Fix _encode()."
        )
        return latent
