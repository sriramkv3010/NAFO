"""
src/models/shared_head.py
--------------------------
Shared Classifier Head — 64 → 32 → 1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS IS THE ONLY MODULE TRANSMITTED OVER THE 5G LINK.
All four hospital encoders stay local.
Only this module's state_dict() is federated via Flower.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture:
    Input  : Tensor(batch, 64)  ← from any modality encoder
    Linear(64 → 32) + ReLU + Dropout(0.3)
    Linear(32 → 1)              ← raw logit (no sigmoid)
    Output : Tensor(batch, 1)

Loss pairing:
    Always use nn.BCEWithLogitsLoss — applies sigmoid internally.
    More numerically stable than sigmoid() + BCELoss().
    During inference: prob = torch.sigmoid(logit)

Phase 1 : trained jointly with local encoder
Phase 2 : get_parameters() / set_parameters() operate on this only
Phase 4 : NAFO compression applied to this module's gradient updates
"""

import torch
import torch.nn as nn
from src.encoders.base import LATENT_DIM


class SharedClassifierHead(nn.Module):
    """
    Binary cardiac risk classifier head.
    Input : (batch, 64) latent from any hospital encoder
    Output: (batch, 1)  raw logit
    """

    def __init__(self, latent_dim: int = LATENT_DIM, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),  # raw logit — no sigmoid
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args   : latent — Tensor(batch, 64)
        Returns: Tensor(batch, 1) raw logit
        """
        return self.net(latent)


if __name__ == "__main__":
    head = SharedClassifierHead()
    out = head(torch.randn(8, 64))
    assert out.shape == (8, 1)
    print(f"SharedClassifierHead OK — output: {out.shape}")
