"""
src/utils/device.py
-------------------
Centralised device selector for Apple Silicon M4.

Priority order:
    1. MPS  — Apple Silicon GPU (M4 Air)
    2. CUDA — NVIDIA GPU (not on M4, future-proof)
    3. CPU  — fallback

IMPORTANT: Before running any script, set this in your terminal:
    export PYTORCH_ENABLE_MPS_FALLBACK=1

This makes ops not yet supported on MPS silently fall back to CPU
instead of crashing. Always set it — costs nothing, prevents errors.

Usage (everywhere in the project):
    from src.utils.device import DEVICE
    model = MyModel().to(DEVICE)
    tensor = tensor.to(DEVICE)
"""

import torch


def get_device() -> torch.device:
    """Return best available device for this machine."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Single global constant — import DEVICE everywhere ─────────────────────────
DEVICE = get_device()


if __name__ == "__main__":
    print(f"Active device : {DEVICE}")
    print(f"MPS available : {torch.backends.mps.is_available()}")
