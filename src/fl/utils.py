"""
src/fl/utils.py
---------------
Parameter serialisation helpers for Flower federation.

NAFO federation rule:
    Only SharedClassifierHead parameters are transmitted.
    Encoder parameters stay local and are never serialised.

Flower communicates model weights as List[np.ndarray].
These helpers convert between PyTorch state_dict and that format.

Used by:
    src/fl/client.py  — get/set head parameters each round
    phase2_fedavg/run_fedavg.py — initialise server weights
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as a list of numpy arrays.
    Flower sends these to the server for aggregation.

    Args:
        model: the SharedClassifierHead (NOT the encoder)

    Returns:
        List of numpy arrays, one per parameter tensor
    """
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load parameters received from the server into the local head.

    Args:
        model      : the SharedClassifierHead to update
        parameters : list of numpy arrays from Flower server
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
