"""
src/fl/server.py
----------------
Flower server strategy — FedAvg for Phase 2.

FedAvg aggregation:
    new_global_head = sum(alpha_i * head_i)
    where alpha_i = n_i / sum(n_j)
    n_i = number of training samples at hospital i

This is the vanilla baseline.
Phase 4 replaces this with the NAFO custom strategy where
alpha_i also depends on validation quality and 5G network delay.

Aggregation metrics:
    The server receives per-client metrics from fit() and evaluate().
    We log the weighted average accuracy across all hospitals.
"""

import numpy as np
import flwr as fl
from flwr.common import Metrics
from typing import List, Optional, Tuple, Dict


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate per-client evaluation metrics weighted by n_samples.

    Flower calls this after each round's evaluate() across all clients.
    Returns a single dict that gets logged by the simulation.

    Args:
        metrics: list of (n_samples, metrics_dict) from each client

    Returns:
        aggregated metrics dict
    """
    # Weighted average accuracy
    total_samples = sum(n for n, _ in metrics)
    weighted_acc = sum(n * m["accuracy"] for n, m in metrics) / total_samples

    return {"accuracy": weighted_acc}


def build_fedavg_strategy(
    initial_parameters: fl.common.Parameters,
    min_clients: int = 4,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
) -> fl.server.strategy.FedAvg:
    """
    Build and return the FedAvg strategy.

    Args:
        initial_parameters : initial global head weights
        min_clients        : minimum hospitals per round
        fraction_fit       : fraction of clients selected for training
        fraction_evaluate  : fraction of clients selected for evaluation

    Returns:
        Configured FedAvg strategy
    """
    return fl.server.strategy.FedAvg(
        # Client selection
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        # Start from our initialised head weights
        # (avoids random init divergence in round 1)
        initial_parameters=initial_parameters,
        # Aggregate evaluation metrics across hospitals
        evaluate_metrics_aggregation_fn=weighted_average,
        # Aggregate fit metrics (train losses)
        fit_metrics_aggregation_fn=lambda metrics: {
            "train_loss": np.mean([m["train_loss"] for _, m in metrics])
        },
    )
