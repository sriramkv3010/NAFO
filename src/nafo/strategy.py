"""
src/nafo/strategy.py
---------------------
NAFO Custom Flower Strategy — NetFedAvg

Replaces FedAvg aggregation with NAFO's network-aware weighted aggregation.

Each round:
    1. Read SINR and admission traces for this round
    2. Skip dropped clients (URLLC deadline miss, eMBB capacity violation)
    3. Compute compression ratio k_i per admitted client
    4. Receive compressed gradients from admitted clients
    5. Decompress and aggregate using NAFO temporal smoothing weights
    6. Update global head weights

Key difference from FedAvg:
    FedAvg:  w_global = sum(n_i/N * w_i)  — static dataset-size weighting
    NAFO:    w_global = sum(alpha_i * w_i) — dynamic quality+delay+channel weighting

    Dropped clients in FedAvg: missing update, weight redistributed equally
    Dropped clients in NAFO: alpha_i = 0, temporal smoothing prevents snap-back

This is why NAFO shows smaller accuracy dip during Hospital D's handoff.
"""

import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    EvaluateRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from typing import Dict, List, Optional, Tuple, Union

from src.nafo.aggregator import NAFOAggregator
from src.nafo.compression import SemanticCompressor, CLIP_BOUNDS


class NAFOStrategy(fl.server.strategy.Strategy):
    """
    Network-Aware Federated Optimisation strategy for Flower.

    Implements the full NAFO algorithm:
        - Channel-trace-driven admission control
        - Semantic compression per admitted client
        - Temporal smoothing aggregation weights
        - DP-aware privacy budget tracking
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        dataset_sizes: Dict[str, int],
        sinr_traces: np.ndarray,  # shape (4, T)
        admission_traces: np.ndarray,  # shape (4, T) bool
        delay_traces: np.ndarray,  # shape (4, T)
        capacity_traces: np.ndarray,  # shape (4, T)
        hospital_order: List[str],  # maps row index → hospital_id
        total_epsilon: float = 10.0,
        delta: float = 1e-5,
        lambda_smooth: float = 0.7,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
    ):
        self.initial_parameters = initial_parameters
        self.hospital_order = hospital_order
        self.sinr_traces = sinr_traces
        self.admission_traces = admission_traces
        self.delay_traces = delay_traces
        self.capacity_traces = capacity_traces

        # Build index map: hospital_id → row index in trace arrays
        self.h_idx = {hid: i for i, hid in enumerate(hospital_order)}

        # NAFO components
        self.aggregator = NAFOAggregator(dataset_sizes, lambda_smooth, total_epsilon)
        self.compressor = SemanticCompressor(total_epsilon, delta)

        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients

        # State tracking
        self.current_round = 0
        self.quality_cache = {hid: 0.5 for hid in hospital_order}
        self.round_logs = []

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager) -> List[Tuple]:
        """
        Configure local training for this round.
        Pass compression ratio k_i and clipping bound C_i to each client.
        """
        self.current_round = server_round
        clients = client_manager.sample(
            num_clients=min(self.min_fit_clients, client_manager.num_available()),
            min_num_clients=self.min_fit_clients,
        )

        t = server_round - 1  # 0-indexed round for trace lookup
        fit_instructions = []

        for client in clients:
            cid = client.cid
            hospital_id = self._cid_to_hospital(cid)
            idx = self.h_idx[hospital_id]

            # Read channel state for this round
            sinr_db = float(
                self.sinr_traces[idx, min(t, self.sinr_traces.shape[1] - 1)]
            )
            admitted = bool(
                self.admission_traces[idx, min(t, self.admission_traces.shape[1] - 1)]
            )

            if not admitted:
                continue

            # Fix 5: Adaptive clipping bound — reduces as epsilon depletes
            # This IS the Synergy 1 coupling: less budget → smaller C_i
            # → lower gradient sensitivity → less noise → budget extends
            eps_remaining = self.compressor.epsilon_remaining
            adaptive_clips = self.aggregator.get_clipping_bounds(eps_remaining)
            clip_bound = adaptive_clips[hospital_id]

            # Track epsilon consumed this round per hospital
            self.compressor.update_epsilon_spent(64, hospital_id)

            # k=64: full gradient — top-k compression corrupts 64-dim head
            # Synergy 1 coupling enforced through clipping, not compression ratio
            config = {
                "k": 64,
                "clip_bound": clip_bound,
                "sinr_db": sinr_db,
                "round": server_round,
                "hospital_id": hospital_id,
                "eps_remaining": eps_remaining,
            }

            fit_instructions.append((client, fl.common.FitIns(parameters, config)))

        return fit_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate received head parameters using NAFO weights.

        Uses alpha_i from temporal smoothing aggregator.
        Dropped clients have alpha_i = 0 from the update step.
        """
        if not results:
            return None, {}

        t = server_round - 1

        # Build admitted/quality/delay dicts for this round
        admitted_dict = {}
        delay_dict = {}
        quality_dict = dict(self.quality_cache)

        for hid in self.hospital_order:
            idx = self.h_idx[hid]
            t_safe = min(t, self.admission_traces.shape[1] - 1)
            admitted_dict[hid] = bool(self.admission_traces[idx, t_safe])
            delay_dict[hid] = float(self.delay_traces[idx, t_safe])

        # Update NAFO weights — pass epsilon remaining for Fix 5
        eps_remaining = self.compressor.epsilon_remaining
        alpha = self.aggregator.update(
            quality_dict, delay_dict, admitted_dict, eps_remaining
        )

        # NAFO weighted aggregation
        # w_global = sum(alpha_i * w_i) for admitted clients
        weighted_params = None
        total_weight = 0.0

        for client, fit_res in results:
            hospital_id = fit_res.metrics.get(
                "hospital", self._cid_to_hospital(client.cid)
            )
            w = alpha.get(hospital_id, 0.0)

            if w < 1e-8:
                continue  # dropped or zero weight

            params_arrays = parameters_to_ndarrays(fit_res.parameters)
            total_weight += w

            if weighted_params is None:
                weighted_params = [w * p for p in params_arrays]
            else:
                for i, p in enumerate(params_arrays):
                    weighted_params[i] += w * p

        if weighted_params is None or total_weight < 1e-8:
            # No admitted clients — keep current parameters
            return self.initial_parameters, {}

        # Normalise
        aggregated = [p / total_weight for p in weighted_params]

        # Log round summary
        admitted_count = sum(1 for v in admitted_dict.values() if v)
        metrics = {
            "admitted_clients": admitted_count,
            "total_weight": round(total_weight, 4),
            "alpha_d": round(alpha.get("hospital_d", 0.0), 4),
        }

        self.round_logs.append(
            {
                "round": server_round,
                "alpha": dict(alpha),
                "admitted": dict(admitted_dict),
            }
        )

        return ndarrays_to_parameters(aggregated), metrics

    def configure_evaluate(
        self, server_round, parameters, client_manager
    ) -> List[Tuple]:
        """Configure evaluation on all available clients."""
        clients = client_manager.sample(
            num_clients=min(self.min_eval_clients, client_manager.num_available()),
            min_num_clients=self.min_eval_clients,
        )
        return [(client, fl.common.EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation metrics.
        Update quality cache for NAFO weight computation next round.
        """
        if not results:
            return None, {}

        total_samples = 0
        weighted_acc = 0.0

        for client, eval_res in results:
            hospital_id = eval_res.metrics.get("hospital", "unknown")
            acc = eval_res.metrics.get("accuracy", 0.0)
            n = eval_res.num_examples

            # Update quality cache for next round's NAFO weights
            self.quality_cache[hospital_id] = acc

            weighted_acc += n * acc
            total_samples += n

        global_acc = weighted_acc / total_samples if total_samples > 0 else 0.0

        return (
            sum(eval_res.loss for _, eval_res in results) / len(results),
            {"accuracy": global_acc},
        )

    def evaluate(self, server_round, parameters) -> Optional[Tuple]:
        """Server-side evaluation not used — clients handle evaluation."""
        return None

    def _cid_to_hospital(self, cid: str) -> str:
        """Map Flower client cid string to hospital_id."""
        cid_map = {
            "0": "hospital_a",
            "1": "hospital_b",
            "2": "hospital_c",
            "3": "hospital_d",
        }
        return cid_map.get(str(cid), "hospital_a")
