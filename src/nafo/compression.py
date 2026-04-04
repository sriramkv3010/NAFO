"""
src/nafo/compression.py
-----------------------
NAFO Semantic Compression — Synergy 1 and Synergy 3

Synergy 1 — Sparsity-Aware Privacy Budgeting:
    k = f(SINR, epsilon_remaining)

    Standard approach: compress based only on channel quality.
    NAFO approach: compress based on BOTH channel quality AND
    remaining DP privacy budget.

    When SINR is low → high compression → fewer dimensions sent
    → lower global sensitivity → less noise required for same (ε,δ)
    → ε budget lasts longer.

    Counter-intuitive finding: bad 5G weather extends privacy lifespan.
    This is the paper's key theoretical contribution from Synergy 1.

    Mathematical basis:
        Global sensitivity S_k = C * sqrt(k) / n
        where C = clipping norm, k = compressed dims, n = batch size
        Required noise: σ = S_k * sqrt(2*ln(1.25/δ)) / ε
        Lower k → lower S_k → lower σ → less ε consumed per round

Synergy 3 — EPL Objective for Hospital D (wearable URLLC):
    Given SINR, remaining ε, and URLLC deadline → optimal k (closed form)

    Objective: maximise k (model quality) subject to:
        1. Transmission completes within URLLC deadline (1ms)
        2. DP noise does not exceed a quality floor
        3. Channel capacity can carry k dimensions

    Closed form: k* = min(k_channel, k_privacy, k_deadline, 64)

Usage:
    comp = SemanticCompressor(total_epsilon=10.0, delta=1e-5)
    k    = comp.compute_k(hospital_id, sinr_db, round_num)
    compressed = comp.compress(gradient_vector, k)
    reconstructed = comp.decompress(compressed, original_dim=64)
"""

import numpy as np
from typing import Dict, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
LATENT_DIM = 64  # SharedHead input dimension — fixed contract
URLLC_DEADLINE_MS = 1.0  # 3GPP TS 22.261 URLLC hard deadline

# Modality-adaptive clipping bounds (Synergy 2 — defined here for reference)
# eMBB (Hospital C, 2D CNN): large gradients, higher clip
# URLLC (Hospitals B, D): medium gradients
# mMTC (Hospital A, MLP): small gradients, tighter clip
CLIP_BOUNDS = {
    "hospital_a": 0.5,  # mMTC  — tabular MLP, small gradient norms
    "hospital_b": 1.0,  # URLLC — 1D CNN signal encoder
    "hospital_c": 1.5,  # eMBB  — 2D CNN image encoder, largest norms
    "hospital_d": 1.0,  # URLLC — 1D CNN wearable encoder
}

# Minimum k values per slice — never compress below this
MIN_K = {
    "hospital_a": 8,  # mMTC: can tolerate high compression
    "hospital_b": 16,  # URLLC: moderate minimum
    "hospital_c": 32,  # eMBB: image features need more dims
    "hospital_d": 16,  # URLLC: same as B
}

# Slice type per hospital (for capacity lookup)
HOSPITAL_SLICE = {
    "hospital_a": "mMTC",
    "hospital_b": "URLLC",
    "hospital_c": "eMBB",
    "hospital_d": "URLLC",
}

# Payload size in bytes for one k-dimensional gradient vector
# float32 = 4 bytes per dimension
BYTES_PER_DIM = 4


class SemanticCompressor:
    """
    Channel-aware and privacy-aware semantic compression.

    Computes optimal compression ratio k per hospital per round
    using both SINR (channel quality) and remaining ε (privacy budget).

    Compression mechanism:
        1. Compute gradient vector g ∈ R^64
        2. Select top-k dimensions by absolute magnitude
        3. Transmit (indices, values) — k * (4 + 4) bytes
        4. Server reconstructs sparse vector in R^64
    """

    def __init__(
        self,
        total_epsilon: float = 10.0,  # total DP budget across all rounds
        delta: float = 1e-5,  # DP delta parameter
        num_rounds: int = 20,  # total FL rounds
        lambda_smooth: float = 0.7,  # temporal smoothing (for EPL)
    ):
        self.total_epsilon = total_epsilon
        self.delta = delta
        self.num_rounds = num_rounds
        self.lambda_smooth = lambda_smooth

        # Per-round epsilon budget: uniform allocation baseline
        # NAFO dynamically adjusts based on compression ratio
        self.epsilon_per_round = total_epsilon / num_rounds
        self.epsilon_spent = 0.0  # cumulative epsilon spent

    @property
    def epsilon_remaining(self) -> float:
        """Remaining privacy budget."""
        return max(self.total_epsilon - self.epsilon_spent, 0.0)

    def epsilon_per_dim(self, k: int, clip_bound: float) -> float:
        """
        Epsilon consumed per round for k compressed dimensions.

        Global sensitivity of compressed gradient:
            S_k = clip_bound * sqrt(k) / sqrt(batch_size)
        We approximate batch contribution to sensitivity as clip_bound/sqrt(k).

        Lower k → lower sensitivity → less epsilon consumed.
        This is the mathematical core of Synergy 1.

        Args:
            k          : number of dimensions transmitted
            clip_bound : per-hospital gradient clipping norm

        Returns:
            epsilon consumed this round
        """
        # Sensitivity scales as sqrt(k/64) — fewer dims = lower exposure
        sensitivity_ratio = np.sqrt(k / LATENT_DIM)
        epsilon_used = self.epsilon_per_round * sensitivity_ratio
        return float(epsilon_used)

    def compute_k_from_sinr(self, sinr_db: float, bandwidth_mhz: float) -> int:
        """
        Compression ratio from channel capacity (Synergy 1, channel term).

        Maps SINR → Shannon capacity → k dimensions that can be transmitted
        within URLLC deadline or eMBB window.

        Args:
            sinr_db      : current round SINR
            bandwidth_mhz: slice bandwidth

        Returns:
            k_channel: max dimensions transmittable given channel
        """
        sinr_lin = 10.0 ** (np.clip(sinr_db, -20, 40) / 10.0)
        capacity_bps = bandwidth_mhz * 1e6 * np.log2(1.0 + sinr_lin)

        # Payload: k floats (4 bytes each) + k indices (4 bytes each)
        # Must fit within transmission window (10ms for URLLC, 100ms for eMBB)
        window_ms = 1.0 if bandwidth_mhz <= 20 else 100.0
        max_bytes = (capacity_bps / 8.0) * (window_ms / 1000.0)
        k_channel = int(max_bytes / (2 * BYTES_PER_DIM))  # index + value

        return int(np.clip(k_channel, 1, LATENT_DIM))

    def compute_k_from_epsilon(
        self,
        hospital_id: str,
        epsilon_remaining: float,
    ) -> int:
        """
        Compression ratio from remaining privacy budget (Synergy 1, privacy term).

        When epsilon is nearly exhausted, force higher compression
        to reduce sensitivity and stretch the remaining budget.

        This is the Synergy 1 mechanism: bad channel → high compression
        → less sensitivity → less noise → epsilon lasts longer.

        Args:
            hospital_id      : for clip bound lookup
            epsilon_remaining: remaining DP budget

        Returns:
            k_privacy: max dims allowed given remaining epsilon
        """
        clip_bound = CLIP_BOUNDS[hospital_id]
        budget_fraction = epsilon_remaining / self.total_epsilon

        if budget_fraction > 0.5:
            # Plenty of budget — allow up to full dimension
            k_privacy = LATENT_DIM
        elif budget_fraction > 0.25:
            # Budget getting low — moderate compression
            k_privacy = LATENT_DIM // 2
        elif budget_fraction > 0.1:
            # Budget tight — high compression
            k_privacy = LATENT_DIM // 4
        else:
            # Near exhaustion — minimum transmission
            k_privacy = MIN_K[hospital_id]

        return int(k_privacy)

    def compute_k_epl(
        self,
        sinr_db: float,
        epsilon_remaining: float,
        bandwidth_mhz: float = 20.0,
    ) -> int:
        """
        EPL (Energy-Privacy-Latency) closed-form optimal k for Hospital D.
        Synergy 3.

        Hospital D is a wearable URLLC device. Before each transmission it
        solves:
            "Given my SINR, remaining ε, and URLLC 1ms deadline,
             what is the maximum k I can transmit without violating
             any constraint?"

        k* = min(k_channel, k_privacy, k_deadline, 64)

        k_deadline: maximum k transmittable within 1ms URLLC hard deadline
        k_channel : maximum k given Shannon capacity
        k_privacy : maximum k given remaining ε budget
        k_full    : 64 (never exceed full dimension)

        Args:
            sinr_db          : current SINR
            epsilon_remaining: remaining DP budget
            bandwidth_mhz    : URLLC slice bandwidth

        Returns:
            k* optimal compression ratio for Hospital D
        """
        # k_channel from Shannon capacity
        k_channel = self.compute_k_from_sinr(sinr_db, bandwidth_mhz)

        # k_privacy from epsilon budget
        k_privacy = self.compute_k_from_epsilon("hospital_d", epsilon_remaining)

        # k_deadline: URLLC hard deadline 1ms
        # At given capacity, max bytes in 1ms window
        sinr_lin = 10.0 ** (np.clip(sinr_db, -20, 40) / 10.0)
        cap_bps = bandwidth_mhz * 1e6 * np.log2(1.0 + sinr_lin)
        max_bytes = (cap_bps / 8.0) * (URLLC_DEADLINE_MS / 1000.0)
        k_deadline = int(max_bytes / (2 * BYTES_PER_DIM))

        # Optimal k: minimum of all constraints
        k_star = min(k_channel, k_privacy, k_deadline, LATENT_DIM)
        k_star = max(k_star, MIN_K["hospital_d"])  # never below minimum

        return int(k_star)

    def compute_k(
        self,
        hospital_id: str,
        sinr_db: float,
        round_num: int,
        bw_mhz: float = 20.0,
    ) -> int:
        """
        Master k computation — dispatches to EPL for Hospital D,
        standard Synergy 1 for others.

        Args:
            hospital_id : hospital identifier
            sinr_db     : current round SINR
            round_num   : current FL round (for logging)
            bw_mhz      : slice bandwidth in MHz

        Returns:
            k: number of gradient dimensions to transmit (1-64)
        """
        eps_remaining = self.epsilon_remaining
        min_k = MIN_K[hospital_id]

        if hospital_id == "hospital_d":
            # Synergy 3: EPL objective (closed form)
            k = self.compute_k_epl(sinr_db, eps_remaining, bw_mhz)
        else:
            # Synergy 1: joint SINR + epsilon constraint
            k_channel = self.compute_k_from_sinr(sinr_db, bw_mhz)
            k_privacy = self.compute_k_from_epsilon(hospital_id, eps_remaining)
            k = min(k_channel, k_privacy, LATENT_DIM)
            k = max(k, min_k)

        return int(k)

    def compress(
        self,
        gradient: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Top-k semantic compression.

        Selects the k dimensions with largest absolute magnitude.
        These carry the most gradient information — semantic content.

        Args:
            gradient : flat numpy array, shape (D,) where D=64
            k        : number of dimensions to keep

        Returns:
            indices : int array shape (k,) — which dimensions
            values  : float32 array shape (k,) — their values
        """
        k = min(k, len(gradient))
        abs_g = np.abs(gradient)
        indices = np.argsort(abs_g)[-k:]  # top-k by magnitude
        values = gradient[indices].astype(np.float32)
        return indices.astype(np.int32), values

    def decompress(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        original_dim: int = LATENT_DIM,
    ) -> np.ndarray:
        """
        Reconstruct full-dimensional gradient from sparse representation.
        Unselected dimensions are set to zero.

        Args:
            indices     : which dimensions were transmitted
            values      : their values
            original_dim: full dimension (64)

        Returns:
            dense gradient array shape (original_dim,)
        """
        dense = np.zeros(original_dim, dtype=np.float32)
        dense[indices] = values
        return dense

    def update_epsilon_spent(self, k: int, hospital_id: str) -> float:
        """
        Record epsilon consumed this round and update running total.

        Args:
            k          : compression ratio used
            hospital_id: for clip bound lookup

        Returns:
            epsilon consumed this round
        """
        clip_bound = CLIP_BOUNDS[hospital_id]
        eps_used = self.epsilon_per_dim(k, clip_bound)
        self.epsilon_spent = min(self.epsilon_spent + eps_used, self.total_epsilon)
        return eps_used

    def compression_stats(self, hospital_id: str, k: int, sinr_db: float) -> dict:
        """Return compression statistics for logging."""
        return {
            "hospital": hospital_id,
            "k": k,
            "compression_ratio": round(k / LATENT_DIM, 3),
            "sinr_db": round(sinr_db, 2),
            "epsilon_remaining": round(self.epsilon_remaining, 4),
            "clip_bound": CLIP_BOUNDS[hospital_id],
        }
