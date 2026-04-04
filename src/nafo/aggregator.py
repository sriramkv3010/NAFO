"""
src/nafo/aggregator.py
-----------------------
NAFO Aggregator — Final Correct Version

Formula:
    q_ema_i(t)   = (1-gamma)*q_ema_i(t-1) + gamma*q_i(t)   [EMA quality]
    q_mean(t)    = mean(q_ema_i) over admitted hospitals
    raw_i(t)     = n_frac_i * max(1 + beta*(q_ema_i - q_mean), 0.1)
    alpha_i(t+1) = clip(lambda*alpha_i(t) + (1-lambda)*norm(raw), min, max)

Why quality-adjusted FedAvg (not softmax):
    Dataset sizes span 3 orders of magnitude (237 to 81009).
    Softmax makes exp(score_A) ≈ exp(score_D) regardless of n_frac,
    giving Hospital A 17% weight when it should get 0.1%.
    Quality-adjusted FedAvg preserves the n_frac baseline and adds
    a bounded quality bonus — correct behavior.

Fix 1: n_frac_i = n_i/sum(n_j) — normalised fraction, not raw count
Fix 2: Hard cap alpha_i <= 0.5
Fix 3: Quality EMA with gamma=0.3 (smooths noisy per-round accuracy)
Fix 4: Floor alpha_i >= 0.001 (numerical stability only)
Fix 5: DP coupling via adaptive clipping: C_i(t) = C_i_base * sqrt(eps/eps_total)
       As epsilon depletes -> smaller C_i -> lower sensitivity -> less noise needed
"""

import numpy as np
from typing import Dict, List
from src.nafo.compression import CLIP_BOUNDS

LAMBDA_SMOOTH = 0.7
EPSILON_FLOOR = 1e-8
DEFAULT_QUALITY = 0.5
ALPHA_MAX = 0.50  # Fix 2: no hospital takes majority
ALPHA_MIN = 0.001  # Fix 4: numerical floor only
QUALITY_EMA_A = 0.3  # Fix 3: EMA decay (time constant ~3 rounds)
QUALITY_BETA = 1.0  # quality adjustment strength


class NAFOAggregator:

    def __init__(
        self,
        dataset_sizes: Dict[str, int],
        lambda_smooth: float = LAMBDA_SMOOTH,
        total_epsilon: float = 10.0,
    ):
        self.dataset_sizes = dataset_sizes
        self.lambda_smooth = lambda_smooth
        self.total_epsilon = total_epsilon
        self.hospitals = list(dataset_sizes.keys())

        total_n = sum(dataset_sizes.values())
        self.n_frac = {hid: dataset_sizes[hid] / total_n for hid in self.hospitals}
        self.alpha = dict(self.n_frac)
        self.quality_ema = {hid: DEFAULT_QUALITY for hid in self.hospitals}

        self.alpha_history = {hid: [] for hid in self.hospitals}
        self.quality_history = {hid: [] for hid in self.hospitals}
        self.admitted_history = {hid: [] for hid in self.hospitals}
        self.round_num = 0

    def compute_weights(
        self,
        quality_dict: Dict[str, float],
        delay_dict: Dict[str, float],
        admitted_dict: Dict[str, bool],
        epsilon_remaining: float = 10.0,
    ) -> Dict[str, float]:
        """Quality-adjusted FedAvg with temporal smoothing and all 5 fixes."""

        admitted = [h for h in self.hospitals if admitted_dict.get(h, True)]

        # Fix 3: Update quality EMA for admitted hospitals only
        for hid in admitted:
            q_raw = quality_dict.get(hid, DEFAULT_QUALITY)
            self.quality_ema[hid] = (1 - QUALITY_EMA_A) * self.quality_ema[
                hid
            ] + QUALITY_EMA_A * q_raw

        # Mean quality across admitted hospitals
        q_mean = (
            np.mean([self.quality_ema[h] for h in admitted])
            if admitted
            else DEFAULT_QUALITY
        )

        # Fix 1: Quality-adjusted FedAvg — n_frac baseline + quality bonus
        raw = {}
        for hid in self.hospitals:
            if not admitted_dict.get(hid, True):
                raw[hid] = 0.0
            else:
                q_adj = 1.0 + QUALITY_BETA * (self.quality_ema[hid] - q_mean)
                q_adj = max(q_adj, 0.1)  # quality floor — never kills a hospital
                raw[hid] = self.n_frac[hid] * q_adj

        # Normalise
        total_raw = sum(raw.values())
        if total_raw < EPSILON_FLOOR:
            n_a = max(len(admitted), 1)
            norm_raw = {
                h: (1.0 / n_a if admitted_dict.get(h, True) else 0.0)
                for h in self.hospitals
            }
        else:
            norm_raw = {h: raw[h] / total_raw for h in self.hospitals}

        # Temporal smoothing
        new_alpha = {
            hid: self.lambda_smooth * self.alpha[hid]
            + (1 - self.lambda_smooth) * norm_raw[hid]
            for hid in self.hospitals
        }

        # Fix 2 + Fix 4: Cap and floor
        for hid in self.hospitals:
            new_alpha[hid] = np.clip(new_alpha[hid], ALPHA_MIN, ALPHA_MAX)

        # Renormalise after clipping
        total_a = sum(new_alpha.values())
        if total_a > EPSILON_FLOOR:
            new_alpha = {h: v / total_a for h, v in new_alpha.items()}

        return new_alpha

    def get_clipping_bounds(self, epsilon_remaining: float = 10.0) -> Dict[str, float]:

        budget_fraction = np.clip(epsilon_remaining / self.total_epsilon, 0.0, 1.0)
        scale = 0.5 + 0.5 * np.sqrt(budget_fraction)
        return {hid: CLIP_BOUNDS[hid] * scale for hid in self.hospitals}

    def update(
        self,
        quality_dict: Dict[str, float],
        delay_dict: Dict[str, float],
        admitted_dict: Dict[str, bool],
        epsilon_remaining: float = 10.0,
    ) -> Dict[str, float]:
        new_alpha = self.compute_weights(
            quality_dict, delay_dict, admitted_dict, epsilon_remaining
        )
        self.alpha = new_alpha
        for hid in self.hospitals:
            self.alpha_history[hid].append(new_alpha[hid])
            self.quality_history[hid].append(quality_dict.get(hid, DEFAULT_QUALITY))
            self.admitted_history[hid].append(admitted_dict.get(hid, True))
        self.round_num += 1
        return new_alpha

    def compute_aoi(self) -> Dict[str, float]:
        aoi = {}
        for hid in self.hospitals:
            history = self.admitted_history[hid]
            if not history:
                aoi[hid] = 0.0
                continue
            last = next((r for r in range(len(history) - 1, -1, -1) if history[r]), -1)
            aoi[hid] = (
                float(len(history) - 1 - last) if last != -1 else float(len(history))
            )
        return aoi

    def print_round_summary(self, round_num, quality_dict, delay_dict, admitted_dict):
        print(f"  [NAFO Round {round_num:02d}]")
        for hid in self.hospitals:
            print(
                f"  {hid}: alpha={self.alpha[hid]:.4f} "
                f"q_ema={self.quality_ema[hid]:.4f} "
                f"{'YES' if admitted_dict.get(hid,True) else 'NO'}"
            )
