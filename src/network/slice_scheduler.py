"""
src/network/slice_scheduler.py
-------------------------------
5G NR Slice Scheduler using SimPy discrete-event simulation.

This is what makes the 5G claim real.
WiFi has no slice concept. A WiFi simulator would just
randomly drop clients. This scheduler drops specific clients
based on slice-specific QoS violations — causally different.

Three slices, three QoS contracts:

URLLC (Hospital B — ECG, Hospital D — PPG wearable):
    - Target delay   : <1ms (3GPP TS 22.261 URLLC requirement)
    - Min SINR       : 10 dB (below this, packet error rate too high)
    - Miss behaviour : client DROPPED from round — no update transmitted
    - Clinical reason: ECG arrhythmia detection is time-critical

eMBB (Hospital C — Chest X-ray):
    - Min capacity   : 5 Mbps (DICOM file transfer requirement)
    - Miss behaviour : client DELAYED — waits for next round slot
    - Clinical reason: radiologist review is not real-time

mMTC (Hospital A — Tabular EHR):
    - Always admitted (delay-tolerant by design)
    - Low power periodic sync
    - Clinical reason: blood pressure/cholesterol sync is batch

Usage:
    scheduler = SliceScheduler(seed=42)
    decisions = scheduler.schedule_round(sinr_dict, capacity_dict)
    # decisions = {"hospital_a": True, "hospital_b": False, ...}
"""

import simpy
import numpy as np
from typing import Dict, Tuple

# ── QoS thresholds per slice ───────────────────────────────────────────────────
# Source: 3GPP TS 22.261 V17.0.0 Table 5.7.4-1
SLICE_QOS = {
    "URLLC": {
        "max_delay_ms": 1.0,  # hard deadline — miss = dropped
        "min_sinr_db": -3.0,  # minimum for reliable transmission
        "min_capacity_mbps": 0.5,  # minimum throughput
        "on_violation": "drop",  # client excluded from this round
    },
    "eMBB": {
        "max_delay_ms": 100.0,
        "min_sinr_db": -10.0,  # 3GPP minimum for 100MHz eMBB operation
        "min_capacity_mbps": 5.0,  # FL gradient payload (64 floats) + overhead
        "on_violation": "drop",  # drop — cannot transmit imaging payload
    },
    "mMTC": {
        "max_delay_ms": 10000.0,  # effectively always admitted
        "min_sinr_db": -5.0,  # very low SINR tolerance (NB-IoT)
        "min_capacity_mbps": 0.1,
        "on_violation": "admit",  # always let through
    },
}

# Hospital to slice mapping
HOSPITAL_SLICE = {
    "hospital_a": "mMTC",
    "hospital_b": "URLLC",
    "hospital_c": "eMBB",
    "hospital_d": "URLLC",
}


class SliceScheduler:
    """
    Discrete-event 5G slice scheduler.

    For each FL round, determines which hospitals are admitted
    based on their channel quality and slice QoS requirements.

    The scheduling decision causally affects:
        - Which clients participate in that round
        - NAFO aggregation weights (dropped = alpha_i = 0)
        - AoI tracking (dropped = information grows staler)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _compute_delay_ms(
        self,
        sinr_db: float,
        capacity_mbps: float,
        slice_type: str,
    ) -> float:
        """
        Estimate transmission delay for one hospital.

        Delay model: base propagation + queuing + retransmission
        Retransmission probability increases with lower SINR.

        Args:
            sinr_db      : current channel SINR
            capacity_mbps: Shannon capacity
            slice_type   : URLLC / eMBB / mMTC

        Returns:
            estimated round-trip delay in ms
        """
        # Base propagation delay (speed of light over typical cell radius)
        base_delay_ms = 0.1

        # Queuing delay inversely proportional to capacity
        queue_delay_ms = 10.0 / max(capacity_mbps, 0.1)

        # Retransmission probability: higher when SINR is low
        # BLER ≈ 10% at SINR=5dB, 1% at SINR=15dB (approximate)
        sinr_clamped = max(min(sinr_db, 25.0), -10.0)
        bler = 0.1 * np.exp(-0.1 * (sinr_clamped - 5.0))
        retx_overhead = 1.0 + bler * 4.0  # retransmission multiplier

        total_delay_ms = (base_delay_ms + queue_delay_ms) * retx_overhead

        # Add jitter: URLLC requires deterministic latency
        jitter_ms = self.rng.uniform(0, 0.2 if slice_type == "URLLC" else 2.0)

        return float(total_delay_ms + jitter_ms)

    def schedule_round(
        self,
        sinr_dict: Dict[str, float],
        capacity_dict: Dict[str, float],
        round_num: int = 0,
    ) -> Dict[str, dict]:
        """
        Make admission decisions for all four hospitals in one FL round.

        Args:
            sinr_dict    : {hospital_id: sinr_db} for this round
            capacity_dict: {hospital_id: capacity_mbps} for this round
            round_num    : current FL round number (for logging)

        Returns:
            decisions dict:
            {
                hospital_id: {
                    "admitted"    : bool,
                    "delay_ms"    : float,
                    "sinr_db"     : float,
                    "capacity_mbps": float,
                    "slice"       : str,
                    "violation"   : str | None,
                }
            }
        """
        decisions = {}

        for hospital_id, slice_type in HOSPITAL_SLICE.items():
            sinr = sinr_dict.get(hospital_id, 0.0)
            capacity = capacity_dict.get(hospital_id, 0.0)
            qos = SLICE_QOS[slice_type]

            delay_ms = self._compute_delay_ms(sinr, capacity, slice_type)
            violation = None
            admitted = True

            # ── URLLC: hard deadline enforcement ──────────────────────────
            if slice_type == "URLLC":
                if delay_ms > qos["max_delay_ms"]:
                    violation = f"deadline_miss (delay={delay_ms:.2f}ms > {qos['max_delay_ms']}ms)"
                    admitted = False
                elif sinr < qos["min_sinr_db"]:
                    violation = f"low_sinr ({sinr:.1f}dB < {qos['min_sinr_db']}dB)"
                    admitted = False

            # ── eMBB: SINR floor + capacity requirement ────────────────────
            elif slice_type == "eMBB":
                if sinr < qos["min_sinr_db"]:
                    violation = f"low_sinr ({sinr:.1f}dB < {qos['min_sinr_db']}dB)"
                    admitted = False
                elif capacity < qos["min_capacity_mbps"]:
                    violation = f"insufficient_capacity ({capacity:.1f}Mbps < {qos['min_capacity_mbps']}Mbps)"
                    admitted = False

            # ── mMTC: always admitted ─────────────────────────────────────
            # else: mMTC hospitals always go through

            decisions[hospital_id] = {
                "admitted": admitted,
                "delay_ms": round(delay_ms, 3),
                "sinr_db": round(sinr, 2),
                "capacity_mbps": round(capacity, 2),
                "slice": slice_type,
                "violation": violation,
            }
            for hid, d in decisions.items():
                if not d["admitted"]:
                    print(f"    DROPPED {hid}: {d['violation']}")

        return decisions

    def run_simulation(
        self,
        sinr_traces: Dict[str, np.ndarray],
        capacity_traces: Dict[str, np.ndarray],
        num_rounds: int,
    ) -> Dict[str, list]:
        """
        Run the full SimPy discrete-event simulation for T rounds.

        Args:
            sinr_traces    : {hospital_id: array of shape (T,)}
            capacity_traces: {hospital_id: array of shape (T,)}
            num_rounds     : total FL rounds

        Returns:
            results: {hospital_id: list of per-round decision dicts}
        """
        env = simpy.Environment()
        results = {hid: [] for hid in HOSPITAL_SLICE}

        def fl_round(env, round_num):
            """SimPy process: one FL round."""
            sinr_now = {
                hid: float(sinr_traces[hid][round_num]) for hid in HOSPITAL_SLICE
            }
            capacity_now = {
                hid: float(capacity_traces[hid][round_num]) for hid in HOSPITAL_SLICE
            }

            decisions = self.schedule_round(sinr_now, capacity_now, round_num)

            for hid, decision in decisions.items():
                results[hid].append(decision)

            # Simulate round duration (each round takes some simulated time)
            yield env.timeout(1.0)

        def run_all_rounds(env):
            for r in range(num_rounds):
                yield env.process(fl_round(env, r))

        env.process(run_all_rounds(env))
        env.run()

        return results

    def print_summary(self, results: Dict[str, list]) -> None:
        """Print admission statistics across all rounds."""
        print("\n[Slice Scheduler Summary]")
        print(
            f"  {'Hospital':<14} {'Slice':<8} {'Admitted':<10} {'Dropped':<10} {'Admit rate'}"
        )
        print(f"  {'-'*58}")
        for hid in ["hospital_a", "hospital_b", "hospital_c", "hospital_d"]:
            rounds = results[hid]
            n_admit = sum(1 for r in rounds if r["admitted"])
            n_drop = len(rounds) - n_admit
            rate = n_admit / len(rounds) if rounds else 0.0
            slice_t = rounds[0]["slice"] if rounds else "?"
            print(f"  {hid:<14} {slice_t:<8} {n_admit:<10} {n_drop:<10} {rate:.1%}")
