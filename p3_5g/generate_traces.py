"""
phase3_5g/generate_traces.py
-----------------------------
Run this ONCE to generate all channel traces.
Output files are committed to git — deterministic and reproducible.

Produces:
    channel_traces/sinr_traces.npy       — shape (4, T) SINR in dB
    channel_traces/capacity_traces.npy   — shape (4, T) capacity in Mbps
    channel_traces/admission_traces.npy  — shape (4, T) bool admission
    channel_traces/delay_traces.npy      — shape (4, T) delay in ms
    channel_traces/metadata.json         — parameters for paper citation

Run from project root:
    python phase3_5g/generate_traces.py

After running, verify with:
    python phase3_5g/verify_traces.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.network.channel_model import (
    TR38901ChannelModel,
    HOSPITAL_PROFILES,
    SLICE_BANDWIDTH_MHZ,
)
from src.network.slice_scheduler import SliceScheduler, HOSPITAL_SLICE
from src.network.handoff import (
    apply_handoff,
    get_distance_trace,
    get_velocity_trace,
    HANDOFF_ROUND,
    HANDOFF_DURATION,
    SINR_DROP_DB,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUM_ROUNDS = 20
SEED = 42
OUTPUT_DIR = "channel_traces"

# Hospital order (index 0-3) for .npy arrays
HOSPITAL_ORDER = ["hospital_a", "hospital_b", "hospital_c", "hospital_d"]


def generate_traces(num_rounds: int = NUM_ROUNDS, seed: int = SEED):
    """
    Generate SINR, capacity, admission, and delay traces for all hospitals.

    Each hospital gets T SINR samples computed from the 3GPP TR 38.901
    channel model. Hospital D additionally has the gNB handoff injected.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    channel_model = TR38901ChannelModel(freq_ghz=3.5, seed=seed)
    scheduler = SliceScheduler(seed=seed)

    # Allocate arrays
    sinr_arr = np.zeros((4, num_rounds), dtype=np.float32)
    capacity_arr = np.zeros((4, num_rounds), dtype=np.float32)
    admission_arr = np.zeros((4, num_rounds), dtype=bool)
    delay_arr = np.zeros((4, num_rounds), dtype=np.float32)

    print(f"\n[Trace Generator]")
    print(f"  Rounds     : {num_rounds}")
    print(f"  Seed       : {seed}")
    print(f"  Frequency  : 3.5 GHz (5G NR n78)")
    print(
        f"  Handoff    : Hospital D at round {HANDOFF_ROUND} "
        f"(duration {HANDOFF_DURATION} rounds, -{SINR_DROP_DB}dB drop)"
    )
    print()

    # ── Generate per-hospital SINR traces ─────────────────────────────────────
    for idx, hospital_id in enumerate(HOSPITAL_ORDER):
        profile = HOSPITAL_PROFILES[hospital_id]
        is_mobile = profile["is_mobile"]

        if is_mobile:
            # Hospital D: use distance and velocity traces
            dist_trace = get_distance_trace(num_rounds)
            vel_trace = get_velocity_trace(num_rounds)
        else:
            dist_trace = np.full(num_rounds, profile["distance_m"])
            vel_trace = np.zeros(num_rounds)

        # Compute raw SINR each round
        raw_sinr = np.zeros(num_rounds)
        for r in range(num_rounds):
            result = channel_model.compute(
                hospital_id=hospital_id,
                distance_m=float(dist_trace[r]),
                is_los=(hospital_id == "hospital_a"),  # A is LOS (close)
                velocity_mps=float(vel_trace[r]),
            )
            raw_sinr[r] = result["sinr_db"]

        # Inject handoff for Hospital D
        if is_mobile:
            final_sinr, handoff_info = apply_handoff(raw_sinr)
            print(
                f"  [hospital_d] Handoff injected at rounds "
                f"{handoff_info['affected_rounds']}"
            )
        else:
            final_sinr = raw_sinr

        sinr_arr[idx] = final_sinr.astype(np.float32)

        # Compute capacity from SINR — correct units: B in Hz → result in bps → /1e6 = Mbps
        slice_type = HOSPITAL_PROFILES[hospital_id]["slice"]
        bw_mhz = SLICE_BANDWIDTH_MHZ[slice_type]
        for r in range(num_rounds):
            sinr_clipped = float(np.clip(final_sinr[r], -20.0, 40.0))
            sinr_lin = 10.0 ** (sinr_clipped / 10.0)
            bw_hz = bw_mhz * 1e6
            cap_bps = bw_hz * np.log2(1.0 + sinr_lin)
            cap_mbps = cap_bps / 1e6
            max_cap = 8.0 * bw_mhz
            # LTE-M hard cap — 3GPP TS 36.306 peak rate = 1 Mbps
            if bw_mhz <= 1.4:
                max_cap = min(max_cap, 1.0)
            capacity_arr[idx, r] = min(cap_mbps, max_cap)

        print(
            f"  [{hospital_id}] SINR: {final_sinr.min():.1f} to "
            f"{final_sinr.max():.1f} dB  "
            f"(mean {final_sinr.mean():.1f} dB)  "
            f"Capacity: {capacity_arr[idx].mean():.1f} Mbps avg"
        )

    # ── Run slice scheduler ────────────────────────────────────────────────────
    sinr_dict = {HOSPITAL_ORDER[i]: sinr_arr[i] for i in range(4)}
    capacity_dict = {HOSPITAL_ORDER[i]: capacity_arr[i] for i in range(4)}

    sim_results = scheduler.run_simulation(sinr_dict, capacity_dict, num_rounds)

    # Fill admission and delay arrays
    for idx, hospital_id in enumerate(HOSPITAL_ORDER):
        for r in range(num_rounds):
            d = sim_results[hospital_id][r]
            admission_arr[idx, r] = d["admitted"]
            delay_arr[idx, r] = d["delay_ms"]

    # ── Save to disk ───────────────────────────────────────────────────────────
    np.save(os.path.join(OUTPUT_DIR, "sinr_traces.npy"), sinr_arr)
    np.save(os.path.join(OUTPUT_DIR, "capacity_traces.npy"), capacity_arr)
    np.save(os.path.join(OUTPUT_DIR, "admission_traces.npy"), admission_arr)
    np.save(os.path.join(OUTPUT_DIR, "delay_traces.npy"), delay_arr)

    # Save metadata for paper citation
    metadata = {
        "num_rounds": num_rounds,
        "seed": seed,
        "freq_ghz": 3.5,
        "standard": "3GPP TR 38.901 V17.0.0",
        "path_loss_model": "UMa LOS/NLOS (Table 7.4.1-1)",
        "shadowing": "Log-normal (Section 7.4.1)",
        "fast_fading": "Rayleigh/Rician (Section 7.5)",
        "handoff_round": HANDOFF_ROUND,
        "handoff_duration": HANDOFF_DURATION,
        "handoff_drop_db": SINR_DROP_DB,
        "hospital_order": HOSPITAL_ORDER,
        "hospital_profiles": {
            hid: {
                "distance_m": HOSPITAL_PROFILES[hid]["distance_m"],
                "slice": HOSPITAL_PROFILES[hid]["slice"],
                "is_mobile": HOSPITAL_PROFILES[hid]["is_mobile"],
            }
            for hid in HOSPITAL_ORDER
        },
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Print admission summary ────────────────────────────────────────────────
    scheduler.print_summary(sim_results)

    print(f"\n[Saved]")
    print(f"  {OUTPUT_DIR}/sinr_traces.npy      — shape {sinr_arr.shape}")
    print(f"  {OUTPUT_DIR}/capacity_traces.npy  — shape {capacity_arr.shape}")
    print(f"  {OUTPUT_DIR}/admission_traces.npy — shape {admission_arr.shape}")
    print(f"  {OUTPUT_DIR}/delay_traces.npy      — shape {delay_arr.shape}")
    print(f"  {OUTPUT_DIR}/metadata.json")
    print(f"\n  Run verify_traces.py to validate output.")

    return sinr_arr, capacity_arr, admission_arr, delay_arr


if __name__ == "__main__":
    generate_traces(num_rounds=NUM_ROUNDS, seed=SEED)
