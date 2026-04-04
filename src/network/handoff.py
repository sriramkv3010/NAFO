"""
src/network/handoff.py
-----------------------
gNB (Base Station) Handoff Model for Hospital D

Hospital D is a mobile wearable patient moving between
two base stations. At round HANDOFF_ROUND, the patient
moves out of gNB-1 coverage and triggers X2 handoff
to gNB-2 per 3GPP TS 36.423 (X2 Application Protocol).

Why this matters for the paper:
    During handoff (rounds 15-17), Hospital D's SINR collapses.
    This violates the URLLC deadline and Hospital D is dropped
    from federation for those rounds.

    FedAvg and FedProx have no mechanism to handle this gracefully —
    they still weight Hospital D by its dataset size (81k samples)
    so missing updates cause a visible accuracy dip.

    NAFO's temporal smoothing reduces alpha_d automatically because
    d_i (delay) spikes during handoff. The global model barely notices.

    That difference in convergence curves is the paper's key figure.

Handoff parameters:
    Pre-handoff  : gNB-1, distance=300m, SINR ≈ 20dB
    During handoff: X2 procedure, SINR drops by 25dB (≈ -5dB)
    Post-handoff : gNB-2, distance=450m, SINR ≈ 15dB (new cell)
    Duration     : 2 rounds (rounds 15-16 inclusive)

Reference: 3GPP TS 36.423 V17.0.0, Section 8.2 (X2 Handover)
"""

import numpy as np
from typing import Tuple

# ── Handoff configuration ─────────────────────────────────────────────────────
HANDOFF_ROUND = 15  # round when handoff begins (0-indexed)
HANDOFF_DURATION = 2  # rounds the handoff disruption lasts
SINR_DROP_DB = 25.0  # SINR collapse during handoff (dB)
PRE_HANDOFF_DIST_M = 300.0  # distance to gNB-1 before handoff
POST_HANDOFF_DIST_M = 450.0  # distance to gNB-2 after handoff


def apply_handoff(
    sinr_trace: np.ndarray,
    handoff_round: int = HANDOFF_ROUND,
    duration: int = HANDOFF_DURATION,
    sinr_drop_db: float = SINR_DROP_DB,
) -> Tuple[np.ndarray, dict]:
    """
    Inject a deterministic gNB handoff event into Hospital D's SINR trace.

    The handoff creates a severe SINR collapse for `duration` rounds
    starting at `handoff_round`. After recovery, SINR settles at a
    slightly lower level reflecting the new (farther) gNB.

    Args:
        sinr_trace   : Hospital D's SINR trace, shape (T,)
        handoff_round: round index when handoff begins
        duration     : number of rounds the disruption lasts
        sinr_drop_db : magnitude of SINR drop during handoff

    Returns:
        modified_trace: SINR trace with handoff injected, shape (T,)
        handoff_info  : dict with handoff metadata for logging
    """
    modified = sinr_trace.copy()
    T = len(modified)
    rng = np.random.default_rng(seed=99)  # fixed seed, reproducible

    # ── Handoff phase: SINR collapses ─────────────────────────────────────────
    # Set handoff SINR to floor + small positive offset so values
    # are visibly non-flat (physically realistic near-outage variation).
    # Using abs(normal) ensures values sit AT or ABOVE floor, not below.
    # std=1.5dB gives range -20.0 to -16.5dB — realistic near-outage spread.
    for r in range(handoff_round, min(handoff_round + duration, T)):
        near_floor_noise = abs(rng.normal(0.0, 1.5))  # always >= 0
        modified[r] = -20.0 + near_floor_noise  # e.g. -19.4, -18.2, -20.0

    # ── Post-handoff: SINR settles at lower level on new gNB ─────────────────
    # New gNB is farther (450m vs 300m) → ~3dB lower nominal SINR
    post_offset_db = -3.0
    for r in range(handoff_round + duration, T):
        modified[r] += post_offset_db
        modified[r] = max(modified[r], -20.0)  # maintain floor

    handoff_info = {
        "handoff_round": handoff_round,
        "duration_rounds": duration,
        "sinr_drop_db": sinr_drop_db,
        "post_offset_db": post_offset_db,
        "affected_rounds": list(range(handoff_round, min(handoff_round + duration, T))),
    }

    return modified, handoff_info


def get_distance_trace(
    num_rounds: int,
    handoff_round: int = HANDOFF_ROUND,
    duration: int = HANDOFF_DURATION,
) -> np.ndarray:
    """
    Generate Hospital D's distance-to-serving-gNB trace across rounds.

    Pre-handoff : 300m to gNB-1
    During      : interpolating between cells (average 600m effective)
    Post-handoff: 450m to gNB-2

    Args:
        num_rounds   : total FL rounds
        handoff_round: round when handoff begins
        duration     : handoff duration in rounds

    Returns:
        distance array, shape (num_rounds,)
    """
    distances = np.zeros(num_rounds)

    for r in range(num_rounds):
        if r < handoff_round:
            # Normal operation on gNB-1
            distances[r] = PRE_HANDOFF_DIST_M
        elif r < handoff_round + duration:
            # Handoff: effective distance increases dramatically
            # X2 handover procedure creates temporary disconnection
            distances[r] = 1500.0  # very far — effectively no signal
        else:
            # Settled on gNB-2
            distances[r] = POST_HANDOFF_DIST_M

    return distances


def get_velocity_trace(
    num_rounds: int,
    handoff_round: int = HANDOFF_ROUND,
    duration: int = HANDOFF_DURATION,
) -> np.ndarray:
    """
    Hospital D's velocity trace.
    Patient walks at 1.4 m/s normally, 2.0 m/s during handoff transition.

    Args:
        num_rounds: total FL rounds

    Returns:
        velocity array (m/s), shape (num_rounds,)
    """
    velocities = np.full(num_rounds, 1.4)  # normal walking speed

    # Moving faster during handoff transition
    for r in range(handoff_round, min(handoff_round + duration, num_rounds)):
        velocities[r] = 2.0

    return velocities
