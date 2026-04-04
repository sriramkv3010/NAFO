"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 4 — NAFO: Network-Aware Federated Optimisation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Runs the full NAFO simulation:
    - Loads all 4 hospital datasets and checkpoints
    - Loads Phase 3 channel traces
    - Runs Flower simulation with NAFOStrategy
    - Prints convergence table comparing FedAvg vs FedProx vs NAFO
    - Shows alpha_d during handoff (key paper figure)

Triple constraint active each round:
    Synergy 1: k = f(SINR, epsilon_remaining)
    Synergy 2: C_i = clip bound per slice/modality
    Synergy 3: Hospital D solves EPL closed form

Run from project root:
    pip install simpy opacus
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python phase4_nafo/run_nafo.py

Expected result: NAFO accuracy > 85.56% (FedAvg baseline)
Key figure     : alpha_d recovers smoothly after handoff at round 15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import warnings
import logging

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

warnings.filterwarnings("ignore")
for name in ["flwr", "ray", "ray._private", "grpc"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import flwr as fl

from src.utils.device import DEVICE
from src.models.shared_head import SharedClassifierHead
from src.fl.utils import get_parameters
from src.fl.client import HospitalClient
from src.nafo.strategy import NAFOStrategy

from src.datasets.hospital_a import load_hospital_a
from src.datasets.hospital_b import load_hospital_b
from src.datasets.hospital_c import load_hospital_c
from src.datasets.hospital_d import load_hospital_d

from src.encoders.tabular_encoder import TabularEncoder
from src.encoders.signal_encoder import SignalEncoder
from src.encoders.image_encoder import ImageEncoder
from src.encoders.wearable_encoder import WearableEncoder

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUM_ROUNDS = 20
NUM_CLIENTS = 4
BATCH_SIZE = 32
TOTAL_EPSILON = 10.0
DELTA = 1e-5
LAMBDA_SMOOTH = 0.7
TRACES_DIR = "channel_traces"
HOSPITAL_ORDER = ["hospital_a", "hospital_b", "hospital_c", "hospital_d"]

DATA_PATHS = {
    "hospital_a": "data/hospital_a/processed.cleveland.data",
    "hospital_b": "data/hospital_b",
    "hospital_c": "data/hospital_c/chestmnist.npz",
    "hospital_d": "data/hospital_d",
}
CHECKPOINTS = {
    "hospital_a": "logs/hospital_a_best.pt",
    "hospital_b": "logs/hospital_b_best.pt",
    "hospital_c": "logs/hospital_c_best.pt",
    "hospital_d": "logs/hospital_d_best.pt",
}

# Phase 2 baselines for comparison table
FEDAVG_ACCURACY = 0.8556
FEDPROX_ACCURACY = 0.8482


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _load_encoder(encoder, ckpt_path, hospital_id):
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "encoder" in ckpt:
                encoder.load_state_dict(ckpt["encoder"])
                print(f"  [{hospital_id}] encoder loaded from checkpoint")
                return
        except Exception:
            pass
    print(f"  [{hospital_id}] no checkpoint — random init")


def load_all():
    print("\n[Loading hospital data and encoders]")

    tl_a, vl_a, m_a = load_hospital_a(DATA_PATHS["hospital_a"], batch_size=BATCH_SIZE)
    enc_a = TabularEncoder()
    _load_encoder(enc_a, CHECKPOINTS["hospital_a"], "hospital_a")

    tl_b, vl_b, m_b = load_hospital_b(
        DATA_PATHS["hospital_b"], batch_size=BATCH_SIZE, verbose=False
    )
    enc_b = SignalEncoder()
    _load_encoder(enc_b, CHECKPOINTS["hospital_b"], "hospital_b")

    tl_c, vl_c, m_c = load_hospital_c(DATA_PATHS["hospital_c"], batch_size=BATCH_SIZE)
    enc_c = ImageEncoder()
    _load_encoder(enc_c, CHECKPOINTS["hospital_c"], "hospital_c")

    tl_d, vl_d, m_d = load_hospital_d(DATA_PATHS["hospital_d"], batch_size=BATCH_SIZE)
    enc_d = WearableEncoder()
    _load_encoder(enc_d, CHECKPOINTS["hospital_d"], "hospital_d")

    return {
        "hospital_a": (tl_a, vl_a, enc_a, m_a),
        "hospital_b": (tl_b, vl_b, enc_b, m_b),
        "hospital_c": (tl_c, vl_c, enc_c, m_c),
        "hospital_d": (tl_d, vl_d, enc_d, m_d),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("  PHASE 4 — NAFO: Network-Aware Federated Optimisation")
    print(f"  Rounds : {NUM_ROUNDS}  |  Clients : {NUM_CLIENTS}")
    print(f"  ε total: {TOTAL_EPSILON}  |  λ smooth: {LAMBDA_SMOOTH}")
    print(f"  Device : {DEVICE}")
    print("=" * 60)

    # ── Load channel traces ────────────────────────────────────────────────────
    print("\n[Loading Phase 3 channel traces]")
    sinr_arr = np.load(os.path.join(TRACES_DIR, "sinr_traces.npy"))
    admission_arr = np.load(os.path.join(TRACES_DIR, "admission_traces.npy"))
    delay_arr = np.load(os.path.join(TRACES_DIR, "delay_traces.npy"))
    capacity_arr = np.load(os.path.join(TRACES_DIR, "capacity_traces.npy"))
    print(f"  Loaded traces: {sinr_arr.shape} — {NUM_ROUNDS} rounds, 4 hospitals")

    # Verify traces have enough rounds
    assert sinr_arr.shape[1] >= NUM_ROUNDS, (
        f"Traces have {sinr_arr.shape[1]} rounds but NUM_ROUNDS={NUM_ROUNDS}. "
        f"Regenerate traces with more rounds."
    )

    # ── Load hospital data ─────────────────────────────────────────────────────
    clients_data = load_all()

    dataset_sizes = {hid: data[3]["n_train"] for hid, data in clients_data.items()}
    print(f"\n[Dataset sizes]")
    for hid, n in dataset_sizes.items():
        print(f"  {hid}: {n:,}")

    # ── Build NAFO strategy ────────────────────────────────────────────────────
    global_head = SharedClassifierHead()
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(global_head))

    strategy = NAFOStrategy(
        initial_parameters=initial_params,
        dataset_sizes=dataset_sizes,
        sinr_traces=sinr_arr,
        admission_traces=admission_arr,
        delay_traces=delay_arr,
        capacity_traces=capacity_arr,
        hospital_order=HOSPITAL_ORDER,
        total_epsilon=TOTAL_EPSILON,
        delta=DELTA,
        lambda_smooth=LAMBDA_SMOOTH,
        min_fit_clients=4,
        min_eval_clients=4,
    )

    # ── Client factory ─────────────────────────────────────────────────────────
    fl_round_ref = [0]
    cid_map = {
        "0": "hospital_a",
        "1": "hospital_b",
        "2": "hospital_c",
        "3": "hospital_d",
    }

    def client_fn(cid: str) -> fl.client.Client:
        hid = cid_map[cid]
        tl, vl, enc, meta = clients_data[hid]
        return HospitalClient(
            hospital_id=hid,
            train_loader=tl,
            val_loader=vl,
            encoder=enc,
            meta=meta,
            fl_round_ref=fl_round_ref,
        ).to_client()

    # ── Run simulation ─────────────────────────────────────────────────────────
    print("\n[Running NAFO simulation]\n")

    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0},
        )
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr

    # ── Results ────────────────────────────────────────────────────────────────
    acc_history = dict(history.metrics_distributed.get("accuracy", []))

    print("=" * 60)
    print("  PHASE 4 RESULTS — NAFO")
    print("=" * 60)

    print(f"\n  {'Round':<8} {'NAFO Acc':<12} {'alpha_D':<10} {'Admitted':<10} Note")
    print(f"  {'-'*55}")

    for rnd in range(1, NUM_ROUNDS + 1):
        acc = acc_history.get(rnd, 0.0)

        # Get NAFO alpha_d and admission for this round
        alpha_d = 0.0
        admitted = "?"
        if rnd - 1 < len(strategy.round_logs):
            log = strategy.round_logs[rnd - 1]
            alpha_d = log["alpha"].get("hospital_d", 0.0)
            admitted = "YES" if log["admitted"].get("hospital_d", True) else "NO"

        note = ""
        if rnd in [15, 16]:
            note = "<-- handoff"
        elif rnd == 17:
            note = "<-- recovery"

        print(
            f"  Round {rnd:02d}   {acc:.4f}      {alpha_d:.4f}     {admitted:<10} {note}"
        )

    best_nafo = max(acc_history.values()) if acc_history else 0.0

    # ── Ablation comparison ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ABLATION TABLE — Phase 2 vs Phase 4")
    print(f"{'='*60}")
    print(f"  Method                  Accuracy   vs FedAvg")
    print(f"  {'-'*45}")
    print(f"  FedAvg (baseline)       {FEDAVG_ACCURACY:.4f}    —")
    print(
        f"  FedProx (mu=0.01)       {FEDPROX_ACCURACY:.4f}    "
        f"{FEDPROX_ACCURACY - FEDAVG_ACCURACY:+.4f}"
    )
    print(
        f"  NAFO (this work)        {best_nafo:.4f}    "
        f"{best_nafo - FEDAVG_ACCURACY:+.4f}"
    )
    print()

    if best_nafo > FEDAVG_ACCURACY:
        print(
            f"  NAFO beats FedAvg by {best_nafo - FEDAVG_ACCURACY:.4f} ({(best_nafo-FEDAVG_ACCURACY)*100:.2f}%)"
        )
    else:
        print(f"  NAFO does not beat FedAvg — check configuration")

    # ── ε budget summary ───────────────────────────────────────────────────────
    eps_spent = strategy.compressor.epsilon_spent
    print(f"\n  [Privacy Budget]")
    print(f"  ε total    : {TOTAL_EPSILON:.2f}")
    print(f"  ε spent    : {eps_spent:.4f}")
    print(f"  ε remaining: {strategy.compressor.epsilon_remaining:.4f}")
    print(f"  Budget used: {100*eps_spent/TOTAL_EPSILON:.1f}%")

    # ── AoI summary ───────────────────────────────────────────────────────────
    aoi = strategy.aggregator.compute_aoi()
    print(f"\n  [Age of Information — final round]")
    for hid, a in aoi.items():
        print(f"  {hid}: AoI = {a:.0f} rounds")

    print("=" * 60)


if __name__ == "__main__":
    main()
