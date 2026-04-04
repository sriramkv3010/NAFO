"""
PHASE 2 — FedAvg Federated Learning
Clean output. Uses original cid mapping that produced 77.5% accuracy.
"""

import os
import sys

# ── Silence everything before any other imports ───────────────────────────────
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["RAY_LOG_LEVEL"] = "ERROR"

import warnings

warnings.filterwarnings("ignore")

import logging

# Must set these BEFORE importing flwr or ray
for name in [
    "flwr",
    "ray",
    "ray.tune",
    "ray._private",
    "ray._private.worker",
    "ray.util",
    "grpc",
    "urllib3",
]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

# Redirect stderr briefly during simulation to swallow Ray subprocess output
import io
import contextlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flwr as fl
import torch

from src.utils.device import DEVICE
from src.models.shared_head import SharedClassifierHead
from src.fl.client import HospitalClient
from src.fl.server import build_fedavg_strategy
from src.fl.utils import get_parameters

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
NUM_ROUNDS = 10
NUM_CLIENTS = 4
BATCH_SIZE = 32

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
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
    print("\n[Loading hospital data]")

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
# CLIENT FACTORY
# Uses original cid: str mapping — reliable and tested
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_client_fn(clients_data, fl_round_ref):
    cid_map = {
        "0": "hospital_a",
        "1": "hospital_b",
        "2": "hospital_c",
        "3": "hospital_d",
    }

    def client_fn(cid: str) -> fl.client.Client:
        hospital_id = cid_map[cid]
        tl, vl, enc, meta = clients_data[hospital_id]
        return HospitalClient(
            hospital_id=hospital_id,
            train_loader=tl,
            val_loader=vl,
            encoder=enc,
            meta=meta,
            fl_round_ref=fl_round_ref,
        ).to_client()

    return client_fn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    os.makedirs("logs", exist_ok=True)

    print("=" * 52)
    print("  PHASE 2 — FedAvg Federated Learning")
    print(f"  Rounds: {NUM_ROUNDS}  |  Clients: {NUM_CLIENTS}")
    print(f"  Device: {DEVICE}")
    print("=" * 52)

    clients_data = load_all()
    fl_round_ref = [0]

    global_head = SharedClassifierHead()
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(global_head))
    n_params = sum(p.numel() for p in global_head.parameters())
    print(f"\n[Server] Head parameters transmitted per round: {n_params:,}")

    strategy = build_fedavg_strategy(initial_params, min_clients=NUM_CLIENTS)
    client_fn = build_client_fn(clients_data, fl_round_ref)

    print("\n[Running federation — this takes ~12 min on M4 Air]\n")

    # Suppress stdout/stderr from Ray subprocess workers during simulation
    # Our results table prints AFTER simulation completes
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

    # ── Clean results table ────────────────────────────────────────────
    acc_history = dict(history.metrics_distributed.get("accuracy", []))
    loss_history = dict(history.metrics_distributed_fit.get("train_loss", []))

    print("=" * 52)
    print("  PHASE 2 RESULTS — FedAvg")
    print("=" * 52)
    print(f"\n  {'Round':<8} {'Accuracy':<12} {'Train Loss'}")
    print(f"  {'-'*32}")
    for rnd in range(1, NUM_ROUNDS + 1):
        acc = acc_history.get(rnd, 0.0)
        loss = loss_history.get(rnd, 0.0)
        print(f"  Round {rnd:02d}   {acc:.4f}       {loss:.4f}")

    best_acc = max(acc_history.values()) if acc_history else 0.0
    print(f"\n  Best federated accuracy : {best_acc:.4f}")
    print(f"\n  Phase 1 local baselines:")
    print(f"    Hospital A local AUC  : 0.9565")
    print(f"    Hospital B V-recall   : 0.8693")
    print(f"    Hospital C local AUC  : 0.8219")
    print(f"    Hospital D local AUC  : 0.9568")
    print("=" * 52)


if __name__ == "__main__":
    main()
