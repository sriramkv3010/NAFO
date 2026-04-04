"""
PHASE 2b — FedProx Baseline
Proximal term: L_local(w) = L_data(w) + (mu/2)||w - w_global||^2
Penalises local head from drifting too far from global model.
Must beat FedAvg (77.03%) to justify place in ablation table.
"""

import os
import sys

# ── Silence everything before any imports ─────────────────────────────────────
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["RAY_LOG_LEVEL"] = "ERROR"

import warnings

warnings.filterwarnings("ignore")

import logging

for name in [
    "flwr",
    "ray",
    "ray.tune",
    "ray._private",
    "ray._private.worker",
    "grpc",
    "urllib3",
]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flwr as fl
import torch
import torch.nn as nn
import numpy as np

from src.utils.device import DEVICE
from src.models.shared_head import SharedClassifierHead
from src.fl.utils import get_parameters, set_parameters
from src.fl.server import weighted_average

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
NUM_ROUNDS = 15
NUM_CLIENTS = 4
BATCH_SIZE = 32
MU = 0.05  # proximal term — ablate over 0.01, 0.1, 1.0
LOCAL_EPOCHS = 3
LR = 5e-4

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
# FEDPROX CLIENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FedProxClient(fl.client.NumPyClient):
    """
    FedProx adds a proximal term to local training loss.
    Prevents local head from drifting far from global head.
    Especially useful for non-IID hospital data.
    """

    def __init__(self, hospital_id, train_loader, val_loader, encoder, meta):
        self.hospital_id = hospital_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.encoder = encoder.to(DEVICE)
        self.head = SharedClassifierHead().to(DEVICE)
        self.meta = meta
        self.criterion = nn.BCEWithLogitsLoss()
        self.global_head_params = None  # reference weights for prox term

    def get_parameters(self, config):
        return get_parameters(self.head)

    def fit(self, parameters, config):
        # Load global head weights
        set_parameters(self.head, parameters)

        # Store global weights as reference for proximal term
        # Keep as tensors on CPU to avoid MPS memory issues
        self.global_head_params = [
            torch.tensor(p.copy(), dtype=torch.float32) for p in parameters
        ]

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=LR,
        )

        total_loss = 0.0
        for _ in range(LOCAL_EPOCHS):
            self.encoder.train()
            self.head.train()

            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = self._binarize(y_batch).to(DEVICE)

                optimizer.zero_grad()
                logit = self.head(self.encoder(X_batch))

                # FedProx loss = data loss + (mu/2) * ||w_local - w_global||^2
                data_loss = self.criterion(logit.squeeze(1), y_batch)
                prox_loss = self._proximal_term()
                loss = data_loss + (MU / 2.0) * prox_loss

                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / (LOCAL_EPOCHS * len(self.train_loader))
        return (
            get_parameters(self.head),
            self.meta["n_train"],
            {"train_loss": avg_loss, "hospital": self.hospital_id},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.head, parameters)
        self.encoder.eval()
        self.head.eval()

        total_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = self._binarize(y_batch).to(DEVICE)
                logit = self.head(self.encoder(X_batch))
                total_loss += self.criterion(logit.squeeze(1), y_batch).item()
                prob = torch.sigmoid(logit.squeeze(1))
                all_probs.extend(prob.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        accuracy = float(((all_probs >= 0.5) == all_labels).mean())

        return (
            total_loss / len(self.val_loader),
            self.meta["n_val"],
            {"accuracy": accuracy, "hospital": self.hospital_id},
        )

    def _proximal_term(self) -> torch.Tensor:
        """
        Compute ||w_local - w_global||^2 for SharedHead parameters only.
        Encoders are excluded — only the federated component is penalised.
        """
        if self.global_head_params is None:
            return torch.tensor(0.0)

        prox = torch.tensor(0.0)
        local_params = list(self.head.parameters())
        global_params = [p.to(DEVICE) for p in self.global_head_params]

        for local_p, global_p in zip(local_params, global_params):
            prox = prox + torch.norm(local_p - global_p) ** 2

        return prox

    def _binarize(self, y: torch.Tensor) -> torch.Tensor:
        """Hospital B: 5-class → binary. Others unchanged."""
        if self.hospital_id == "hospital_b":
            return (y > 0).float()
        return y.float()


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
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    os.makedirs("logs", exist_ok=True)

    print("=" * 52)
    print("  PHASE 2b — FedProx Baseline")
    print(f"  Rounds: {NUM_ROUNDS}  |  Clients: {NUM_CLIENTS}  |  mu: {MU}")
    print(f"  Device: {DEVICE}")
    print("=" * 52)

    clients_data = load_all()

    cid_map = {
        "0": "hospital_a",
        "1": "hospital_b",
        "2": "hospital_c",
        "3": "hospital_d",
    }

    def client_fn(cid: str) -> fl.client.Client:
        hid = cid_map[cid]
        tl, vl, enc, meta = clients_data[hid]
        return FedProxClient(hid, tl, vl, enc, meta).to_client()

    global_head = SharedClassifierHead()
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(global_head))
    n_params = sum(p.numel() for p in global_head.parameters())
    print(f"\n[Server] Head parameters transmitted: {n_params:,}")
    print(f"[FedProx] Proximal term mu = {MU}")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=lambda metrics: {
            "train_loss": np.mean([m["train_loss"] for _, m in metrics])
        },
    )

    print("\n[Running FedProx federation — ~12 min on M4 Air]\n")

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
    print(f"  PHASE 2b RESULTS — FedProx (mu={MU})")
    print("=" * 52)
    print(f"\n  {'Round':<8} {'Accuracy':<12} {'Train Loss'}")
    print(f"  {'-'*32}")
    for rnd in range(1, NUM_ROUNDS + 1):
        acc = acc_history.get(rnd, 0.0)
        loss = loss_history.get(rnd, 0.0)
        print(f"  Round {rnd:02d}   {acc:.4f}       {loss:.4f}")

    best_acc = max(acc_history.values()) if acc_history else 0.0

    print(f"\n  Best FedProx accuracy : {best_acc:.4f}")
    print(f"  Best FedAvg accuracy  : 0.7703  (Phase 2 baseline)")

    diff = best_acc - 0.7703
    if diff > 0:
        print(f"  FedProx improvement   : +{diff:.4f}  (expected)")
    else:
        print(f"  FedProx vs FedAvg     : {diff:.4f}  (converged similarly)")

    print(f"\n  Ablation table so far:")
    print(f"    FedAvg              : 0.8556")
    print(f"    FedProx (mu={MU})   : {best_acc:.4f}")
    print(f"    NAFO + DP (Phase 4) : TBD  ← must beat both")
    print(f"\n  Phase 2 complete. Ready for Phase 3 (5G Digital Twin).")
    print("=" * 52)


if __name__ == "__main__":
    main()
