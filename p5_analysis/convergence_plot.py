"""
phase5_analysis/convergence_plot.py
-------------------------------------
Figure 1: Convergence curves — FedAvg vs FedProx vs NAFO
Figure 2: Alpha_D temporal smoothing during gNB handoff

Both figures are the paper's primary empirical contribution.

Run from project root:
    python phase5_analysis/convergence_plot.py

Outputs:
    figures/fig1_convergence.pdf
    figures/fig2_handoff_alpha.pdf
    figures/fig1_convergence.png
    figures/fig2_handoff_alpha.png
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs("figures", exist_ok=True)

# ── Experimental results (paste your actual results here) ─────────────────────

# FedAvg — 10 rounds, all 4 hospitals with checkpoints
FEDAVG_ACC = [
    0.7563,
    0.7632,
    0.7616,
    0.7552,
    0.7549,
    0.7553,
    0.7594,
    0.7645,
    0.7673,
    0.7703,
]

# FedProx mu=0.01 — 10 rounds, all checkpoints
FEDPROX_ACC = [
    0.8227,
    0.8239,
    0.8303,
    0.8360,
    0.8376,
    0.8389,
    0.8403,
    0.8433,
    0.8464,
    0.8482,
]

# NAFO — 20 rounds
NAFO_ACC = [
    0.8614,
    0.8556,
    0.8523,
    0.8482,
    0.8452,
    0.8491,
    0.8490,
    0.8486,
    0.8449,
    0.8419,
    0.8462,
    0.8471,
    0.8477,
    0.8438,
    0.8400,
    0.8385,
    0.8358,
    0.8320,
    0.8303,
    0.8243,
]

# NAFO alpha_D over 20 rounds
NAFO_ALPHA_D = [
    0.3844,
    0.4523,
    0.4373,
    0.4277,
    0.4218,
    0.4617,
    0.5698,
    0.5122,
    0.4834,
    0.4631,
    0.4928,
    0.4699,
    0.4537,
    0.4422,
    0.4930,
    0.3451,
    0.2416,
    0.2941,
    0.3310,
    0.4153,
]

HANDOFF_ROUNDS = [16, 17]  # 1-indexed rounds where Hospital D is dropped


def style():
    """IEEE-compatible plot style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 1 — Convergence curves
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_convergence():
    style()
    fig, ax = plt.subplots(figsize=(6, 4))

    rounds_fedavg = list(range(1, len(FEDAVG_ACC) + 1))
    rounds_fedprox = list(range(1, len(FEDPROX_ACC) + 1))
    rounds_nafo = list(range(1, len(NAFO_ACC) + 1))

    ax.plot(
        rounds_fedavg,
        FEDAVG_ACC,
        color="#2166ac",
        linestyle="--",
        marker="o",
        markevery=2,
        label="FedAvg",
    )

    ax.plot(
        rounds_fedprox,
        FEDPROX_ACC,
        color="#4dac26",
        linestyle="-.",
        marker="s",
        markevery=2,
        label=r"FedProx ($\mu$=0.01)",
    )

    ax.plot(
        rounds_nafo,
        NAFO_ACC,
        color="#d6604d",
        linestyle="-",
        marker="^",
        markevery=2,
        label="NAFO (ours)",
    )

    # Handoff annotation
    for r in HANDOFF_ROUNDS:
        ax.axvline(x=r, color="gray", linestyle=":", alpha=0.7, linewidth=1.0)
    ax.annotate(
        "gNB handoff\n(Hospital D)",
        xy=(HANDOFF_ROUNDS[0], 0.840),
        xytext=(HANDOFF_ROUNDS[0] + 0.5, 0.825),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        color="gray",
    )

    # Baseline reference lines
    ax.axhline(
        y=max(FEDAVG_ACC), color="#2166ac", linestyle=":", alpha=0.5, linewidth=1.0
    )
    ax.axhline(
        y=max(FEDPROX_ACC), color="#4dac26", linestyle=":", alpha=0.5, linewidth=1.0
    )
    ax.axhline(
        y=max(NAFO_ACC), color="#d6604d", linestyle=":", alpha=0.5, linewidth=1.0
    )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Accuracy")
    ax.set_title(
        "Convergence: FedAvg vs FedProx vs NAFO\n"
        "Multi-modal Cardiac FL over 5G (4 hospitals)"
    )
    ax.legend(loc="lower left")
    ax.set_xlim(1, max(len(NAFO_ACC), len(FEDPROX_ACC)))
    ax.set_ylim(0.72, 0.90)

    # Annotate best accuracy per method
    ax.annotate(
        f"FedAvg: {max(FEDAVG_ACC):.4f}",
        xy=(len(FEDAVG_ACC), max(FEDAVG_ACC)),
        xytext=(len(FEDAVG_ACC) - 3, max(FEDAVG_ACC) + 0.004),
        fontsize=7,
        color="#2166ac",
    )
    ax.annotate(
        f"FedProx: {max(FEDPROX_ACC):.4f}",
        xy=(len(FEDPROX_ACC), max(FEDPROX_ACC)),
        xytext=(len(FEDPROX_ACC) - 4, max(FEDPROX_ACC) + 0.004),
        fontsize=7,
        color="#4dac26",
    )
    ax.annotate(
        f"NAFO: {max(NAFO_ACC):.4f}",
        xy=(1, max(NAFO_ACC)),
        xytext=(2, max(NAFO_ACC) + 0.004),
        fontsize=7,
        color="#d6604d",
    )

    fig.tight_layout()
    fig.savefig("figures/fig1_convergence.pdf", bbox_inches="tight")
    fig.savefig("figures/fig1_convergence.png", bbox_inches="tight")
    print("[Fig 1] Saved: figures/fig1_convergence.pdf")
    plt.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 2 — Alpha_D temporal smoothing during handoff
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_handoff_alpha():
    style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    rounds = list(range(1, len(NAFO_ACC) + 1))

    # FedAvg hypothetical alpha_D (static n_frac = 0.384)
    fedavg_alpha_d = [0.384] * len(NAFO_ACC)
    # During dropped rounds, FedAvg redistributes: alpha_D effectively = 0
    # then snaps back next round
    fedavg_alpha_hypothetical = list(fedavg_alpha_d)
    for r in HANDOFF_ROUNDS:
        fedavg_alpha_hypothetical[r - 1] = 0.0  # snaps to zero
    fedavg_alpha_hypothetical[HANDOFF_ROUNDS[-1]] = 0.384  # snaps back

    # Plot alpha_D
    ax1.plot(
        rounds,
        NAFO_ALPHA_D,
        color="#d6604d",
        linewidth=1.8,
        marker="^",
        markevery=1,
        markersize=4,
        label="NAFO alpha_D",
    )
    ax1.step(
        rounds,
        fedavg_alpha_hypothetical,
        color="#2166ac",
        linestyle="--",
        linewidth=1.5,
        where="mid",
        label="FedAvg alpha_D (static → snap)",
    )

    # Handoff shading
    for r in HANDOFF_ROUNDS:
        ax1.axvspan(r - 0.5, r + 0.5, alpha=0.15, color="gray")

    ax1.axhline(
        y=0.384,
        color="black",
        linestyle=":",
        alpha=0.4,
        linewidth=1.0,
        label="FedAvg static baseline",
    )
    ax1.set_ylabel(r"$\alpha_D$ (Hospital D weight)")
    ax1.set_title("NAFO Temporal Smoothing During gNB Handoff")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(0, 0.65)

    # Annotate handoff region
    ax1.annotate(
        "Handoff\n(dropped)",
        xy=(16.5, 0.05),
        ha="center",
        fontsize=8,
        color="gray",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="gray"
        ),
    )

    # Plot accuracy during handoff region
    ax2.plot(
        rounds,
        NAFO_ACC,
        color="#d6604d",
        linewidth=1.8,
        marker="^",
        markevery=2,
        markersize=4,
        label="NAFO",
    )
    ax2.axhline(
        y=max(FEDAVG_ACC),
        color="#2166ac",
        linestyle="--",
        linewidth=1.5,
        label="FedAvg peak",
    )

    for r in HANDOFF_ROUNDS:
        ax2.axvspan(r - 0.5, r + 0.5, alpha=0.15, color="gray")

    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Global Accuracy")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.set_ylim(0.80, 0.88)

    fig.tight_layout()
    fig.savefig("figures/fig2_handoff_alpha.pdf", bbox_inches="tight")
    fig.savefig("figures/fig2_handoff_alpha.png", bbox_inches="tight")
    print("[Fig 2] Saved: figures/fig2_handoff_alpha.pdf")
    plt.close()


if __name__ == "__main__":
    plot_convergence()
    plot_handoff_alpha()
    print("\nAll convergence figures saved to figures/")
