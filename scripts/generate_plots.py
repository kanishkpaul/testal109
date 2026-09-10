"""Generates publication-quality scientific plots from raw experimental data."""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_context_sweep(csv_path: str = "results/context_sweep_results.csv", out_dir: str = "results/figures"):
    if not os.path.exists(csv_path):
        return
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    models = set(r["model"] for r in rows)
    contexts = sorted(list(set(int(r["context"]) for r in rows)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for m in models:
        m_rows = [r for r in rows if r["model"] == m]
        m_rows.sort(key=lambda x: int(x["context"]))
        xs = [int(r["context"]) for r in m_rows]
        ppls = [float(r["ppl"]) for r in m_rows]
        accs = [float(r["acc"]) for r in m_rows]

        marker = "o" if "Resonator" in m else "s"
        color = "#2b5c8f" if "Resonator" in m else "#d95f02"

        ax1.plot(xs, ppls, marker=marker, color=color, linewidth=2, label=m)
        ax2.plot(xs, accs, marker=marker, color=color, linewidth=2, label=m)

    ax1.set_title("Perplexity vs Context Length (Trained on 256)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Context Length (Tokens)")
    ax1.set_ylabel("Test Perplexity (Lower is Better)")
    ax1.set_xticks(contexts)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.set_title("Next-Char Accuracy vs Context Length", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Context Length (Tokens)")
    ax2.set_ylabel("Test Accuracy % (Higher is Better)")
    ax2.set_xticks(contexts)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    out_file = os.path.join(out_dir, "context_sweep.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved plot: {out_file}")


def plot_memory_scaling(out_dir: str = "results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    contexts = [2048, 4096, 8192, 16384, 32768]
    # In KiB for float32
    trans_kv = [3968.0, 7936.0, 15872.0, 31744.0, 63488.0]
    res_state = [2.0, 2.0, 2.0, 2.0, 2.0]

    plt.figure(figsize=(8, 5))
    plt.plot(contexts, trans_kv, marker="s", color="#d95f02", linewidth=2.5, label="Transformer KV Cache (Linear O(T))")
    plt.plot(contexts, res_state, marker="o", color="#2b5c8f", linewidth=2.5, label="ResonatorLM Recurrent State (Constant O(1))")

    plt.title("State Memory Footprint vs Context Length", fontsize=13, fontweight="bold")
    plt.xlabel("Context Length (Tokens)", fontsize=11)
    plt.ylabel("Memory Footprint (KiB, log scale)", fontsize=11)
    plt.yscale("log")
    plt.xticks(contexts, [f"{c//1024}K" for c in contexts])
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "memory_scaling.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved plot: {out_file}")


def plot_half_life_decay(out_dir: str = "results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    t = np.linspace(0, 10000, 1000)
    half_lives = [2, 16, 64, 256, 1024, 2048]

    plt.figure(figsize=(9, 5))
    for hl in half_lives:
        amp = 2.0 ** (-t / hl)
        plt.plot(t, amp, label=f"$t_{{1/2}} = {hl}$ tokens", linewidth=2)

    plt.title("Learned Mode Impulse Decay Across Token Distance", fontsize=13, fontweight="bold")
    plt.xlabel("Token Distance (Lags)", fontsize=11)
    plt.ylabel("Remaining Amplitude ($2^{-t/t_{1/2}}$)", fontsize=11)
    plt.axvline(2048, color="black", linestyle=":", alpha=0.6, label="Max Half-Life (2048)")
    plt.axvline(8192, color="gray", linestyle="--", alpha=0.6, label="8K Context")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "half_life_decay.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved plot: {out_file}")


if __name__ == "__main__":
    plot_context_sweep()
    plot_memory_scaling()
    plot_half_life_decay()
