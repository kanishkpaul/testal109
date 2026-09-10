"""Orchestrator for 6-seed primary reproduction, core ablations, and statistical analysis."""

import argparse
import csv
import json
import math
import os
import numpy as np
from src.train import train_model


def compute_statistics(values: list) -> dict:
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    # 95% CI using t-distribution approximation for small n or 1.96 / t_crit
    # For n=6, t_crit(0.025, df=5) = 2.571
    t_crit = 2.571 if n == 6 else (1.96 if n > 30 else 2.776 if n == 5 else 2.0)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    return {
        "mean": mean,
        "std": std,
        "ci_95": (ci_lower, ci_upper),
        "raw": values,
    }


def compute_paired_differences(trans_vals: list, res_vals: list) -> dict:
    diffs = np.array(res_vals) - np.array(trans_vals)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_diff = std_diff / math.sqrt(n) if n > 1 else 0.0
    t_crit = 2.571 if n == 6 else 2.0
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Bootstrap 95% CI
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(diffs, size=n, replace=True)) for _ in range(5000)]
    boot_ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_95": (ci_lower, ci_upper),
        "bootstrap_ci_95": boot_ci,
    }


def run_full_study(steps: int = 2000, seeds: list = [0, 1, 2, 3, 4, 5], output_dir: str = "results/repro"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f"RUNNING PRIMARY REPRODUCTION STUDY ({len(seeds)} SEEDS, {steps} STEPS)")
    print(f"=======================================================\n")

    # 1. Train TransformerLM across seeds
    trans_ppls = []
    trans_accs = []
    for s in seeds:
        print(f"\n>>> Running Transformer Baseline (Seed {s}) <<<")
        res = train_model(
            model_type="transformer",
            pos_encoding="learned",
            seed=s,
            steps=steps,
            batch_size=32,
            seq_len=256,
            eval_interval=500,
            output_dir=output_dir,
        )
        trans_ppls.append(res["test_ppl"])
        trans_accs.append(res["test_acc"])

    # 2. Train ResonatorLM across seeds
    res_ppls = []
    res_accs = []
    for s in seeds:
        print(f"\n>>> Running ResonatorLM Balanced (Seed {s}) <<<")
        res = train_model(
            model_type="resonator",
            preset="balanced",
            seed=s,
            steps=steps,
            batch_size=32,
            seq_len=256,
            eval_interval=500,
            output_dir=output_dir,
        )
        res_ppls.append(res["test_ppl"])
        res_accs.append(res["test_acc"])

    # 3. Core Ablations for ResonatorLM (Seeds 0, 1, 2)
    ablation_presets = ["no_coupling", "no_local", "no_both"]
    ablation_results = {}
    for preset in ablation_presets:
        ablation_results[preset] = {"ppl": [], "acc": []}
        for s in seeds[:3]:
            print(f"\n>>> Running ResonatorLM Ablation: {preset} (Seed {s}) <<<")
            res = train_model(
                model_type="resonator",
                preset=preset,
                seed=s,
                steps=steps,
                batch_size=32,
                seq_len=256,
                eval_interval=500,
                output_dir=output_dir,
            )
            ablation_results[preset]["ppl"].append(res["test_ppl"])
            ablation_results[preset]["acc"].append(res["test_acc"])

    # Compute statistics
    stat_trans_ppl = compute_statistics(trans_ppls)
    stat_trans_acc = compute_statistics(trans_accs)
    stat_res_ppl = compute_statistics(res_ppls)
    stat_res_acc = compute_statistics(res_accs)

    paired_ppl = compute_paired_differences(trans_ppls, res_ppls)
    paired_acc = compute_paired_differences(trans_accs, res_accs)

    summary = {
        "steps": steps,
        "seeds": seeds,
        "transformer": {
            "ppl": stat_trans_ppl,
            "acc": stat_trans_acc,
        },
        "resonator": {
            "ppl": stat_res_ppl,
            "acc": stat_res_acc,
        },
        "paired_comparison": {
            "ppl_delta": paired_ppl,
            "acc_delta": paired_acc,
            "relative_ppl_reduction_pct": ((stat_trans_ppl["mean"] - stat_res_ppl["mean"]) / stat_trans_ppl["mean"]) * 100.0,
            "accuracy_gain_points": stat_res_acc["mean"] - stat_trans_acc["mean"],
        },
        "ablations": {
            k: {
                "ppl": compute_statistics(v["ppl"]),
                "acc": compute_statistics(v["acc"]),
            }
            for k, v in ablation_results.items()
        },
    }

    summary_path = os.path.join(output_dir, "statistical_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("STATISTICAL REPRODUCTION SUMMARY")
    print("=======================================================")
    print(f"Transformer PPL : {stat_trans_ppl['mean']:.3f} +/- {stat_trans_ppl['std']:.3f} (95% CI: [{stat_trans_ppl['ci_95'][0]:.3f}, {stat_trans_ppl['ci_95'][1]:.3f}])")
    print(f"ResonatorLM PPL : {stat_res_ppl['mean']:.3f} +/- {stat_res_ppl['std']:.3f} (95% CI: [{stat_res_ppl['ci_95'][0]:.3f}, {stat_res_ppl['ci_95'][1]:.3f}])")
    print(f"Relative PPL Reduction: {summary['paired_comparison']['relative_ppl_reduction_pct']:.2f}%")
    print(f"Transformer Acc : {stat_trans_acc['mean']:.2f}% +/- {stat_trans_acc['std']:.2f}%")
    print(f"ResonatorLM Acc : {stat_res_acc['mean']:.2f}% +/- {stat_res_acc['std']:.2f}%")
    print(f"Accuracy Gain   : +{summary['paired_comparison']['accuracy_gain_points']:.2f} percentage points")
    print("=======================================================\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--output_dir", type=str, default="results/repro")
    args = parser.parse_args()

    run_full_study(steps=args.steps, seeds=args.seeds, output_dir=args.output_dir)
