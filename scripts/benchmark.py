"""Run 4-way benchmark (Native vs KineNN for PPO & SAC) and generate Thesis-style plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import REPO_ROOT, run_training


def _load_metrics(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps, rewards, collisions, successes = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["timestep"]))
            rewards.append(float(row["reward"]))
            collisions.append(int(row["collision"]))
            successes.append(int(row["success"]))
    return np.array(steps), np.array(rewards), np.array(collisions), np.array(successes)


def _moving_average(x: np.ndarray, window: int = 200) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


def plot_benchmark(runs: list[tuple[str, Path]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=160)
    axes_flat = axes.ravel()

    for ax, (label, csv_path) in zip(axes_flat, runs):
        steps, rewards, collisions, successes = _load_metrics(csv_path)
        smoothed = _moving_average(rewards, window=min(250, max(10, len(rewards) // 25)))
        ax.plot(steps, smoothed, color="#1f77b4", linewidth=1.6, label=f"{label} (Smoothed Reward)")

        coll_steps = steps[collisions == 1]
        succ_steps = steps[successes == 1]
        if len(coll_steps) > 0:
            ax.scatter(coll_steps, np.zeros_like(coll_steps), color="#d62728", s=8, marker="*", label="Collision", zorder=4)
        if len(succ_steps) > 0:
            y_mark = np.full_like(succ_steps, max(smoothed.max() * 0.25, 0.005), dtype=np.float64)
            ax.scatter(succ_steps, y_mark, color="#d4a017", s=10, marker="*", label="Success", zorder=5)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Timesteps (Cycles)")
        ax.set_ylabel("Step Reward R(t)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "ABB IRB 1200 Welding Station — Native vs. KineNN-Informed RL (PPO & SAC)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4-way Native vs. KineNN benchmark.")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configs = [
        ("Native PPO", REPO_ROOT / "configs" / "ppo_native.yaml"),
        ("KineNN-Informed PPO", REPO_ROOT / "configs" / "ppo_kinenn.yaml"),
        ("Native SAC", REPO_ROOT / "configs" / "sac_native.yaml"),
        ("KineNN-Informed SAC", REPO_ROOT / "configs" / "sac_kinenn.yaml"),
    ]

    summaries = []
    plot_inputs = []
    for label, cfg_path in configs:
        print(f"=== Running {label} ({args.timesteps} steps) ===")
        s = run_training(cfg_path, total_timesteps=args.timesteps, seed=args.seed)
        s["label"] = label
        summaries.append(s)
        plot_inputs.append((label, Path(s["run_dir"]) / "metrics.csv"))
        print(
            f"  Episodes: {s['episodes']} | Successes: {s['successes']} ({s['success_rate']*100:.1f}%) "
            f"| Collisions: {s['collisions']} ({s['collision_rate']*100:.1f}%) "
            f"| Mean Reward: {s['mean_reward']:.5f}"
        )

    out_png = REPO_ROOT / "assets" / "benchmark_comparison.png"
    plot_benchmark(plot_inputs, out_png)

    out_json = REPO_ROOT / "runs" / "benchmark_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nSaved comparison plot to: {out_png}")
    print("\n| Method | Episodes | Successes | Success Rate | Collisions | Collision Rate | Mean Reward |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        print(
            f"| {s['label']} | {s['episodes']} | {s['successes']} | {s['success_rate']*100:.1f}% "
            f"| {s['collisions']} | {s['collision_rate']*100:.1f}% | {s['mean_reward']:.5f} |"
        )


if __name__ == "__main__":
    main()
