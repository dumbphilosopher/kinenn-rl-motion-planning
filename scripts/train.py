"""Train Native or KineNN-Informed PPO / SAC on the ABB IRB 1200 Welding Station."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from kinenn_rl.agents import KineNNPPO, KineNNSAC, MetricsCallback
from kinenn_rl.env import ABBIrb1200WeldingEnv
from kinenn_rl.kinenn import KineNNHTM

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_training(
    config_path: Path,
    total_timesteps: int | None = None,
    seed: int | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    algo = cfg["algo"].lower()
    mode = cfg["mode"].lower()
    timesteps = int(total_timesteps if total_timesteps is not None else cfg["total_timesteps"])
    run_seed = int(seed if seed is not None else cfg.get("seed", 42))

    run_name = f"{algo}_{mode}_s{run_seed}"
    save_dir = (out_dir or (REPO_ROOT / "runs")) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = cfg.get("env", {})
    vec_env = DummyVecEnv([lambda: Monitor(ABBIrb1200WeldingEnv(**env_cfg))])

    hp = dict(cfg.get("hyperparams", {}))
    hp["seed"] = run_seed
    hp["device"] = "cpu"

    kinenn_model: KineNNHTM | None = None
    if mode == "kinenn":
        kcfg = cfg.get("kinenn", {})
        kinenn_model = KineNNHTM(
            perturbation_std=float(kcfg.get("perturbation_std", 0.015)),
            ik_damping=float(kcfg.get("ik_damping", 0.08)),
            ik_gain=float(kcfg.get("ik_gain", 0.85)),
            seed=run_seed,
        )
        cls = KineNNPPO if algo == "ppo" else KineNNSAC
        model = cls(
            "MlpPolicy",
            vec_env,
            kinenn=kinenn_model,
            filter_steps=int(kcfg.get("filter_steps", 35)),
            dh_update_freq=int(kcfg.get("dh_update_freq", 32)),
            verbose=0,
            **hp,
        )
    else:
        cls = PPO if algo == "ppo" else SAC
        model = cls("MlpPolicy", vec_env, verbose=0, **hp)

    metrics_cb = MetricsCallback(log_csv_path=save_dir / "metrics.csv")
    model.learn(total_timesteps=timesteps, callback=metrics_cb)

    model.save(save_dir / "model.zip")
    if kinenn_model is not None:
        torch.save(kinenn_model.state_dict(), save_dir / "kinenn_dh.pt")

    summary = metrics_cb.summary()
    summary.update({"algo": algo, "mode": mode, "timesteps": timesteps, "seed": run_seed, "run_dir": str(save_dir)})
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    vec_env.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Native or KineNN-Informed PPO/SAC.")
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True)
    parser.add_argument("--mode", choices=["native", "kinenn"], required=True)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg_path = REPO_ROOT / "configs" / f"{args.algo}_{args.mode}.yaml"
    summary = run_training(cfg_path, total_timesteps=args.timesteps, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
