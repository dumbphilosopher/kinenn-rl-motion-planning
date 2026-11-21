"""Record a high-res MuJoCo rollout GIF of KineNN-Informed PPO on the ABB IRB 1200 Welding Cell."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import PPO

from kinenn_rl.env import ABBIrb1200WeldingEnv
from kinenn_rl.kinenn import KineNNHTM

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record MuJoCo demo GIF.")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "runs" / "ppo_kinenn_s42")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "demo.gif")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--filter-steps", type=int, default=35)
    args = parser.parse_args()

    env = ABBIrb1200WeldingEnv(render_mode="rgb_array", render_width=640, render_height=480)
    model = PPO.load(args.run_dir / "model.zip")

    kinenn = KineNNHTM(perturbation_std=0.0)
    dh_path = args.run_dir / "kinenn_dh.pt"
    if dh_path.exists():
        kinenn.load_state_dict(torch.load(dh_path, map_location="cpu", weights_only=True))

    frames: list[np.ndarray] = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=100 + ep)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        for step in range(env.max_steps):
            if step < args.filter_steps:
                q = env.data.qpos[:6].copy()
                base_pos = env.data.xpos[env.model.body("robot_base").id]
                action = kinenn.compute_ik_action(q, env.target_pos - base_pos, env.max_delta_q)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            if step % 2 == 0 or terminated or truncated:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            if terminated or truncated:
                # Pause briefly on terminal frame so viewer sees target reached
                for _ in range(8):
                    if frame is not None:
                        frames.append(frame)
                print(
                    f"Episode {ep+1}: steps={step+1}, success={info['success']}, "
                    f"collision={info['collision']}, dist={info['distance_error']:.4f}m"
                )
                break

    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=15, loop=0)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
