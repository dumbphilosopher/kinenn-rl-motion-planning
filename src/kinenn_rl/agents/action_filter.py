"""Timestep Action Sampling Filter (Chapter 4, Figures 4.1-4.3) & Metrics Callback."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from kinenn_rl.kinenn.kinenn_htm import KineNNHTM


class ActionSamplingFilter:
    """Routes actions between KineNN (steps n < t_filter) and RL Actor (steps n >= t_filter).

    Matches Thesis Section 4.1 & Algorithms 4-5:
    - For the initial `filter_steps` (t) timesteps of each episode, selects a_k(t) from KineNN.
    - For the remaining (T - t) timesteps, selects a_ac(t) from the RL actor policy.
    - Maintains a small buffer of observed (q, eef_pos_base) pairs to update KineNN's
      learnable DH parameters online via backpropagation.
    """

    def __init__(
        self,
        kinenn: KineNNHTM,
        filter_steps: int = 40,
        dh_update_freq: int = 32,
        dh_batch_size: int = 64,
        dh_buffer_capacity: int = 2000,
    ) -> None:
        self.kinenn = kinenn
        self.filter_steps = int(filter_steps)
        self.dh_update_freq = int(dh_update_freq)
        self.dh_batch_size = int(dh_batch_size)
        self._q_buf: deque[np.ndarray] = deque(maxlen=dh_buffer_capacity)
        self._p_buf: deque[np.ndarray] = deque(maxlen=dh_buffer_capacity)
        self._total_transitions = 0
        self.last_dh_loss: float | None = None

    def use_kinenn(self, episode_step: int) -> bool:
        """Return True if step n < t_filter (use KineNN), False otherwise (use RL actor)."""
        return int(episode_step) < self.filter_steps

    def select_action(
        self,
        episode_step: int,
        actor_action: np.ndarray,
        q: np.ndarray,
        target_pos_base: np.ndarray,
        max_delta_q: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Select either KineNN action or Actor action according to the timestep filter."""
        if self.use_kinenn(episode_step):
            a_k = self.kinenn.compute_ik_action(q, target_pos_base, max_delta_q)
            return a_k, True
        return np.asarray(actor_action, dtype=np.float32), False

    def record_transition(self, q: np.ndarray, eef_pos_base: np.ndarray) -> float | None:
        """Record (q, eef_pos_base) and periodically update KineNN's learnable DH parameters."""
        self._q_buf.append(np.asarray(q, dtype=np.float32).copy())
        self._p_buf.append(np.asarray(eef_pos_base, dtype=np.float32).copy())
        self._total_transitions += 1

        if (
            self._total_transitions % self.dh_update_freq == 0
            and len(self._q_buf) >= self.dh_batch_size
        ):
            idx = np.random.choice(len(self._q_buf), size=self.dh_batch_size, replace=False)
            q_batch = np.stack([self._q_buf[i] for i in idx], axis=0)
            p_batch = np.stack([self._p_buf[i] for i in idx], axis=0)
            self.last_dh_loss = self.kinenn.update_world_model(q_batch, p_batch)
            return self.last_dh_loss
        return None


class MetricsCallback(BaseCallback):
    """Logs per-step reward, collision events, and success events for Thesis-style plots."""

    def __init__(self, log_csv_path: str | Path | None = None, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.log_csv_path = Path(log_csv_path) if log_csv_path is not None else None
        self.records: list[dict[str, Any]] = []
        self.total_episodes = 0
        self.total_collisions = 0
        self.total_successes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            reward = float(rewards[i]) if i < len(rewards) else 0.0
            collision = bool(info.get("collision", False))
            success = bool(info.get("success", False))
            done = bool(dones[i]) if i < len(dones) else False

            if done:
                self.total_episodes += 1
                if collision:
                    self.total_collisions += 1
                if success:
                    self.total_successes += 1

            self.records.append(
                {
                    "timestep": int(self.num_timesteps),
                    "reward": reward,
                    "collision": int(collision),
                    "success": int(success),
                    "distance_error": float(info.get("distance_error", 0.0)),
                }
            )
        return True

    def _on_training_end(self) -> None:
        if self.log_csv_path is not None:
            self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["timestep", "reward", "collision", "success", "distance_error"]
                )
                writer.writeheader()
                writer.writerows(self.records)

    def summary(self) -> dict[str, Any]:
        eps = max(self.total_episodes, 1)
        rewards = [r["reward"] for r in self.records] or [0.0]
        return {
            "episodes": self.total_episodes,
            "successes": self.total_successes,
            "collisions": self.total_collisions,
            "success_rate": self.total_successes / eps,
            "collision_rate": self.total_collisions / eps,
            "mean_reward": float(np.mean(rewards)),
            "final_10pct_mean_reward": float(np.mean(rewards[max(0, int(len(rewards) * 0.9)) :])),
        }
