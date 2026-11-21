"""KineNN-Informed PPO (Thesis Section 4.3, Figure 4.3, Algorithm 5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from kinenn_rl.agents.action_filter import ActionSamplingFilter
from kinenn_rl.kinenn.kinenn_htm import KineNNHTM


class KineNNPPO(PPO):
    """PPO with KineNN Timestep Action Sampling Filter (Algorithm 5)."""

    def __init__(
        self,
        *args: Any,
        kinenn: KineNNHTM | None = None,
        filter_steps: int = 40,
        dh_update_freq: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kinenn = kinenn if kinenn is not None else KineNNHTM()
        self.action_filter = ActionSamplingFilter(
            kinenn=self.kinenn,
            filter_steps=filter_steps,
            dh_update_freq=dh_update_freq,
        )
        self._ep_steps: np.ndarray | None = None

    def _get_base_env(self, env: VecEnv, idx: int = 0) -> Any:
        return env.envs[idx].unwrapped

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect rollouts using KineNN for step n < t_filter and PPO actor for n >= t_filter."""
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        if self._ep_steps is None or len(self._ep_steps) != env.num_envs:
            self._ep_steps = np.zeros(env.num_envs, dtype=np.int64)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions_np = actions.cpu().numpy().copy()
            replaced_any = False

            # Apply Action Sampling Filter per environment (Algorithm 5)
            for idx in range(env.num_envs):
                if self.action_filter.use_kinenn(int(self._ep_steps[idx])):
                    base_env = self._get_base_env(env, idx)
                    q = base_env.data.qpos[:6].copy()
                    target_pos_base = base_env.target_pos - base_env.data.xpos[base_env.model.body("robot_base").id]
                    a_k = self.kinenn.compute_ik_action(q, target_pos_base, base_env.max_delta_q)
                    actions_np[idx] = a_k
                    replaced_any = True

            # Recompute policy log_probs for the executed actions so PPO ratio is well-defined
            if replaced_any:
                with th.no_grad():
                    exec_actions_t = th.as_tensor(actions_np, dtype=th.float32, device=self.device)
                    values, log_probs, _ = self.policy.evaluate_actions(obs_tensor, exec_actions_t)

            clipped_actions = actions_np
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions_np, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            # Update episode step counters and KineNN world model
            for idx in range(env.num_envs):
                info = infos[idx]
                if "joint_angles" in info and "eef_pos_base" in info:
                    self.action_filter.record_transition(info["joint_angles"], info["eef_pos_base"])
                if dones[idx]:
                    self._ep_steps[idx] = 0
                else:
                    self._ep_steps[idx] += 1

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True
