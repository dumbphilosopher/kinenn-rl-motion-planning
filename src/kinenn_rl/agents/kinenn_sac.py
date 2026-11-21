"""KineNN-Informed SAC (Thesis Section 4.2, Figure 4.2, Algorithm 4)."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.noise import ActionNoise

from kinenn_rl.agents.action_filter import ActionSamplingFilter
from kinenn_rl.kinenn.kinenn_htm import KineNNHTM


class KineNNSAC(SAC):
    """SAC with KineNN Timestep Action Sampling Filter (Algorithm 4)."""

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

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Route to KineNN for episode step n < t_filter, else SAC policy (Algorithm 4)."""
        unscaled_action, buffer_action = super()._sample_action(
            learning_starts=learning_starts,
            action_noise=action_noise,
            n_envs=n_envs,
        )

        if self.env is None:
            return unscaled_action, buffer_action

        for idx in range(n_envs):
            base_env = self.env.envs[idx].unwrapped
            ep_step = int(getattr(base_env, "_step_count", 0))
            if self.action_filter.use_kinenn(ep_step):
                q = base_env.data.qpos[:6].copy()
                base_pos = base_env.data.xpos[base_env.model.body("robot_base").id]
                target_pos_base = base_env.target_pos - base_pos
                a_k = self.kinenn.compute_ik_action(q, target_pos_base, base_env.max_delta_q)
                unscaled_action[idx] = a_k
                buffer_action[idx] = self.policy.scale_action(a_k)
            # Update KineNN world model with current (q, eef_pos_base)
            eef_pos, _ = base_env._get_eef_pose()
            base_pos = base_env.data.xpos[base_env.model.body("robot_base").id]
            self.action_filter.record_transition(base_env.data.qpos[:6], eef_pos - base_pos)

        return unscaled_action, buffer_action
