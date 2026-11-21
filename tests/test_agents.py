"""Integration tests for ActionSamplingFilter, KineNNPPO, and KineNNSAC."""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from kinenn_rl.agents import ActionSamplingFilter, KineNNPPO, KineNNSAC
from kinenn_rl.env import ABBIrb1200WeldingEnv
from kinenn_rl.kinenn import KineNNHTM


def test_action_sampling_filter_routing():
    kinenn = KineNNHTM(perturbation_std=0.0)
    filt = ActionSamplingFilter(kinenn=kinenn, filter_steps=10)
    actor_a = np.full(6, 0.5, dtype=np.float32)
    q = np.zeros(6, dtype=np.float64)
    target = np.array([0.55, 0.18, 0.20], dtype=np.float64)
    max_dq = np.full(6, 0.1, dtype=np.float64)

    a_early, used_k_early = filt.select_action(3, actor_a, q, target, max_dq)
    assert used_k_early is True
    assert not np.allclose(a_early, actor_a)

    a_late, used_k_late = filt.select_action(10, actor_a, q, target, max_dq)
    assert used_k_late is False
    assert np.allclose(a_late, actor_a)


def test_kinenn_ppo_smoke_train():
    vec_env = DummyVecEnv([lambda: ABBIrb1200WeldingEnv(max_steps=40)])
    model = KineNNPPO(
        "MlpPolicy",
        vec_env,
        filter_steps=15,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=128)
    vec_env.close()


def test_kinenn_sac_smoke_train():
    vec_env = DummyVecEnv([lambda: ABBIrb1200WeldingEnv(max_steps=40)])
    model = KineNNSAC(
        "MlpPolicy",
        vec_env,
        filter_steps=15,
        learning_starts=10,
        batch_size=32,
        buffer_size=500,
        train_freq=10,
        gradient_steps=2,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=128)
    vec_env.close()
