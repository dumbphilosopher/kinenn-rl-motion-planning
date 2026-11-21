"""Unit tests for HTM-based KineNN (FK accuracy, DH online learning, differentiable IK)."""

import numpy as np
import torch

from kinenn_rl.env import ABBIrb1200WeldingEnv
from kinenn_rl.kinenn import KineNNHTM


def test_kinenn_fk_matches_mujoco_exact_dh():
    """Ground-truth DH parameters in KineNNHTM must match MuJoCo FK within < 1 mm."""
    env = ABBIrb1200WeldingEnv()
    kinenn = KineNNHTM(perturbation_std=0.0)
    rng = np.random.default_rng(0)

    for _ in range(10):
        q = rng.uniform(-1.0, 1.0, size=6)
        env.data.qpos[:6] = q
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        eef_pos_world, _ = env._get_eef_pose()
        base_pos = env.data.xpos[env.model.body("robot_base").id]
        eef_pos_base = eef_pos_world - base_pos

        with torch.no_grad():
            pred = kinenn(torch.as_tensor(q, dtype=torch.float32)).numpy()

        err_m = float(np.linalg.norm(pred - eef_pos_base))
        assert err_m < 1e-3, f"FK mismatch: {err_m:.6f} m (pred={pred}, true={eef_pos_base})"
    env.close()


def test_kinenn_online_dh_learning_converges():
    """Perturbed KineNN DH parameters should reduce FK MSE by >90% via backprop."""
    gt_kinenn = KineNNHTM(perturbation_std=0.0)
    learner = KineNNHTM(perturbation_std=0.05, lr=1e-2, seed=7)

    rng = np.random.default_rng(42)
    q_samples = rng.uniform(-1.2, 1.2, size=(256, 6)).astype(np.float32)
    with torch.no_grad():
        p_true = gt_kinenn(torch.from_numpy(q_samples)).numpy()

    with torch.no_grad():
        init_mse = float(np.mean((learner(torch.from_numpy(q_samples)).numpy() - p_true) ** 2))

    for _ in range(120):
        learner.update_world_model(q_samples, p_true)

    with torch.no_grad():
        final_mse = float(np.mean((learner(torch.from_numpy(q_samples)).numpy() - p_true) ** 2))

    assert final_mse < init_mse * 0.05, f"DH learning failed to converge: {init_mse:.6f} -> {final_mse:.6f}"


def test_kinenn_ik_reduces_distance_to_target():
    """Iterative KineNN damped-LS IK should smoothly move the end-effector toward target."""
    kinenn = KineNNHTM(perturbation_std=0.0)
    q = np.array([-0.5, -0.2, 0.2, 0.0, 0.4, 0.0], dtype=np.float64)
    target = np.array([0.55, 0.18, 0.20], dtype=np.float64)
    max_dq = np.full(6, 0.12, dtype=np.float64)

    with torch.no_grad():
        init_dist = float(np.linalg.norm(kinenn(torch.as_tensor(q, dtype=torch.float32)).numpy() - target))

    for _ in range(25):
        a = kinenn.compute_ik_action(q, target, max_dq)
        assert a.shape == (6,)
        assert np.all(a >= -1.0) and np.all(a <= 1.0)
        q = q + a * max_dq

    with torch.no_grad():
        final_dist = float(np.linalg.norm(kinenn(torch.as_tensor(q, dtype=torch.float32)).numpy() - target))

    assert final_dist < 0.02 and final_dist < init_dist * 0.1
