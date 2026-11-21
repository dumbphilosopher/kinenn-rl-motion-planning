"""HTM-based Kinematic Neural Network (KineNN) with learnable DH parameters.

Implements Section 3.3 (Diprasetya et al., 2024) and Chapter 4:
- Replaces abstract NN weights with learnable Denavit-Hartenberg (DH) parameters
  (a, alpha, d, theta_offset) chained via 4x4 Homogeneous Transformation Matrices.
- Acts as both a differentiable Forward Kinematics World Model (updated via backprop
  on observed end-effector poses) and a Differentiable Inverse Kinematics controller
  computing joint-angle delta actions via damped least-squares Jacobian pseudoinverse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Exact ABB IRB 1200-5/0.9 standard DH parameters (Table 5.3, meters & radians)
ABB_IRB1200_DH = {
    "a": [0.0, 0.448, 0.042, 0.0, 0.0, 0.0],
    "alpha": [-math.pi / 2, 0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, 0.0],
    "d": [0.399, 0.0, 0.0, 0.451, 0.0, 0.082],
    "theta_offset": [0.0, -math.pi / 2, 0.0, 0.0, 0.0, -math.pi],
}


def dh_htm_batch(
    theta: torch.Tensor,
    d: torch.Tensor,
    a: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Build standard DH 4x4 Homogeneous Transformation Matrices for a batch.

    Args:
        theta: (B,) joint angle + offset in radians
        d, a, alpha: scalar tensors for link i
    Returns:
        (B, 4, 4) transformation matrices.
    """
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)

    row0 = torch.stack([ct, -st * ca, st * sa, a * ct], dim=-1)
    row1 = torch.stack([st, ct * ca, -ct * sa, a * st], dim=-1)
    row2 = torch.stack([zeros, sa.expand_as(theta), ca.expand_as(theta), d.expand_as(theta)], dim=-1)
    row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


class KineNNHTM(nn.Module):
    """Learnable Denavit-Hartenberg Homogeneous Transformation Matrix Network."""

    def __init__(
        self,
        init_dh: dict[str, list[float]] | None = None,
        perturbation_std: float = 0.015,
        lr: float = 1e-3,
        ik_damping: float = 0.08,
        ik_gain: float = 0.85,
        seed: int = 0,
    ) -> None:
        super().__init__()
        base_dh = ABB_IRB1200_DH if init_dh is None else init_dh
        gen = torch.Generator().manual_seed(seed)

        def _make_param(vals: list[float], noise_scale: float) -> nn.Parameter:
            t = torch.tensor(vals, dtype=torch.float32)
            if noise_scale > 0.0:
                t = t + torch.randn(t.shape, generator=gen) * noise_scale
            return nn.Parameter(t)

        # Learnable DH parameters (Table 5.3)
        self.a = _make_param(base_dh["a"], perturbation_std)
        self.d = _make_param(base_dh["d"], perturbation_std)
        self.alpha = _make_param(base_dh["alpha"], perturbation_std * 0.5)
        self.theta_offset = _make_param(base_dh["theta_offset"], perturbation_std * 0.5)

        self.ik_damping = float(ik_damping)
        self.ik_gain = float(ik_gain)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward_htm(self, q: torch.Tensor) -> torch.Tensor:
        """Compute base-to-EEF 4x4 HTM T_0^6(q) for q of shape (B, 6) or (6,)."""
        single = q.ndim == 1
        if single:
            q = q.unsqueeze(0)
        batch_size = q.shape[0]
        T = torch.eye(4, dtype=q.dtype, device=q.device).unsqueeze(0).expand(batch_size, 4, 4).clone()
        for i in range(6):
            Ti = dh_htm_batch(q[:, i] + self.theta_offset[i], self.d[i], self.a[i], self.alpha[i])
            T = torch.bmm(T, Ti)
        return T.squeeze(0) if single else T

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Predict end-effector Cartesian position (base frame) for joint angles q."""
        T = self.forward_htm(q)
        return T[..., :3, 3]

    def update_world_model(self, q_batch: np.ndarray, pos_true_batch: np.ndarray) -> float:
        """Backpropagate end-effector position MSE to refine learnable DH parameters."""
        q_t = torch.as_tensor(q_batch, dtype=torch.float32)
        p_true = torch.as_tensor(pos_true_batch, dtype=torch.float32)
        self.optimizer.zero_grad()
        p_pred = self.forward(q_t)
        loss = nn.functional.mse_loss(p_pred, p_true)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def compute_ik_action(
        self,
        q: np.ndarray,
        target_pos_base: np.ndarray,
        max_delta_q: np.ndarray,
    ) -> np.ndarray:
        """Compute normalized 6D joint action a_k(t) in [-1, 1]^6 via damped-LS Jacobian IK."""
        q_t = torch.tensor(q, dtype=torch.float32, requires_grad=True)
        p_curr = self.forward(q_t)

        # Compute 3x6 positional Jacobian J = dp / dq via autograd
        jac_rows = []
        for coord in range(3):
            grad = torch.autograd.grad(p_curr[coord], q_t, retain_graph=(coord < 2))[0]
            jac_rows.append(grad)
        J = torch.stack(jac_rows, dim=0).detach().numpy().astype(np.float64)

        err = np.asarray(target_pos_base, dtype=np.float64) - p_curr.detach().numpy().astype(np.float64)
        # Damped least-squares pseudoinverse: dq = J^T (J J^T + lambda^2 I)^-1 * (gain * err)
        jjt = J @ J.T + (self.ik_damping**2) * np.eye(3)
        delta_q = J.T @ np.linalg.solve(jjt, self.ik_gain * err)

        # Normalize to environment action space [-1, 1]^6
        norm_action = np.clip(delta_q / (np.asarray(max_delta_q, dtype=np.float64) + 1e-8), -1.0, 1.0)
        return norm_action.astype(np.float32)
