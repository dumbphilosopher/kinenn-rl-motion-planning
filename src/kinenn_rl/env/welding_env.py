"""MuJoCo Gymnasium environment for the ABB IRB 1200-5/0.9 Welding Station.

Matches the thesis specification (Chapter 5 & 6):
- Robot: ABB IRB 1200-5/0.9 (6 revolute joints, exact Table 5.2 & 5.3 geometry)
- State Space (10D, Table 5.8): [J1..J6, Delta_Tx, Delta_Ty, Delta_Tz, Delta_TO]
- Action Space (6D, Table 5.9): Joint angle deltas scaled by max joint velocity * dt
- Reward (Eq. 5.1): R(t) = (1 - C)(1 - S) * (1 / (alpha * d_t^3)) + S, alpha = 1e4
- Horizon: 240 cycles at dt = 0.05 s
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# ABB IRB 1200-5/0.9 Joint Limits in radians (Table 5.2)
JOINT_LIMITS_LOW = np.deg2rad([-170.0, -100.0, -200.0, -270.0, -130.0, -400.0]).astype(np.float64)
JOINT_LIMITS_HIGH = np.deg2rad([170.0, 130.0, 70.0, 270.0, 130.0, 400.0]).astype(np.float64)

# ABB IRB 1200-5/0.9 Max Joint Speeds in rad/s (Table 5.2)
MAX_JOINT_SPEEDS = np.deg2rad([288.0, 240.0, 297.0, 400.0, 405.0, 600.0]).astype(np.float64)

# Robot base offset in world coordinates (matches assets/abb_irb1200_welding.xml)
ROBOT_BASE_POS = np.array([0.0, 0.0, 0.36], dtype=np.float64)

# Default home pose & static target position (reachable, clear of workpiece obstacle)
DEFAULT_HOME_Q = np.array([-0.6, -0.25, 0.25, 0.0, 0.45, 0.0], dtype=np.float64)
DEFAULT_TARGET_POS = np.array([0.55, 0.18, 0.56], dtype=np.float64)


def _quat_geodesic_error(q1: np.ndarray, q2: np.ndarray) -> float:
    """Return orientation error in [0, 1]: 1 - |<q1, q2>| (unit quaternions)."""
    q1_n = q1 / (np.linalg.norm(q1) + 1e-12)
    q2_n = q2 / (np.linalg.norm(q2) + 1e-12)
    return float(1.0 - np.clip(abs(np.dot(q1_n, q2_n)), 0.0, 1.0))


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniform random unit quaternion [w, x, y, z]."""
    u1, u2, u3 = rng.random(3)
    return np.array(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ],
        dtype=np.float64,
    )


class ABBIrb1200WeldingEnv(gym.Env):
    """6-DoF ABB IRB 1200 welding station motion-planning environment in MuJoCo."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        action_dt: float = 0.05,
        max_steps: int = 240,
        success_tolerance: float = 0.025,
        reward_alpha: float = 1e4,
        max_step_scale: float = 0.35,
        randomize_target_orientation: bool = True,
        target_pos: np.ndarray | None = None,
        render_mode: str | None = None,
        render_width: int = 640,
        render_height: int = 480,
    ) -> None:
        super().__init__()
        if xml_path is None:
            xml_path = Path(__file__).resolve().parents[3] / "assets" / "abb_irb1200_welding.xml"
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.action_dt = float(action_dt)
        self.n_substeps = max(1, int(round(self.action_dt / self.model.opt.timestep)))
        self.max_steps = int(max_steps)
        self.success_tolerance = float(success_tolerance)
        self.reward_alpha = float(reward_alpha)
        self.max_delta_q = MAX_JOINT_SPEEDS * self.action_dt * float(max_step_scale)
        self.randomize_target_orientation = bool(randomize_target_orientation)
        self.target_pos = (
            DEFAULT_TARGET_POS.copy() if target_pos is None else np.asarray(target_pos, dtype=np.float64)
        )
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height
        self._renderer: mujoco.Renderer | None = None

        self._eef_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        self._target_mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        ]

        # Track obstacle & robot geom IDs for exact collision classification
        self._obstacle_geom_ids = {
            i
            for i in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("obs_")
            or (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "") == "floor"
        }
        self._robot_geom_ids = {
            i
            for i in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
                ("link", "welding_torch")
            )
        }

        # State space (10D, Table 5.8): [J1..J6, Delta_Tx, Delta_Ty, Delta_Tz, Delta_TO]
        obs_low = np.concatenate([JOINT_LIMITS_LOW, [-5.0, -5.0, -5.0, 0.0]]).astype(np.float32)
        obs_high = np.concatenate([JOINT_LIMITS_HIGH, [5.0, 5.0, 5.0, 1.0]]).astype(np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Normalized continuous action space in [-1, 1]^6 mapped to [-max_delta_q, +max_delta_q]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self._step_count = 0

    def _get_eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (eef_pos_world [3], eef_quat_wxyz [4])."""
        pos = self.data.site_xpos[self._eef_site_id].copy()
        mat = self.data.site_xmat[self._eef_site_id].reshape(3, 3)
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, mat.ravel())
        return pos, quat

    def _check_collision(self) -> bool:
        """Check whether any robot link contacts an obstacle, floor, or non-adjacent link."""
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in self._robot_geom_ids and g2 in self._obstacle_geom_ids) or (
                g2 in self._robot_geom_ids and g1 in self._obstacle_geom_ids
            ):
                return True
        return False

    def _build_obs_and_info(self, collision: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        q = self.data.qpos[:6].copy()
        eef_pos, eef_quat = self._get_eef_pose()
        delta_t = self.target_pos - eef_pos
        dist = float(np.linalg.norm(delta_t))
        delta_to = _quat_geodesic_error(eef_quat, self.target_quat)
        success = dist <= self.success_tolerance and not collision

        obs = np.concatenate([q, delta_t, [delta_to]]).astype(np.float32)
        info: dict[str, Any] = {
            "joint_angles": q,
            "eef_pos": eef_pos,
            "eef_pos_base": eef_pos - ROBOT_BASE_POS,
            "eef_quat": eef_quat,
            "target_pos": self.target_pos.copy(),
            "target_pos_base": self.target_pos - ROBOT_BASE_POS,
            "target_quat": self.target_quat.copy(),
            "collision": bool(collision),
            "success": bool(success),
            "distance_error": dist,
            "orientation_error": delta_to,
            "step": self._step_count,
        }
        return obs, info

    def compute_reward(self, distance_error: float, collision: bool, success: bool) -> float:
        """Thesis Equation (5.1): R(t) = (1 - C)(1 - S)(1 / (alpha * d_t^3)) + S."""
        c = 1.0 if collision else 0.0
        s = 1.0 if success else 0.0
        d = max(float(distance_error), 1e-3)
        shaped = min(1.0 / (self.reward_alpha * (d**3)), 0.99)
        return float((1.0 - c) * (1.0 - s) * shaped + s)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        self.data.qpos[:6] = DEFAULT_HOME_Q.copy()
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:6] = DEFAULT_HOME_Q.copy()

        if self.randomize_target_orientation:
            self.target_quat = _random_unit_quat(self.np_random)
        else:
            self.target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.data.mocap_pos[self._target_mocap_id] = self.target_pos
        self.data.mocap_quat[self._target_mocap_id] = self.target_quat
        mujoco.mj_forward(self.model, self.data)

        obs, info = self._build_obs_and_info(collision=False)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        delta_q = action * self.max_delta_q
        target_q = np.clip(self.data.qpos[:6] + delta_q, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        self.data.ctrl[:6] = target_q

        collision = False
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            if self._check_collision():
                collision = True
                break

        self._step_count += 1
        obs, info = self._build_obs_and_info(collision=collision)
        reward = self.compute_reward(info["distance_error"], info["collision"], info["success"])

        terminated = bool(info["collision"] or info["success"])
        truncated = bool(self._step_count >= self.max_steps and not terminated)
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        self._renderer.update_scene(self.data, camera="station_cam")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
