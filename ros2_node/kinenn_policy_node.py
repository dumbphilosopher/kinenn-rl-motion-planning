"""Optional ROS 2 Humble/Jazzy inference node for KineNN-Informed RL policies.

Subscribes to `/joint_states` and `/welding_target_pose`, computes 6-DoF joint
commands at 20 Hz (dt = 0.05 s), and publishes JointTrajectory messages to
`/joint_trajectory_controller/joint_trajectory` for ABB IRB 1200 / EGM control.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except ImportError as exc:
    raise ImportError(
        "ROS 2 (rclpy) is not installed in this Python environment. "
        "Source your ROS 2 installation (e.g., `source /opt/ros/jazzy/setup.bash`) before running."
    ) from exc

from stable_baselines3 import PPO

from kinenn_rl.env.welding_env import JOINT_LIMITS_HIGH, JOINT_LIMITS_LOW, MAX_JOINT_SPEEDS
from kinenn_rl.kinenn import KineNNHTM


class KineNNPolicyNode(Node):
    """20 Hz ROS 2 closed-loop policy node for ABB IRB 1200 welding motion planning."""

    JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    def __init__(self, model_path: str, kinenn_dh_path: str | None = None, filter_steps: int = 35) -> None:
        super().__init__("kinenn_rl_policy_node")
        self.model = PPO.load(model_path)
        self.kinenn = KineNNHTM(perturbation_std=0.0)
        if kinenn_dh_path and Path(kinenn_dh_path).exists():
            self.kinenn.load_state_dict(torch.load(kinenn_dh_path, map_location="cpu", weights_only=True))

        self.filter_steps = filter_steps
        self.dt = 0.05
        self.max_delta_q = MAX_JOINT_SPEEDS * self.dt * 0.35
        self.step_count = 0

        self.q_curr = np.zeros(6, dtype=np.float64)
        self.target_pos_base = np.array([0.55, 0.18, 0.20], dtype=np.float64)
        self.has_state = False

        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(PoseStamped, "/welding_target_pose", self._on_target_pose, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
        )
        self.create_timer(self.dt, self._control_loop)
        self.get_logger().info("KineNN-RL Policy Node started at 20 Hz.")

    def _on_joint_state(self, msg: JointState) -> None:
        if len(msg.position) >= 6:
            self.q_curr = np.array(msg.position[:6], dtype=np.float64)
            self.has_state = True

    def _on_target_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.target_pos_base = np.array([p.x, p.y, p.z], dtype=np.float64)
        self.step_count = 0

    def _control_loop(self) -> None:
        if not self.has_state:
            return

        with torch.no_grad():
            eef_pos = self.kinenn(torch.as_tensor(self.q_curr, dtype=torch.float32)).numpy()
        delta_t = self.target_pos_base - eef_pos
        obs = np.concatenate([self.q_curr, delta_t, [0.0]]).astype(np.float32)

        if self.step_count < self.filter_steps:
            action = self.kinenn.compute_ik_action(self.q_curr, self.target_pos_base, self.max_delta_q)
        else:
            action, _ = self.model.predict(obs, deterministic=True)

        q_next = np.clip(self.q_curr + action * self.max_delta_q, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        self.step_count += 1

        traj = JointTrajectory()
        traj.joint_names = self.JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = q_next.tolist()
        pt.time_from_start.nanosec = int(self.dt * 1e9)
        traj.points = [pt]
        self.traj_pub.publish(traj)


def main() -> None:
    rclpy.init()
    node = KineNNPolicyNode(model_path="runs/ppo_kinenn_s42/model.zip", kinenn_dh_path="runs/ppo_kinenn_s42/kinenn_dh.pt")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
