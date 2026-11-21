# ROS 2 Policy Inference Node (Optional)

This standalone `rclpy` node deploys a trained `KineNN-PPO` checkpoint at **20 Hz** ($\Delta t = 0.05\text{ s}$) onto an ABB IRB 1200 ROS 2 stack (e.g., via `abb_ros2` + EGM/RWS as described in Section 5.1.3 of the thesis).

## Topics
- **Subscribes:**
  - `/joint_states` (`sensor_msgs/JointState`)
  - `/welding_target_pose` (`geometry_msgs/PoseStamped`, robot base frame)
- **Publishes:**
  - `/joint_trajectory_controller/joint_trajectory` (`trajectory_msgs/JointTrajectory`)

## Usage
```bash
source /opt/ros/jazzy/setup.bash
python3 ros2_node/kinenn_policy_node.py
```
