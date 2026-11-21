"""Unit tests for the ABB IRB 1200 MuJoCo Welding Station Gymnasium environment."""

import numpy as np

from kinenn_rl.env import ABBIrb1200WeldingEnv


def test_env_spaces_and_reset():
    env = ABBIrb1200WeldingEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (10,)
    assert env.action_space.shape == (6,)
    assert not info["collision"]
    assert info["distance_error"] > 0.05
    env.close()


def test_reward_equation_5_1():
    env = ABBIrb1200WeldingEnv(reward_alpha=1e4)
    # Collision -> reward = 0.0
    assert env.compute_reward(distance_error=0.2, collision=True, success=False) == 0.0
    # Success -> reward = 1.0
    assert env.compute_reward(distance_error=0.004, collision=False, success=True) == 1.0
    # Closer distance yields strictly higher shaped reward
    r_far = env.compute_reward(distance_error=0.5, collision=False, success=False)
    r_near = env.compute_reward(distance_error=0.1, collision=False, success=False)
    assert 0.0 < r_far < r_near < 1.0
    env.close()


def test_env_step_and_headless_render():
    env = ABBIrb1200WeldingEnv(render_mode="rgb_array", render_width=320, render_height=240)
    obs, _ = env.reset(seed=1)
    obs2, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
    assert obs2.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    frame = env.render()
    assert frame is not None and frame.shape == (240, 320, 3)
    env.close()
