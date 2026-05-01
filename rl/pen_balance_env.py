from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class MotorMixing:
    """Maps motor angular velocity to linear X/Y carriage velocity."""

    meters_per_rad: float = 0.03839 / (2.0 * np.pi)

    def to_linear_xy(self, motor_a_rad_s: float, motor_b_rad_s: float) -> tuple[float, float]:
        # Same direction -> trolley X, opposite direction -> slide Y.
        vx = -0.5 * self.meters_per_rad * (motor_a_rad_s + motor_b_rad_s)
        vy = -0.5 * self.meters_per_rad * (motor_a_rad_s - motor_b_rad_s)
        return vx, vy


class TrolleyCircleEnv(gym.Env[np.ndarray, np.ndarray]):
    """MuJoCo env: drive slide + trolley so the trolley body tracks a small horizontal circle.

    Reference motion is a point moving CCW on a circle of radius ``circle_radius_m`` centered at the
    trolley XY position when the episode starts. Reward penalizes Cartesian error to that moving target.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str | Path = "PModel/PModel.urdf",
        frame_skip: int = 5,
        max_motor_speed_rad_s: float = 14.0,
        max_steps: int = 600,
        circle_radius_m: float = 0.005,
        circle_omega_rad_s: float = np.pi,
        track_reward_scale: float = 1.0,
        radius_shaping_scale: float = 0.35,
        action_delta_penalty: float = 0.02,
        success_track_m: float = 0.002,
        success_radius_m: float = 0.0015,
    ) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_motor_speed_rad_s = max_motor_speed_rad_s
        self.max_steps = max_steps
        self.circle_radius_m = float(circle_radius_m)
        self.circle_omega_rad_s = float(circle_omega_rad_s)
        self.track_reward_scale = float(track_reward_scale)
        self.radius_shaping_scale = float(radius_shaping_scale)
        self.action_delta_penalty = float(action_delta_penalty)
        self.success_track_m = float(success_track_m)
        self.success_radius_m = float(success_radius_m)

        self.mix = MotorMixing()
        self.step_count = 0
        self.phase = 0.0
        self.episode_center_xy = np.zeros(2, dtype=np.float64)
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.prev_trolley_xy = np.zeros(2, dtype=np.float64)

        self.slide_jid = self._joint_id("Plate_Slider-7")
        self.trolley_jid = self._joint_id("Slide_Slider-8")
        self.trolley_bid = self._body_id("Trolley")

        self.slide_qpos_adr = int(self.model.jnt_qposadr[self.slide_jid])
        self.trolley_qpos_adr = int(self.model.jnt_qposadr[self.trolley_jid])
        self.slide_dof_adr = int(self.model.jnt_dofadr[self.slide_jid])
        self.trolley_dof_adr = int(self.model.jnt_dofadr[self.trolley_jid])

        slide_range = self.model.jnt_range[self.slide_jid]
        trolley_range = self.model.jnt_range[self.trolley_jid]
        self.slide_min, self.slide_max = float(slide_range[0]), float(slide_range[1])
        self.trolley_min, self.trolley_max = float(trolley_range[0]), float(trolley_range[1])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # rel trolley/center, trolley vel, phase encoding, error to target (all scaled).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in model.")
        return jid

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model.")
        return bid

    def _trolley_xy(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.trolley_bid, :2], dtype=np.float64).copy()

    def _target_xy(self) -> np.ndarray:
        r = self.circle_radius_m
        return self.episode_center_xy + r * np.array(
            [np.cos(self.phase), np.sin(self.phase)], dtype=np.float64
        )

    def _obs(self) -> np.ndarray:
        trolley_xy = self._trolley_xy()
        target_xy = self._target_xy()
        dt = self.model.opt.timestep * self.frame_skip
        vel = (trolley_xy - self.prev_trolley_xy) / max(dt, 1e-8)
        scale_r = max(self.circle_radius_m, 1e-6)
        v_scale = 0.05
        rel = (trolley_xy - self.episode_center_xy) / scale_r
        err = (trolley_xy - target_xy) / scale_r
        return np.array(
            [
                rel[0],
                rel[1],
                vel[0] / v_scale,
                vel[1] / v_scale,
                np.cos(self.phase),
                np.sin(self.phase),
                err[0],
                err[1],
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.phase = 0.0
        self.prev_action = np.zeros(2, dtype=np.float64)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.slide_qpos_adr] = 0.0
        self.data.qpos[self.trolley_qpos_adr] = 0.0
        self.data.qvel[self.slide_dof_adr] = 0.0
        self.data.qvel[self.trolley_dof_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.episode_center_xy = self._trolley_xy()
        self.prev_trolley_xy = self.episode_center_xy.copy()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        motor_a = float(action[0] * self.max_motor_speed_rad_s)
        motor_b = float(action[1] * self.max_motor_speed_rad_s)

        dt = self.model.opt.timestep
        for _ in range(self.frame_skip):
            vx, vy = self.mix.to_linear_xy(motor_a, motor_b)
            slide = float(self.data.qpos[self.slide_qpos_adr]) + vy * dt
            trolley = float(self.data.qpos[self.trolley_qpos_adr]) + vx * dt
            slide = float(np.clip(slide, self.slide_min, self.slide_max))
            trolley = float(np.clip(trolley, self.trolley_min, self.trolley_max))
            self.data.qpos[self.slide_qpos_adr] = slide
            self.data.qpos[self.trolley_qpos_adr] = trolley
            self.data.qvel[self.slide_dof_adr] = vy
            self.data.qvel[self.trolley_dof_adr] = vx
            mujoco.mj_step(self.model, self.data)

        dt_step = dt * self.frame_skip
        trolley_xy = self._trolley_xy()
        target_xy = self._target_xy()
        err_vec = trolley_xy - target_xy
        dist_track = float(np.linalg.norm(err_vec))
        delta_center = trolley_xy - self.episode_center_xy
        r_actual = float(np.linalg.norm(delta_center))
        radius_err = r_actual - self.circle_radius_m

        denom = max(self.circle_radius_m**2, 1e-12)
        reward = -self.track_reward_scale * float(np.dot(err_vec, err_vec)) / denom
        reward -= self.radius_shaping_scale * (radius_err**2) / denom

        da = action - self.prev_action
        reward -= self.action_delta_penalty * float(np.dot(da, da))
        self.prev_action = action.copy()

        self.phase += self.circle_omega_rad_s * dt_step
        self.phase = float(np.remainder(self.phase, 2.0 * np.pi))

        track_ok = dist_track < self.success_track_m
        radius_ok = abs(radius_err) < self.success_radius_m
        step_success = bool(track_ok and radius_ok)

        self.prev_trolley_xy = trolley_xy.copy()
        self.step_count += 1
        terminated = False
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "tracking_error_m": dist_track,
            "radius_error_m": abs(radius_err),
            "target_xy": target_xy,
            "trolley_xy": trolley_xy,
            "circle_radius_m": self.circle_radius_m,
            "step_success": step_success,
            "track_ok": track_ok,
            "radius_ok": radius_ok,
        }
        return self._obs(), float(reward), terminated, truncated, info


# Backwards-compatible alias (previous task name).
PenBalanceEnv = TrolleyCircleEnv
