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


class PenBalanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Simple MuJoCo/Gymnasium environment for pen balancing on a 2-axis trolley."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str | Path = "PModel/PModel.urdf",
        frame_skip: int = 5,
        max_motor_speed_rad_s: float = 14.0,
        max_steps: int = 600,
        fail_radius_m: float = 0.20,
        reset_tilt_rad: float = 0.10,
        ring_near_m: float = 0.015,
        ring_touch_m: float = 0.010,
        target_near_m: float = 0.035,
        stuck_speed_m_s: float = 0.01,
        stuck_steps: int = 10,
    ) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_motor_speed_rad_s = max_motor_speed_rad_s
        self.max_steps = max_steps
        self.fail_radius_m = fail_radius_m
        self.reset_tilt_rad = reset_tilt_rad
        self.ring_near_m = ring_near_m
        self.ring_touch_m = ring_touch_m
        self.target_near_m = target_near_m
        self.stuck_speed_m_s = stuck_speed_m_s
        self.stuck_steps = stuck_steps
        self.mix = MotorMixing()
        self.step_count = 0
        self.ring_stuck_counter = 0

        self.slide_jid = self._joint_id("Plate_Slider-7")
        self.trolley_jid = self._joint_id("Slide_Slider-8")
        self.pen_tip_gid = self._geom_id_any("PenTip_collision", "Pen_collision")
        self.ring_seg_gids = self._ring_segment_geom_ids()

        self.slide_qpos_adr = int(self.model.jnt_qposadr[self.slide_jid])
        self.trolley_qpos_adr = int(self.model.jnt_qposadr[self.trolley_jid])
        self.slide_dof_adr = int(self.model.jnt_dofadr[self.slide_jid])
        self.trolley_dof_adr = int(self.model.jnt_dofadr[self.trolley_jid])

        slide_range = self.model.jnt_range[self.slide_jid]
        trolley_range = self.model.jnt_range[self.trolley_jid]
        self.slide_min, self.slide_max = float(slide_range[0]), float(slide_range[1])
        self.trolley_min, self.trolley_max = float(trolley_range[0]), float(trolley_range[1])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Obs: tip x/y, tip vx/vy, slide pos, trolley pos.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.target_xz = self._tip_xz().copy()
        self.prev_tip_xz = self.target_xz.copy()
        self.ring_center_x = self._ring_center_x()

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in model.")
        return jid

    def _geom_id_any(self, *names: str) -> int:
        for name in names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                return gid
        raise ValueError(f"None of geoms {names} found in model.")

    def _tip_xz(self) -> np.ndarray:
        return self.data.geom_xpos[self.pen_tip_gid, [0, 2]].copy()

    def _ring_segment_geom_ids(self) -> list[int]:
        ids: list[int] = []
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name.startswith("Ring_collision_seg_"):
                ids.append(i)
        if not ids:
            raise ValueError("No ring segment geoms named 'Ring_collision_seg_*' found.")
        return ids

    def _tip_ring_min_dist(self, tip_xz: np.ndarray) -> float:
        ring_xz = self.data.geom_xpos[self.ring_seg_gids][:, [0, 2]]
        return float(np.min(np.linalg.norm(ring_xz - tip_xz, axis=1)))

    def _ring_center_x(self) -> float:
        return float(np.mean(self.data.geom_xpos[self.ring_seg_gids, 0]))

    def _obs(self) -> np.ndarray:
        tip_xz = self._tip_xz()
        dt = self.model.opt.timestep * self.frame_skip
        tip_v = (tip_xz - self.prev_tip_xz) / max(dt, 1e-8)
        obs = np.array(
            [
                tip_xz[0],
                tip_xz[1],
                tip_v[0],
                tip_v[1],
                self.data.qpos[self.slide_qpos_adr],
                self.data.qpos[self.trolley_qpos_adr],
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.ring_stuck_counter = 0
        mujoco.mj_resetData(self.model, self.data)

        # Randomize initial tilt to prevent overfitting one nominal start state.
        for joint_name in ("Trolley_Revolute-15", "YBAll_Revolute-16"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                qadr = int(self.model.jnt_qposadr[jid])
                self.data.qpos[qadr] += self.np_random.uniform(
                    -self.reset_tilt_rad, self.reset_tilt_rad
                )

        mujoco.mj_forward(self.model, self.data)
        self.target_xz = self._tip_xz().copy()
        self.prev_tip_xz = self.target_xz.copy()
        self.ring_center_x = self._ring_center_x()
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

        tip_xz = self._tip_xz()
        x_err = float(tip_xz[0] - self.ring_center_x)
        z_err = float(tip_xz[1] - self.target_xz[1])
        z_drop = max(0.0, -z_err)
        tip_vx = float((tip_xz[0] - self.prev_tip_xz[0]) / (dt * self.frame_skip))
        dist = float(np.linalg.norm(tip_xz - self.target_xz))
        tip_speed = float(np.linalg.norm((tip_xz - self.prev_tip_xz) / (dt * self.frame_skip)))

        ring_min_dist = self._tip_ring_min_dist(tip_xz)
        near_ring = ring_min_dist < self.ring_near_m
        touching_ring = ring_min_dist < self.ring_touch_m
        near_target = dist < self.target_near_m

        # Base objective:
        # - significantly emphasize staying at maximum Z (upright)
        # - no direct reward for absolute X position
        # - slight penalty for high X motion to discourage unnecessary sweeping.
        reward = 2.0
        reward -= 60.0 * z_drop
        reward -= 120.0 * z_drop * z_drop
        reward -= 0.15 * abs(tip_vx)

        # Only encourage smoothness near target and away from the ring.
        if near_target and not near_ring:
            reward -= 0.03 * tip_speed

        # Strongly discourage relying on ring contact/support.
        ring_margin_penalty = max(0.0, self.ring_near_m - ring_min_dist)
        reward -= 25.0 * ring_margin_penalty * ring_margin_penalty
        if touching_ring:
            reward -= 0.8

        # If tip is near ring and nearly stationary, treat it as "stuck on boundary".
        if near_ring and tip_speed < self.stuck_speed_m_s:
            self.ring_stuck_counter += 1
        else:
            self.ring_stuck_counter = 0
        stuck_near_ring = self.ring_stuck_counter >= self.stuck_steps
        if stuck_near_ring:
            reward -= 1.5

        self.prev_tip_xz = tip_xz.copy()
        self.step_count += 1
        terminated = bool(dist > self.fail_radius_m or z_err < -0.12)
        truncated = bool(self.step_count >= self.max_steps)
        info = {
            "tip_xz": tip_xz,
            "tip_xy": tip_xz,  # Backward-compatible key name for existing scripts.
            "tip_dist": dist,
            "tip_speed": tip_speed,
            "tip_vx": tip_vx,
            "x_err": x_err,
            "z_err": z_err,
            "z_drop": z_drop,
            "ring_center_x": self.ring_center_x,
            "ring_min_dist": ring_min_dist,
            "near_ring": near_ring,
            "touching_ring": touching_ring,
            "stuck_near_ring": stuck_near_ring,
            "ring_stuck_counter": self.ring_stuck_counter,
        }
        return self._obs(), reward, terminated, truncated, info
