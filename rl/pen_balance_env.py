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
        # Reward: Gaussian closeness exp(-decay * dist_xy^2) per component; higher decay = tighter peak.
        reward_tip_scale: float = 2.0,
        reward_slide_scale: float = 1.25,
        reward_trolley_scale: float = 1.25,
        reward_tip_decay: float = 350.0,
        reward_slide_decay: float = 4500.0,
        reward_trolley_decay: float = 4500.0,
        origin_close_tip_m: float = 0.012,
        origin_close_slide_m: float = 0.008,
        origin_close_trolley_m: float = 0.008,
        origin_all_close_amplify: float = 2.25,
        velocity_xy_penalty: float = 0.06,
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
        self.reward_tip_scale = reward_tip_scale
        self.reward_slide_scale = reward_slide_scale
        self.reward_trolley_scale = reward_trolley_scale
        self.reward_tip_decay = reward_tip_decay
        self.reward_slide_decay = reward_slide_decay
        self.reward_trolley_decay = reward_trolley_decay
        self.origin_close_tip_m = origin_close_tip_m
        self.origin_close_slide_m = origin_close_slide_m
        self.origin_close_trolley_m = origin_close_trolley_m
        self.origin_all_close_amplify = origin_all_close_amplify
        self.velocity_xy_penalty = velocity_xy_penalty
        self.mix = MotorMixing()
        self.step_count = 0
        self.ring_stuck_counter = 0

        self.slide_jid = self._joint_id("Plate_Slider-7")
        self.trolley_jid = self._joint_id("Slide_Slider-8")
        self.pen_tip_gid = self._geom_id_any("PenTip_collision", "Pen_collision")
        self.slide_bid = self._body_id("Slide")
        self.trolley_bid = self._body_id("Trolley")
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
        # Errors vs nominal URDF XY origins (pen tip geom, Slide/Trolley body COM), tip XY velocity.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        self._capture_nominal_xy_refs()

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.prev_tip_xy = self._tip_xy().copy()
        self.ring_center_xy = self._ring_center_xy()

    def _capture_nominal_xy_refs(self) -> None:
        """Nominal XY at URDF assembly origin: prismatic joints 0, pen revolute joints 0."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.slide_qpos_adr] = 0.0
        self.data.qpos[self.trolley_qpos_adr] = 0.0
        for joint_name in ("Trolley_Revolute-15", "YBAll_Revolute-16"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                qadr = int(self.model.jnt_qposadr[jid])
                self.data.qpos[qadr] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.tip_xy_ref = np.asarray(self.data.geom_xpos[self.pen_tip_gid, :2], dtype=np.float64).copy()
        self.slide_xy_ref = np.asarray(self.data.xpos[self.slide_bid, :2], dtype=np.float64).copy()
        self.trolley_xy_ref = np.asarray(self.data.xpos[self.trolley_bid, :2], dtype=np.float64).copy()

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

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model.")
        return bid

    def _tip_xy(self) -> np.ndarray:
        return self.data.geom_xpos[self.pen_tip_gid, :2].copy()

    def _slide_xy(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.slide_bid, :2], dtype=np.float64).copy()

    def _trolley_xy(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.trolley_bid, :2], dtype=np.float64).copy()

    def _ring_segment_geom_ids(self) -> list[int]:
        ids: list[int] = []
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name.startswith("Ring_collision_seg_"):
                ids.append(i)
        if not ids:
            raise ValueError("No ring segment geoms named 'Ring_collision_seg_*' found.")
        return ids

    def _tip_ring_min_dist_xy(self, tip_xy: np.ndarray) -> float:
        ring_xy = self.data.geom_xpos[self.ring_seg_gids][:, :2]
        return float(np.min(np.linalg.norm(ring_xy - tip_xy, axis=1)))

    def _ring_center_xy(self) -> np.ndarray:
        return np.mean(self.data.geom_xpos[self.ring_seg_gids][:, :2], axis=0)

    def _obs(self) -> np.ndarray:
        tip_xy = self._tip_xy()
        slide_xy = self._slide_xy()
        trolley_xy = self._trolley_xy()
        dt = self.model.opt.timestep * self.frame_skip
        tip_vxy = (tip_xy - self.prev_tip_xy) / max(dt, 1e-8)
        obs = np.array(
            [
                tip_xy[0] - self.tip_xy_ref[0],
                tip_xy[1] - self.tip_xy_ref[1],
                tip_vxy[0],
                tip_vxy[1],
                slide_xy[0] - self.slide_xy_ref[0],
                slide_xy[1] - self.slide_xy_ref[1],
                trolley_xy[0] - self.trolley_xy_ref[0],
                trolley_xy[1] - self.trolley_xy_ref[1],
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.ring_stuck_counter = 0
        mujoco.mj_resetData(self.model, self.data)

        for joint_name in ("Trolley_Revolute-15", "YBAll_Revolute-16"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                qadr = int(self.model.jnt_qposadr[jid])
                self.data.qpos[qadr] += self.np_random.uniform(
                    -self.reset_tilt_rad, self.reset_tilt_rad
                )

        mujoco.mj_forward(self.model, self.data)
        self.prev_tip_xy = self._tip_xy().copy()
        self.ring_center_xy = self._ring_center_xy()
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

        tip_xy = self._tip_xy()
        slide_xy = self._slide_xy()
        trolley_xy = self._trolley_xy()

        tip_err = tip_xy - self.tip_xy_ref
        slide_err = slide_xy - self.slide_xy_ref
        trolley_err = trolley_xy - self.trolley_xy_ref

        dist_tip_xy = float(np.linalg.norm(tip_err))
        dist_slide_xy = float(np.linalg.norm(slide_err))
        dist_trolley_xy = float(np.linalg.norm(trolley_err))

        dt_step = dt * self.frame_skip
        tip_vxy = (tip_xy - self.prev_tip_xy) / max(dt_step, 1e-8)
        tip_speed_xy = float(np.linalg.norm(tip_vxy))

        ring_min_dist = self._tip_ring_min_dist_xy(tip_xy)
        near_ring = ring_min_dist < self.ring_near_m
        touching_ring = ring_min_dist < self.ring_touch_m
        near_target = dist_tip_xy < self.target_near_m

        dist_tip_sq = float(np.dot(tip_err, tip_err))
        dist_slide_sq = float(np.dot(slide_err, slide_err))
        dist_trolley_sq = float(np.dot(trolley_err, trolley_err))

        reward = (
            self.reward_tip_scale * float(np.exp(-self.reward_tip_decay * dist_tip_sq))
            + self.reward_slide_scale * float(np.exp(-self.reward_slide_decay * dist_slide_sq))
            + self.reward_trolley_scale * float(np.exp(-self.reward_trolley_decay * dist_trolley_sq))
        )

        all_at_origin = (
            dist_tip_xy < self.origin_close_tip_m
            and dist_slide_xy < self.origin_close_slide_m
            and dist_trolley_xy < self.origin_close_trolley_m
        )
        if all_at_origin:
            reward *= self.origin_all_close_amplify

        reward -= self.velocity_xy_penalty * tip_speed_xy

        if near_target and not near_ring:
            reward -= 0.03 * tip_speed_xy

        ring_margin_penalty = max(0.0, self.ring_near_m - ring_min_dist)
        reward -= 25.0 * ring_margin_penalty * ring_margin_penalty
        if touching_ring:
            reward -= 0.8

        if near_ring and tip_speed_xy < self.stuck_speed_m_s:
            self.ring_stuck_counter += 1
        else:
            self.ring_stuck_counter = 0
        stuck_near_ring = self.ring_stuck_counter >= self.stuck_steps
        if stuck_near_ring:
            reward -= 1.5

        self.prev_tip_xy = tip_xy.copy()
        self.step_count += 1
        terminated = bool(dist_tip_xy > self.fail_radius_m)
        truncated = bool(self.step_count >= self.max_steps)

        ring_dx = float(tip_xy[0] - self.ring_center_xy[0])
        ring_dy = float(tip_xy[1] - self.ring_center_xy[1])

        info = {
            "tip_xy": tip_xy,
            "slide_xy": slide_xy,
            "tip_xy_ref": self.tip_xy_ref,
            "tip_dist": dist_tip_xy,
            "slide_xy_dist": dist_slide_xy,
            "trolley_xy_dist": dist_trolley_xy,
            "tip_speed": tip_speed_xy,
            "ring_dx": ring_dx,
            "ring_dy": ring_dy,
            "ring_min_dist": ring_min_dist,
            "near_ring": near_ring,
            "touching_ring": touching_ring,
            "stuck_near_ring": stuck_near_ring,
            "ring_stuck_counter": self.ring_stuck_counter,
            "all_at_origin": all_at_origin,
        }
        return self._obs(), reward, terminated, truncated, info

