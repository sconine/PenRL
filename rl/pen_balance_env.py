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


def _pairwise_xy_sq_sum(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Sum of squared XY distances between three 2D points."""
    return (
        float(np.dot(a - b, a - b))
        + float(np.dot(a - c, a - c))
        + float(np.dot(b - c, b - c))
    )


def _configure_pen_ring_contacts(model: mujoco.MjModel) -> None:
    """Reduce pen vs ring tunneling: thicker contacts, finer stepping, multi-point CCD when available.

    Common causes of visually slipping through ring segments:
    - Discrete simulation steps vs thin box collision (especially after shrinking segment thickness).
    - Small wedge-shaped gaps between neighboring rotated boxes on an octagonal hull.
    """

    opt = model.opt
    opt.timestep = float(min(opt.timestep, 0.001))
    opt.iterations = int(max(opt.iterations, 80))
    if hasattr(opt, "ccd_iterations"):
        opt.ccd_iterations = int(max(int(opt.ccd_iterations), 50))

    mj_multiccd = getattr(getattr(mujoco, "mjtEnableBit", None), "mjENBL_MULTICCD", None)
    if mj_multiccd is not None:
        opt.enableflags |= int(mj_multiccd)
    else:
        opt.enableflags |= 1 << 5  # mjENBL_MULTICCD

    ring_margin = 0.002
    pen_margin = 0.0015
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("Ring_collision_seg_"):
            model.geom_margin[gid] = max(float(model.geom_margin[gid]), ring_margin)
        elif name in ("PenTip_collision", "Pen_collision"):
            model.geom_margin[gid] = max(float(model.geom_margin[gid]), pen_margin)


class NominalXYAlignEnv(gym.Env[np.ndarray, np.ndarray]):
    """Hold slide, trolley, and pen tip near the URDF nominal pose (prismatic + pen joints at zero).

    Slide and trolley prismatic travel are each capped at ``max_prismatic_travel_m`` from joint zero
    (joints ``Plate_Slider-7`` / ``Slide_Slider-8`` in the URDF; default ±20 mm per axis).

    Nominal XY references are captured once from MuJoCo with all those joints at zero. Reward encourages
    each body's XY to stay near its reference and the trio's relative XY layout (pairwise spread) to stay
    near the nominal spread—so coordinates stay similar to each other and to the designed reset pose.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str | Path = "PModel/PModel.urdf",
        frame_skip: int = 5,
        max_motor_speed_rad_s: float = 14.0,
        max_steps: int = 600,
        reset_tilt_rad: float = 0.06,
        reward_tip_scale: float = 2.0,
        reward_slide_scale: float = 1.25,
        reward_trolley_scale: float = 1.25,
        reward_align_scale: float = 1.5,
        reward_tip_decay: float = 350.0,
        reward_slide_decay: float = 4500.0,
        reward_trolley_decay: float = 4500.0,
        reward_align_decay: float = 2.0e8,
        velocity_xy_penalty: float = 0.06,
        action_delta_penalty: float = 0.015,
        success_nominal_m: float = 0.004,
        success_spread_m2: float = 5e-5,
        # ±travel from zero for slide and trolley prismatic joints (meters, each axis).
        max_prismatic_travel_m: float = 0.02,
    ) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        _configure_pen_ring_contacts(self.model)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_motor_speed_rad_s = max_motor_speed_rad_s
        self.max_steps = max_steps
        self.reset_tilt_rad = float(reset_tilt_rad)
        self.reward_tip_scale = float(reward_tip_scale)
        self.reward_slide_scale = float(reward_slide_scale)
        self.reward_trolley_scale = float(reward_trolley_scale)
        self.reward_align_scale = float(reward_align_scale)
        self.reward_tip_decay = float(reward_tip_decay)
        self.reward_slide_decay = float(reward_slide_decay)
        self.reward_trolley_decay = float(reward_trolley_decay)
        self.reward_align_decay = float(reward_align_decay)
        self.velocity_xy_penalty = float(velocity_xy_penalty)
        self.action_delta_penalty = float(action_delta_penalty)
        self.success_nominal_m = float(success_nominal_m)
        self.success_spread_m2 = float(success_spread_m2)
        lim = float(max_prismatic_travel_m)
        self.slide_min, self.slide_max = -lim, lim
        self.trolley_min, self.trolley_max = -lim, lim

        self.mix = MotorMixing()
        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.prev_slide_xy = np.zeros(2, dtype=np.float64)
        self.prev_trolley_xy = np.zeros(2, dtype=np.float64)
        self.prev_tip_xy = np.zeros(2, dtype=np.float64)

        self.slide_jid = self._joint_id("Plate_Slider-7")
        self.trolley_jid = self._joint_id("Slide_Slider-8")
        self.slide_bid = self._body_id("Slide")
        self.trolley_bid = self._body_id("Trolley")
        self.pen_tip_gid = self._geom_id_any("PenTip_collision", "Pen_collision")

        self.slide_qpos_adr = int(self.model.jnt_qposadr[self.slide_jid])
        self.trolley_qpos_adr = int(self.model.jnt_qposadr[self.trolley_jid])
        self.slide_dof_adr = int(self.model.jnt_dofadr[self.slide_jid])
        self.trolley_dof_adr = int(self.model.jnt_dofadr[self.trolley_jid])

        lim_rng = np.array([-lim, lim], dtype=np.float64)
        self.model.jnt_range[self.slide_jid] = lim_rng
        self.model.jnt_range[self.trolley_jid] = lim_rng

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # XY errors vs nominal (6) + XY velocities for slide / trolley / tip (6).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        self._capture_nominal_xy_refs()

    def _capture_nominal_xy_refs(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.slide_qpos_adr] = 0.0
        self.data.qpos[self.trolley_qpos_adr] = 0.0
        for joint_name in ("Trolley_Revolute-15", "YBAll_Revolute-16"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                qadr = int(self.model.jnt_qposadr[jid])
                self.data.qpos[qadr] = 0.0
        mujoco.mj_forward(self.model, self.data)

        s = np.asarray(self.data.xpos[self.slide_bid, :2], dtype=np.float64).copy()
        t = np.asarray(self.data.xpos[self.trolley_bid, :2], dtype=np.float64).copy()
        p = np.asarray(self.data.geom_xpos[self.pen_tip_gid, :2], dtype=np.float64).copy()
        self.slide_xy_ref = s
        self.trolley_xy_ref = t
        self.tip_xy_ref = p
        self.spread_ref_sq = _pairwise_xy_sq_sum(s, t, p)

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

    def _slide_xy(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.slide_bid, :2], dtype=np.float64).copy()

    def _trolley_xy(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.trolley_bid, :2], dtype=np.float64).copy()

    def _tip_xy(self) -> np.ndarray:
        return self.data.geom_xpos[self.pen_tip_gid, :2].copy()

    def _obs(self) -> np.ndarray:
        slide_xy = self._slide_xy()
        trolley_xy = self._trolley_xy()
        tip_xy = self._tip_xy()
        dt = self.model.opt.timestep * self.frame_skip
        inv_dt = 1.0 / max(dt, 1e-8)
        pos_scale = 0.02
        v_scale = 0.05

        e_slide = (slide_xy - self.slide_xy_ref) / pos_scale
        e_trolley = (trolley_xy - self.trolley_xy_ref) / pos_scale
        e_tip = (tip_xy - self.tip_xy_ref) / pos_scale

        v_slide = (slide_xy - self.prev_slide_xy) * inv_dt / v_scale
        v_trolley = (trolley_xy - self.prev_trolley_xy) * inv_dt / v_scale
        v_tip = (tip_xy - self.prev_tip_xy) * inv_dt / v_scale

        return np.array(
            [
                e_slide[0],
                e_slide[1],
                e_trolley[0],
                e_trolley[1],
                e_tip[0],
                e_tip[1],
                v_slide[0],
                v_slide[1],
                v_trolley[0],
                v_trolley[1],
                v_tip[0],
                v_tip[1],
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float64)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.slide_qpos_adr] = 0.0
        self.data.qpos[self.trolley_qpos_adr] = 0.0
        self.data.qvel[self.slide_dof_adr] = 0.0
        self.data.qvel[self.trolley_dof_adr] = 0.0
        for joint_name in ("Trolley_Revolute-15", "YBAll_Revolute-16"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                qadr = int(self.model.jnt_qposadr[jid])
                self.data.qpos[qadr] += self.np_random.uniform(
                    -self.reset_tilt_rad, self.reset_tilt_rad
                )
        mujoco.mj_forward(self.model, self.data)

        self.prev_slide_xy = self._slide_xy()
        self.prev_trolley_xy = self._trolley_xy()
        self.prev_tip_xy = self._tip_xy()
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
        slide_xy = self._slide_xy()
        trolley_xy = self._trolley_xy()
        tip_xy = self._tip_xy()

        err_slide = slide_xy - self.slide_xy_ref
        err_trolley = trolley_xy - self.trolley_xy_ref
        err_tip = tip_xy - self.tip_xy_ref
        dist_slide = float(np.linalg.norm(err_slide))
        dist_trolley = float(np.linalg.norm(err_trolley))
        dist_tip = float(np.linalg.norm(err_tip))

        spread_sq = _pairwise_xy_sq_sum(slide_xy, trolley_xy, tip_xy)
        spread_delta_sq = (spread_sq - self.spread_ref_sq) ** 2

        reward = (
            self.reward_tip_scale * float(np.exp(-self.reward_tip_decay * float(np.dot(err_tip, err_tip))))
            + self.reward_slide_scale
            * float(np.exp(-self.reward_slide_decay * float(np.dot(err_slide, err_slide))))
            + self.reward_trolley_scale
            * float(np.exp(-self.reward_trolley_decay * float(np.dot(err_trolley, err_trolley))))
            + self.reward_align_scale * float(np.exp(-self.reward_align_decay * spread_delta_sq))
        )

        tip_vxy = (tip_xy - self.prev_tip_xy) / max(dt_step, 1e-8)
        tip_speed = float(np.linalg.norm(tip_vxy))
        reward -= self.velocity_xy_penalty * tip_speed

        da = action - self.prev_action
        reward -= self.action_delta_penalty * float(np.dot(da, da))
        self.prev_action = action.copy()

        nominal_ok = (
            dist_slide < self.success_nominal_m
            and dist_trolley < self.success_nominal_m
            and dist_tip < self.success_nominal_m
        )
        spread_ok = abs(spread_sq - self.spread_ref_sq) < self.success_spread_m2
        step_success = bool(nominal_ok and spread_ok)

        self.prev_slide_xy = slide_xy.copy()
        self.prev_trolley_xy = trolley_xy.copy()
        self.prev_tip_xy = tip_xy.copy()
        self.step_count += 1
        terminated = False
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "slide_xy": slide_xy,
            "trolley_xy": trolley_xy,
            "tip_xy": tip_xy,
            "slide_xy_dist": dist_slide,
            "trolley_xy_dist": dist_trolley,
            "tip_xy_dist": dist_tip,
            "spread_sq": spread_sq,
            "spread_ref_sq": self.spread_ref_sq,
            "spread_delta_abs": abs(spread_sq - self.spread_ref_sq),
            "tip_speed_xy": tip_speed,
            "step_success": step_success,
            "nominal_ok": nominal_ok,
            "spread_ok": spread_ok,
        }
        return self._obs(), float(reward), terminated, truncated, info


PenBalanceEnv = NominalXYAlignEnv
