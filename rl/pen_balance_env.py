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
    """Reduce pen vs ring tunneling: thicker contacts, finer stepping, multi-point CCD, stiff overrides.

    Common causes of visually slipping through ring segments:
    - Discrete simulation steps vs thin collision geometry.
    - Wedge-shaped gaps between rotated boxes on a low-count polygonal hull.
    - Short vertical slab extent vs a tilted pen (misses collision entirely).
    """

    opt = model.opt
    opt.timestep = float(min(opt.timestep, 0.0005))
    opt.iterations = int(max(opt.iterations, 120))
    if hasattr(opt, "ccd_iterations"):
        opt.ccd_iterations = int(max(int(opt.ccd_iterations), 80))

    enable = getattr(mujoco, "mjtEnableBit", None)
    mj_multiccd = getattr(enable, "mjENBL_MULTICCD", None) if enable is not None else None
    mj_override = getattr(enable, "mjENBL_OVERRIDE", None) if enable is not None else None
    if mj_multiccd is not None:
        opt.enableflags |= int(mj_multiccd)
    else:
        opt.enableflags |= 1 << 5  # mjENBL_MULTICCD
    if mj_override is not None:
        opt.enableflags |= int(mj_override)
    else:
        opt.enableflags |= 1 << 0  # mjENBL_OVERRIDE (per-geom solref/solimp)

    ring_margin = 0.0035
    pen_margin = 0.0025
    stiff_solref = np.array([0.02, 1.0], dtype=np.float64)
    stiff_solimp = np.array([0.99, 0.995, 0.001, 0.5, 2.0], dtype=np.float64)

    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("Ring_collision_seg_"):
            model.geom_margin[gid] = max(float(model.geom_margin[gid]), ring_margin)
            model.geom_gap[gid] = 0.0
            model.geom_solref[gid, :] = stiff_solref
            model.geom_solimp[gid, :] = stiff_solimp
            model.geom_friction[gid, 0] = max(float(model.geom_friction[gid, 0]), 1.0)
        elif name in ("PenTip_collision", "Pen_collision"):
            model.geom_margin[gid] = max(float(model.geom_margin[gid]), pen_margin)
            model.geom_gap[gid] = 0.0
            model.geom_solref[gid, :] = stiff_solref
            model.geom_solimp[gid, :] = stiff_solimp


class NominalXYAlignEnv(gym.Env[np.ndarray, np.ndarray]):
    """Hold slide, trolley, and pen tip near the URDF nominal pose (prismatic + pen joints at zero).

    Carriage motion uses a **velocity servo** on the two prismatic DOFs (``qfrc_applied``), not direct
    ``qpos`` integration. Writing slide/trolley positions every step would kinematically drag the pen
    through obstacles and ignore ring contacts; letting ``mj_step`` integrate under applied forces keeps
    collisions meaningful (consistent with passive MuJoCo viewer behavior).

    Carriage motion follows motor mixing from **logical motor shaft angles** tracked in the env; each motor
    angle is clamped to ``± max_motor_revolutions · 2π`` rad from its episode start (default ±1 revolution).
    Slide/trolley joint limits are left at **URDF values** (no env override).

    When the trolley wanders beyond ``trolley_recovery_far_m`` from **this episode's** starting XY (stored at
    reset), reward includes progress toward that start plus a mild penalty for excess displacement. A small
    **motor activity** bonus and lower action-smoothing penalties encourage varied commands (combine with
    higher PPO ``ent_coef`` in training).

    Tune ``carriage_velocity_kp`` / ``carriage_velocity_kd`` if tracking is too loose or the model oscillates.

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
        velocity_xy_penalty: float = 0.028,
        action_delta_penalty: float = 0.004,
        trolley_recovery_far_m: float = 0.005,
        trolley_recovery_progress_scale: float = 180.0,
        trolley_far_quadratic_scale: float = 250.0,
        motor_activity_bonus_scale: float = 0.055,
        success_nominal_m: float = 0.004,
        success_spread_m2: float = 5e-5,
        max_motor_revolutions: float = 1.0,
        carriage_velocity_kp: float = 600.0,
        carriage_velocity_kd: float = 35.0,
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
        self.trolley_recovery_far_m = float(trolley_recovery_far_m)
        self.trolley_recovery_progress_scale = float(trolley_recovery_progress_scale)
        self.trolley_far_quadratic_scale = float(trolley_far_quadratic_scale)
        self.motor_activity_bonus_scale = float(motor_activity_bonus_scale)
        self.success_nominal_m = float(success_nominal_m)
        self.success_spread_m2 = float(success_spread_m2)
        self.max_motor_revolutions = float(max_motor_revolutions)
        self.motor_angle_limit_rad = float(max_motor_revolutions) * (2.0 * np.pi)
        self.carriage_velocity_kp = float(carriage_velocity_kp)
        self.carriage_velocity_kd = float(carriage_velocity_kd)

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

        self.stepper_a_qadr = self._optional_joint_qpos_adr("StepperA_joint")
        self.stepper_b_qadr = self._optional_joint_qpos_adr("StepperB_joint")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # XY errors vs nominal (6) + velocities (6) + trolley offset from episode start (2).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        self._motor_theta_a = 0.0
        self._motor_theta_b = 0.0

        self._capture_nominal_xy_refs()
        self._sync_stepper_visual_pose()
        mujoco.mj_forward(self.model, self.data)

        self._episode_trolley_xy_start = self._trolley_xy().copy()
        self.prev_episode_trolley_dist = 0.0

    def _optional_joint_qpos_adr(self, joint_name: str) -> int | None:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return None
        return int(self.model.jnt_qposadr[jid])

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

    def _sync_stepper_visual_pose(self) -> None:
        """Align URDF stepper joint angles with integrated logical motor angles (viewer/debug)."""
        if self.stepper_a_qadr is not None:
            self.data.qpos[self.stepper_a_qadr] = self._motor_theta_a
        if self.stepper_b_qadr is not None:
            self.data.qpos[self.stepper_b_qadr] = self._motor_theta_b

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

        ep_dx = (trolley_xy[0] - self._episode_trolley_xy_start[0]) / pos_scale
        ep_dy = (trolley_xy[1] - self._episode_trolley_xy_start[1]) / pos_scale

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
                ep_dx,
                ep_dy,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float64)
        mujoco.mj_resetData(self.model, self.data)
        self._motor_theta_a = 0.0
        self._motor_theta_b = 0.0
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
        self._sync_stepper_visual_pose()
        mujoco.mj_forward(self.model, self.data)

        tw = self._trolley_xy()
        self._episode_trolley_xy_start = tw.copy()
        self.prev_episode_trolley_dist = 0.0

        self.prev_slide_xy = self._slide_xy()
        self.prev_trolley_xy = tw.copy()
        self.prev_tip_xy = self._tip_xy()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        motor_a = float(action[0] * self.max_motor_speed_rad_s)
        motor_b = float(action[1] * self.max_motor_speed_rad_s)

        dt = self.model.opt.timestep
        lim = self.motor_angle_limit_rad
        for _ in range(self.frame_skip):
            ta_next = float(np.clip(self._motor_theta_a + motor_a * dt, -lim, lim))
            tb_next = float(np.clip(self._motor_theta_b + motor_b * dt, -lim, lim))
            eff_a = (ta_next - self._motor_theta_a) / max(dt, 1e-12)
            eff_b = (tb_next - self._motor_theta_b) / max(dt, 1e-12)
            self._motor_theta_a = ta_next
            self._motor_theta_b = tb_next

            vx, vy = self.mix.to_linear_xy(eff_a, eff_b)
            self.data.qfrc_applied[:] = 0.0
            v_slide = float(self.data.qvel[self.slide_dof_adr])
            v_trolley = float(self.data.qvel[self.trolley_dof_adr])
            self.data.qfrc_applied[self.slide_dof_adr] = self.carriage_velocity_kp * (
                vy - v_slide
            ) - self.carriage_velocity_kd * v_slide
            self.data.qfrc_applied[self.trolley_dof_adr] = self.carriage_velocity_kp * (
                vx - v_trolley
            ) - self.carriage_velocity_kd * v_trolley
            mujoco.mj_step(self.model, self.data)

        self._sync_stepper_visual_pose()
        mujoco.mj_forward(self.model, self.data)

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

        dist_episode_home = float(np.linalg.norm(trolley_xy - self._episode_trolley_xy_start))
        reward_recovery = 0.0
        if dist_episode_home > self.trolley_recovery_far_m:
            reward_recovery += self.trolley_recovery_progress_scale * max(
                0.0, self.prev_episode_trolley_dist - dist_episode_home
            )
            excess = dist_episode_home - self.trolley_recovery_far_m
            reward_recovery -= self.trolley_far_quadratic_scale * excess * excess
        reward += reward_recovery

        reward += self.motor_activity_bonus_scale * float(np.mean(np.abs(action)))

        tip_vxy = (tip_xy - self.prev_tip_xy) / max(dt_step, 1e-8)
        tip_speed = float(np.linalg.norm(tip_vxy))
        reward -= self.velocity_xy_penalty * tip_speed

        da = action - self.prev_action
        reward -= self.action_delta_penalty * float(np.dot(da, da))
        self.prev_action = action.copy()

        self.prev_episode_trolley_dist = dist_episode_home

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
            "motor_theta_a": self._motor_theta_a,
            "motor_theta_b": self._motor_theta_b,
            "motor_angle_limit_rad": self.motor_angle_limit_rad,
            "trolley_dist_episode_home_m": dist_episode_home,
            "reward_recovery": reward_recovery,
        }
        return self._obs(), float(reward), terminated, truncated, info


PenBalanceEnv = NominalXYAlignEnv
