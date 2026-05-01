from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from rl.pen_balance_env import NominalXYAlignEnv


class AlignHoldCallback(BaseCallback):
    """Logs nominal XY + trio spread metrics from env info."""

    def __init__(self) -> None:
        super().__init__()
        self._d_slide: list[float] = []
        self._d_trolley: list[float] = []
        self._d_tip: list[float] = []
        self._spread_delta: list[float] = []
        self._success: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "slide_xy_dist" in info:
                self._d_slide.append(float(info["slide_xy_dist"]))
            if "trolley_xy_dist" in info:
                self._d_trolley.append(float(info["trolley_xy_dist"]))
            if "tip_xy_dist" in info:
                self._d_tip.append(float(info["tip_xy_dist"]))
            if "spread_delta_abs" in info:
                self._spread_delta.append(float(info["spread_delta_abs"]))
            if "step_success" in info:
                self._success.append(float(bool(info["step_success"])))
        return True

    def _on_rollout_end(self) -> None:
        if self._d_slide:
            self.logger.record("align/mean_slide_xy_err_m", float(np.mean(self._d_slide)))
        if self._d_trolley:
            self.logger.record("align/mean_trolley_xy_err_m", float(np.mean(self._d_trolley)))
        if self._d_tip:
            self.logger.record("align/mean_tip_xy_err_m", float(np.mean(self._d_tip)))
        if self._spread_delta:
            self.logger.record("align/mean_spread_delta_abs", float(np.mean(self._spread_delta)))
        if self._success:
            self.logger.record("align/success_step_fraction", float(np.mean(self._success)))
        self._d_slide.clear()
        self._d_trolley.clear()
        self._d_tip.clear()
        self._spread_delta.clear()
        self._success.clear()


def main() -> None:
    env = NominalXYAlignEnv(model_path=Path("PModel/PModel.urdf"))
    check_env(env, warn=True)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        tensorboard_log="runs/ppo_nominal_xy_align",
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="checkpoints",
        name_prefix="ppo_nominal_xy_align",
    )
    callbacks = CallbackList([AlignHoldCallback(), checkpoint_cb])
    model.learn(total_timesteps=20_000, callback=callbacks)
    model.save("ppo_nominal_xy_align")
    print("Saved model to ppo_nominal_xy_align.zip")


if __name__ == "__main__":
    main()
