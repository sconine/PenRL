from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from rl.pen_balance_env import TrolleyCircleEnv


class CircleTrackCallback(BaseCallback):
    """Logs trolley circle-tracking metrics from env info."""

    def __init__(self) -> None:
        super().__init__()
        self._tracking_err: list[float] = []
        self._radius_err: list[float] = []
        self._success: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "tracking_error_m" in info:
                self._tracking_err.append(float(info["tracking_error_m"]))
            if "radius_error_m" in info:
                self._radius_err.append(float(info["radius_error_m"]))
            if "step_success" in info:
                self._success.append(float(bool(info["step_success"])))
        return True

    def _on_rollout_end(self) -> None:
        if self._tracking_err:
            self.logger.record("circle/mean_tracking_error_m", float(np.mean(self._tracking_err)))
        if self._radius_err:
            self.logger.record("circle/mean_radius_error_m", float(np.mean(self._radius_err)))
        if self._success:
            self.logger.record("circle/success_step_fraction", float(np.mean(self._success)))
        self._tracking_err.clear()
        self._radius_err.clear()
        self._success.clear()


def main() -> None:
    env = TrolleyCircleEnv(model_path=Path("PModel/PModel.urdf"))
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
        tensorboard_log="runs/ppo_trolley_circle",
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="checkpoints",
        name_prefix="ppo_trolley_circle",
    )
    callbacks = CallbackList([CircleTrackCallback(), checkpoint_cb])
    model.learn(total_timesteps=200_000, callback=callbacks)
    model.save("ppo_trolley_circle")
    print("Saved model to ppo_trolley_circle.zip")


if __name__ == "__main__":
    main()
