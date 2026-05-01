from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from rl.pen_balance_env import PenBalanceEnv


class RingMetricsCallback(BaseCallback):
    """Logs ring-avoidance and XY diagnostics from env info dictionaries."""

    def __init__(self) -> None:
        super().__init__()
        self._ring_min_dist: list[float] = []
        self._near_ring: list[float] = []
        self._touching_ring: list[float] = []
        self._stuck_near_ring: list[float] = []
        self._tip_x: list[float] = []
        self._tip_y: list[float] = []
        self._tip_dist: list[float] = []
        self._slide_x: list[float] = []
        self._slide_y: list[float] = []
        self._slide_dist: list[float] = []
        self._trolley_x: list[float] = []
        self._trolley_y: list[float] = []
        self._trolley_dist: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "ring_min_dist" in info:
                self._ring_min_dist.append(float(info["ring_min_dist"]))
                self._near_ring.append(float(bool(info.get("near_ring", False))))
                self._touching_ring.append(float(bool(info.get("touching_ring", False))))
                self._stuck_near_ring.append(float(bool(info.get("stuck_near_ring", False))))
            tip_xy = info.get("tip_xy")
            if tip_xy is not None and "tip_dist" in info:
                tip_xy = np.asarray(tip_xy, dtype=np.float64).reshape(-1)
                if tip_xy.size >= 2:
                    self._tip_x.append(float(tip_xy[0]))
                    self._tip_y.append(float(tip_xy[1]))
                    self._tip_dist.append(float(info["tip_dist"]))
            slide_xy = info.get("slide_xy")
            if slide_xy is not None and "slide_xy_dist" in info:
                slide_xy = np.asarray(slide_xy, dtype=np.float64).reshape(-1)
                if slide_xy.size >= 2:
                    self._slide_x.append(float(slide_xy[0]))
                    self._slide_y.append(float(slide_xy[1]))
                    self._slide_dist.append(float(info["slide_xy_dist"]))
            trolley_xy = info.get("trolley_xy")
            if trolley_xy is not None and "trolley_xy_dist" in info:
                trolley_xy = np.asarray(trolley_xy, dtype=np.float64).reshape(-1)
                if trolley_xy.size >= 2:
                    self._trolley_x.append(float(trolley_xy[0]))
                    self._trolley_y.append(float(trolley_xy[1]))
                    self._trolley_dist.append(float(info["trolley_xy_dist"]))
        return True

    def _on_rollout_end(self) -> None:
        if self._ring_min_dist:
            self.logger.record("ring/min_dist_mean", float(np.mean(self._ring_min_dist)))
            self.logger.record("ring/near_fraction", float(np.mean(self._near_ring)))
            self.logger.record("ring/touching_fraction", float(np.mean(self._touching_ring)))
            self.logger.record("ring/stuck_fraction", float(np.mean(self._stuck_near_ring)))
        if self._tip_x:
            self.logger.record("tip/x_mean", float(np.mean(self._tip_x)))
            self.logger.record("tip/y_mean", float(np.mean(self._tip_y)))
            self.logger.record("tip/dist_mean", float(np.mean(self._tip_dist)))
        if self._slide_x:
            self.logger.record("slide/x_mean", float(np.mean(self._slide_x)))
            self.logger.record("slide/y_mean", float(np.mean(self._slide_y)))
            self.logger.record("slide/dist_mean", float(np.mean(self._slide_dist)))
        if self._trolley_x:
            self.logger.record("trolley/x_mean", float(np.mean(self._trolley_x)))
            self.logger.record("trolley/y_mean", float(np.mean(self._trolley_y)))
            self.logger.record("trolley/dist_mean", float(np.mean(self._trolley_dist)))
        self._ring_min_dist.clear()
        self._near_ring.clear()
        self._touching_ring.clear()
        self._stuck_near_ring.clear()
        self._tip_x.clear()
        self._tip_y.clear()
        self._tip_dist.clear()
        self._slide_x.clear()
        self._slide_y.clear()
        self._slide_dist.clear()
        self._trolley_x.clear()
        self._trolley_y.clear()
        self._trolley_dist.clear()


def main() -> None:
    env = PenBalanceEnv(model_path=Path("PModel/PModel.urdf"))
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
        tensorboard_log="runs/ppo_pen_balance",
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="checkpoints",
        name_prefix="ppo_pen_balance",
    )
    callbacks = CallbackList([RingMetricsCallback(), checkpoint_cb])
    model.learn(total_timesteps=200_000, callback=callbacks)
    model.save("ppo_pen_balance")
    print("Saved model to ppo_pen_balance.zip")


if __name__ == "__main__":
    main()
