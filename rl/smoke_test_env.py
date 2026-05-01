from __future__ import annotations

import numpy as np

from rl.pen_balance_env import NominalXYAlignEnv


def main() -> None:
    env = NominalXYAlignEnv()
    obs, _ = env.reset(seed=0)
    print("reset obs shape:", obs.shape, "sample:", np.round(obs[:6], 5))
    total = 0.0
    successes = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if info.get("step_success"):
            successes += 1
        if (i + 1) % 20 == 0:
            print(
                f"step={i+1} reward={reward:.4f} "
                f"slide={info['slide_xy_dist']:.5f} trolley={info['trolley_xy_dist']:.5f} "
                f"tip={info['tip_xy_dist']:.5f} spreadΔ={info['spread_delta_abs']:.2e} "
                f"success={info['step_success']}"
            )
        if terminated or truncated:
            print("episode ended at step", i + 1, "terminated:", terminated, "truncated:", truncated)
            break
    print("total_reward:", round(total, 3), "success_steps:", successes)


if __name__ == "__main__":
    main()
