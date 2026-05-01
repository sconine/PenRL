from __future__ import annotations

import numpy as np

from rl.pen_balance_env import TrolleyCircleEnv


def main() -> None:
    env = TrolleyCircleEnv()
    obs, _ = env.reset(seed=0)
    print("reset obs:", np.round(obs, 5))
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
                f"track_err_m={info['tracking_error_m']:.5f} "
                f"radius_err_m={info['radius_error_m']:.5f} "
                f"success={info['step_success']}"
            )
        if terminated or truncated:
            print("episode ended at step", i + 1, "terminated:", terminated, "truncated:", truncated)
            break
    print("total_reward:", round(total, 3), "success_steps:", successes)


if __name__ == "__main__":
    main()
