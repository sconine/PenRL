from __future__ import annotations

import numpy as np

from rl.pen_balance_env import PenBalanceEnv


def main() -> None:
    env = PenBalanceEnv()
    obs, _ = env.reset(seed=0)
    print("reset obs:", np.round(obs, 5))
    total = 0.0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if (i + 1) % 20 == 0:
            tip_xy = np.asarray(info.get("tip_xy", [np.nan, np.nan]), dtype=np.float32)
            print(
                f"step={i+1} reward={reward:.3f} tip_xy_dist={info['tip_dist']:.4f} "
                f"speed={info['tip_speed']:.4f} "
                f"tip_x={tip_xy[0]:.4f} tip_y={tip_xy[1]:.4f} "
                f"origin={info.get('all_at_origin', False)}"
            )
        if terminated or truncated:
            print("episode ended at step", i + 1, "terminated:", terminated, "truncated:", truncated)
            break
    print("total_reward:", round(total, 3))


if __name__ == "__main__":
    main()
