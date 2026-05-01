from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from rl.pen_balance_env import PenBalanceEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pen-balance policy with live MuJoCo viewer.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="ppo_pen_balance.zip",
        help="Path to Stable-Baselines3 PPO model. If missing, random policy is used.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    parser.add_argument("--max-steps", type=int, default=600, help="Max steps per episode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = PenBalanceEnv(max_steps=args.max_steps)
    model_path = Path(args.model_path)
    model = PPO.load(model_path) if model_path.exists() else None
    if model is None:
        print(f"Model not found at {model_path}; using random actions.")
    else:
        print(f"Loaded model: {model_path}")

    obs, _ = env.reset(seed=0)
    episode_idx = 1
    episode_reward = 0.0
    episode_steps = 0
    dt = env.model.opt.timestep * env.frame_skip

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and episode_idx <= args.episodes:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            action = np.asarray(action, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_steps += 1

            viewer.sync()
            time.sleep(dt)

            if terminated or truncated:
                tip_xy = np.asarray(info.get("tip_xy", [np.nan, np.nan]), dtype=np.float32)
                print(
                    f"episode={episode_idx} steps={episode_steps} "
                    f"reward={episode_reward:.3f} tip_xy_dist={info['tip_dist']:.4f} "
                    f"tip_x={tip_xy[0]:.4f} tip_y={tip_xy[1]:.4f} "
                    f"origin={info.get('all_at_origin', False)}"
                )
                episode_idx += 1
                if episode_idx > args.episodes:
                    break
                obs, _ = env.reset()
                episode_reward = 0.0
                episode_steps = 0

    print("Evaluation ended.")


if __name__ == "__main__":
    main()
