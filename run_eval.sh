#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv/bin/python. Create/activate your virtual environment first."
  exit 1
fi

MODEL_PATH="${1:-ppo_pen_balance.zip}"
EPISODES="${2:-5}"

exec ".venv/bin/python" ".venv/bin/mjpython" -m rl.eval_viewer \
  --model-path "$MODEL_PATH" \
  --episodes "$EPISODES" \
  --deterministic
