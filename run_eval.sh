#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Does not use whatever venv is active in your shell — expects ./.venv next to this script.
VENV_PY=""
if [[ -x ".venv/bin/python" ]]; then
  VENV_PY=".venv/bin/python"
elif [[ -x ".venv/bin/python3" ]]; then
  VENV_PY=".venv/bin/python3"
fi
if [[ -z "$VENV_PY" ]]; then
  echo "No executable interpreter at .venv/bin/python or .venv/bin/python3." >&2
  echo "From this directory run:  ./setup_venv.sh" >&2
  echo "(Activating a venv elsewhere does not satisfy this script.)" >&2
  exit 1
fi
if [[ ! -x ".venv/bin/mjpython" ]]; then
  echo "Missing .venv/bin/mjpython (installed with MuJoCo). Re-run ./setup_venv.sh or: pip install mujoco" >&2
  exit 1
fi

MODEL_PATH="${1:-ppo_nominal_xy_align.zip}"
EPISODES="${2:-5}"

exec "$VENV_PY" ".venv/bin/mjpython" -m rl.eval_viewer \
  --model-path "$MODEL_PATH" \
  --episodes "$EPISODES" \
  --deterministic
