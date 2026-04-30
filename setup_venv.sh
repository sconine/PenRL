#!/usr/bin/env bash
# Create .venv in this repo and install Python dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: '${PYTHON}' not found. Install Python 3 or set PYTHON=/path/to/python3." >&2
  exit 1
fi

KERNEL="$(uname -s)"
UNAME_MACHINE="$(uname -m)"
PYTHON_MACHINE="$("$PYTHON" -c 'import platform; print(platform.machine())')"

echo "Processor / OS"
echo "  OS:              $KERNEL"
echo "  Kernel reports:  $UNAME_MACHINE"
echo "  Python reports:  $PYTHON_MACHINE ($(basename "$("$PYTHON" -c 'import sys; print(sys.executable)')"))"

if [[ "$KERNEL" == "Darwin" ]]; then
  case "$PYTHON_MACHINE" in
    arm64)
      echo "  → Treating as:   macOS, Apple Silicon (arm64)"
      ;;
    x86_64)
      echo "  → Treating as:   macOS, Intel (x86_64)"
      ;;
    *)
      echo "  → Treating as:   macOS ($PYTHON_MACHINE)"
      ;;
  esac
  if [[ "$UNAME_MACHINE" != "$PYTHON_MACHINE" ]]; then
    echo "  note: uname and Python differ (common under Rosetta); wheels follow Python's architecture." >&2
  fi
else
  echo "  → Treating as:   $KERNEL ($PYTHON_MACHINE)"
fi

# PyTorch has no Intel-macOS + Python 3.13+ wheels on PyPI.
if [[ "$KERNEL" == "Darwin" && "$PYTHON_MACHINE" == "x86_64" ]] \
  && ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] < (3, 13) else 1)'; then
  echo "error: PyTorch does not publish wheels for Intel macOS with Python 3.13 or newer." >&2
  echo "Install Python 3.12, e.g.: brew install python@3.12" >&2
  echo "Then rerun:" >&2
  echo "  PYTHON=\"\$(brew --prefix python@3.12)/bin/python3.12\" ./setup_venv.sh" >&2
  exit 1
fi

echo "Using: $($PYTHON -c 'import sys; print(sys.version)')"

"$PYTHON" -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel

choose_requirements() {
  if [[ "$KERNEL" == "Darwin" && "$PYTHON_MACHINE" == "x86_64" ]]; then
    echo "$ROOT/requirements-macos-intel.txt"
  else
    echo "$ROOT/requirements.txt"
  fi
}

REQ="$(choose_requirements)"
echo "Installing from: $REQ"

if [[ "$(basename "$REQ")" == "requirements-macos-intel.txt" ]]; then
  pip install -r "$REQ"
else
  # Stage installs so pip does not spend ages backtracking across torch/SB3 versions.
  pip install "numpy>=1.24,<3" "mujoco>=3.1.0" "gymnasium>=0.29.1,<1.3.0" "tensorboard>=2.14.0"
  pip install "torch>=2.3.0,<3.0"
  pip install "stable-baselines3>=2.8.0,<3.0"
fi

echo
echo "Setup complete. Activate the environment with:"
echo "  source \"$ROOT/.venv/bin/activate\""
echo
echo "Examples (from this directory):"
echo "  python gravity.py"
echo "  python -m rl.smoke_test_env"
echo "  python -m rl.train_ppo"
echo "  python -m rl.eval_viewer"
