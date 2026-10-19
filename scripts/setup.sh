#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3.13}"
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo 'Install Python 3.13 (macOS: brew install python@3.13), then rerun.' >&2
    exit 1
  fi
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install -c requirements-lock.txt -e '.[dev]'
.venv/bin/python -c 'import mujoco; print("MuJoCo", mujoco.__version__, "ready")'
echo 'Setup complete. Run ./scripts/start.sh'
