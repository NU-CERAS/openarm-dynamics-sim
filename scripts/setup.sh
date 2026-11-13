#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
venv_python() {
  # POSIX venvs use bin/, Windows (Git Bash/MSYS) venvs use Scripts/.
  for p in .venv/bin/python .venv/Scripts/python.exe; do
    [[ -x $p ]] && { echo "$p"; return; }
  done
}
if [[ -z "$(venv_python)" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3.13}"
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo 'Install Python 3.13 (macOS: brew install python@3.13), then rerun.' >&2
    exit 1
  fi
  "$PYTHON_BIN" -m venv .venv
fi
VENV_PY="$(venv_python)"
"$VENV_PY" -m pip install -c requirements-lock.txt -e '.[dev]'
"$VENV_PY" -c 'import mujoco; print("MuJoCo", mujoco.__version__, "ready")'
echo 'Setup complete. Run ./scripts/start.sh'
