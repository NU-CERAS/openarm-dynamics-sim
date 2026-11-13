#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
VENV_PY=
for p in .venv/bin/python .venv/Scripts/python.exe; do
  [[ -x $p ]] && VENV_PY=$p && break
done
if [[ -z $VENV_PY ]]; then
  echo 'Run ./scripts/setup.sh first.' >&2
  exit 1
fi
# The single GLFW window runs on the OS main thread (including macOS).
exec "$VENV_PY" -m arm_design.cli view "$@"
