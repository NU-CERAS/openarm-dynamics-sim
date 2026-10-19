#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Run ./scripts/setup.sh first.' >&2
  exit 1
fi
# The single GLFW window runs on the OS main thread (including macOS).
exec .venv/bin/python -m arm_design.cli view "$@"
