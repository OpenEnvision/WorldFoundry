#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_DIR}/3d_metrics:${REPO_DIR}/Depth-Anything-3/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export REWARD_3D_PORT="${REWARD_3D_PORT:-8089}"
export REWARD_3D_USE_LPIPS="${REWARD_3D_USE_LPIPS:-1}"
export REWARD_3D_SCORER="${REWARD_3D_SCORER:-}"
export REWARD_3D_SCORING_MODEL="${REWARD_3D_SCORING_MODEL:-}"
export REWARD_3D_NUM_WORKERS="${REWARD_3D_NUM_WORKERS:-1}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"

cd "${REPO_DIR}"
"${PYTHON_BIN}" 3d_metrics/serve_reward_3d.py "$@"
