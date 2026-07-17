#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'USAGE'
Usage: bash scripts/workspace/run_workspace.sh [options]

Start the WorldFoundry Studio workspace.

Options:
  --host HOST          Bind host. Default: WORLDFOUNDRY_WORKSPACE_HOST or 127.0.0.1.
  --port PORT          Bind port. Default: WORLDFOUNDRY_WORKSPACE_PORT or 7870.
  --max-jobs N         Concurrent job workers. Default: WORLDFOUNDRY_WORKSPACE_MAX_JOBS or 8.
  --ckpt-dir PATH      Checkpoint/model root. Default order: explicit env, repo-sibling ckpt/, env-file, ~/.cache/worldfoundry/checkpoints.
  --data-dir PATH      Benchmark/data root. Default order: explicit env, repo-sibling data/, env-file, ~/.cache/worldfoundry/data.
                      Most model weights are resolved by Hugging Face from repo ids; set HF_HOME or
                      HF_HUB_CACHE when you want a specific local HF cache.
  --env-file PATH      Sourceable env exports. Default: WORLDFOUNDRY_ENV_FILE or tmp/worldfoundry_unified_env.sh.
  --python PATH        Python executable. Defaults to the unified env python when available.
  -h, --help           Show this help.

The script expects the unified runtime environment created by
scripts/setup/bootstrap_worldfoundry.sh or scripts/setup/unified_install.sh.
USAGE
}

WORLDFOUNDRY_SOURCE_ROOT="$ROOT"
ROOT_PARENT="$(cd "$ROOT/.." && pwd)"

PRESET_CKPT_DIR="${WORLDFOUNDRY_CKPT_DIR:-}"
PRESET_DATA_DIR="${WORLDFOUNDRY_DATA_DIR:-}"
PRESET_BENCHMARK_DATA_ROOT="${WORLDFOUNDRY_BENCHMARK_DATA_ROOT:-}"
PRESET_HFD_DATASET_ROOT="${WORLDFOUNDRY_HFD_DATASET_ROOT:-}"
PRESET_HFD_ROOT="${WORLDFOUNDRY_HFD_ROOT:-}"

HOST="${WORLDFOUNDRY_WORKSPACE_HOST:-127.0.0.1}"
PORT="${WORLDFOUNDRY_WORKSPACE_PORT:-7870}"
MAX_JOBS="${WORLDFOUNDRY_WORKSPACE_MAX_JOBS:-8}"
CKPT_DIR_OVERRIDE=""
DATA_DIR_OVERRIDE=""
ENV_FILE="${WORLDFOUNDRY_ENV_FILE:-tmp/worldfoundry_unified_env.sh}"
PYTHON_BIN="${PYTHON:-}"

while (($#)); do
  case "$1" in
    --host)
      HOST="$2"
      shift
      ;;
    --port)
      PORT="$2"
      shift
      ;;
    --max-jobs)
      MAX_JOBS="$2"
      shift
      ;;
    --ckpt-dir)
      CKPT_DIR_OVERRIDE="$2"
      shift
      ;;
    --data-dir)
      DATA_DIR_OVERRIDE="$2"
      shift
      ;;
    --env-file)
      ENV_FILE="$2"
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -z "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX:-}" && -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

choose_asset_dir() {
  local override="$1"
  local preset="$2"
  local env_value="$3"
  local sibling="$4"
  local fallback="$5"
  if [[ -n "$override" ]]; then
    printf '%s\n' "$override"
  elif [[ -n "$preset" ]]; then
    printf '%s\n' "$preset"
  elif [[ -d "$sibling" ]]; then
    printf '%s\n' "$sibling"
  elif [[ -n "$env_value" ]]; then
    printf '%s\n' "$env_value"
  else
    printf '%s\n' "$fallback"
  fi
}

CKPT_DIR="$(choose_asset_dir "$CKPT_DIR_OVERRIDE" "$PRESET_CKPT_DIR" "${WORLDFOUNDRY_CKPT_DIR:-}" "${ROOT_PARENT}/ckpt" "${HOME}/.cache/worldfoundry/checkpoints")"
DATA_DIR="$(choose_asset_dir "$DATA_DIR_OVERRIDE" "$PRESET_DATA_DIR" "${WORLDFOUNDRY_DATA_DIR:-}" "${ROOT_PARENT}/data" "${HOME}/.cache/worldfoundry/data")"
HFD_ROOT="${PRESET_HFD_ROOT:-${CKPT_DIR}/hfd}"
BENCHMARK_DATA_ROOT="${PRESET_BENCHMARK_DATA_ROOT:-${DATA_DIR}/datasets}"
HFD_DATASET_ROOT="${PRESET_HFD_DATASET_ROOT:-${DATA_DIR}}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX:-}" && -x "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}/bin/python"
  elif [[ -n "${WORLDFOUNDRY_CONDA_ENV_PREFIX:-}" && -x "${WORLDFOUNDRY_CONDA_ENV_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${WORLDFOUNDRY_CONDA_ENV_PREFIX}/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

check_python_executable() {
  local label="$1"
  local python_bin="$2"
  local output
  if ! output="$("$python_bin" - <<'PY' 2>&1
import ssl
import sys

print(sys.executable)
print(sys.version.split()[0])
print(ssl.OPENSSL_VERSION)
PY
)"; then
    cat >&2 <<EOF
${label} is not executable.

Python: ${python_bin}
Output:
${output}

This usually means the conda environment was moved, copied, or incompletely
created. Recreate it through the public setup path instead of reusing a
relocated environment:

  bash scripts/setup/bootstrap_worldfoundry.sh
  source tmp/worldfoundry_unified_env.sh
  bash scripts/workspace/run_workspace.sh
EOF
    exit 1
  fi
}

export PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WORLDFOUNDRY_WORKSPACE_MAX_JOBS="$MAX_JOBS"
# Unless explicitly overridden, give every scheduled job a resident-worker
# slot.  This is what lets large video models reuse loaded weights instead of
# silently falling back to a fresh one-shot process once two jobs are active.
export WORLDFOUNDRY_STUDIO_RESIDENT_WORKER_MAX_WORKERS="${WORLDFOUNDRY_STUDIO_RESIDENT_WORKER_MAX_WORKERS:-$MAX_JOBS}"
export WORLDFOUNDRY_CKPT_DIR="$CKPT_DIR"
export WORLDFOUNDRY_DATA_DIR="$DATA_DIR"
export WORLDFOUNDRY_BENCHMARK_DATA_ROOT="$BENCHMARK_DATA_ROOT"
export WORLDFOUNDRY_HFD_DATASET_ROOT="$HFD_DATASET_ROOT"
export WORLDFOUNDRY_HFD_ROOT="$HFD_ROOT"
export PYTHON="$PYTHON_BIN"
export WORLDFOUNDRY_STUDIO_CHILD_PYTHON="$PYTHON_BIN"

check_python_executable "Workspace Python" "$PYTHON_BIN"
if [[ -n "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX:-}" && -x "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}/bin/python" ]]; then
  check_python_executable "Unified worker Python" "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}/bin/python"
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import fastapi
import uvicorn
import worldfoundry.studio.workspace_app
PY
then
  cat >&2 <<EOF
The selected Python cannot import the Workspace runtime dependencies.

Python: ${PYTHON_BIN}

Run:
  bash scripts/setup/bootstrap_worldfoundry.sh
  source tmp/worldfoundry_unified_env.sh
  bash scripts/workspace/run_workspace.sh
EOF
  exit 1
fi

echo "Starting WorldFoundry workspace at http://${HOST}:${PORT}/"
echo "PYTHON=${PYTHON_BIN}"
echo "WORLDFOUNDRY_CKPT_DIR=${WORLDFOUNDRY_CKPT_DIR}"
echo "WORLDFOUNDRY_DATA_DIR=${WORLDFOUNDRY_DATA_DIR}"
echo "HF_HOME=${HF_HOME:-<huggingface default>}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE:-<huggingface default>}"
exec "$PYTHON_BIN" -m worldfoundry.studio.workspace_app --host "$HOST" --port "$PORT"
