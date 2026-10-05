#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORLDFOUNDRY_SOURCE_ROOT="$ROOT"

CUDA_TIER_REQUEST="${WORLDFOUNDRY_CUDA_PROFILE:-${WORLDFOUNDRY_CUDA_TIER:-auto}}"
ENV_NAME_OVERRIDE="${WORLDFOUNDRY_UNIFIED_ENV_NAME:-}"
ENV_PREFIX_OVERRIDE="${WORLDFOUNDRY_UNIFIED_ENV_PREFIX:-}"
HOME_ROOT="${WORLDFOUNDRY_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/worldfoundry}"
DATA_ROOT_OVERRIDE="${WORLDFOUNDRY_DATA_DIR:-}"
MODEL_ROOT_OVERRIDE="${WORLDFOUNDRY_MODEL_DIR:-}"
ARTIFACT_ROOT_OVERRIDE="${WORLDFOUNDRY_ARTIFACT_DIR:-}"
ENV_ROOT_OVERRIDE="${WORLDFOUNDRY_CONDA_ENVS_ROOT:-${WORLDFOUNDRY_CONDA_ENV_ROOT:-}}"
ENV_FILE="${WORLDFOUNDRY_ENV_FILE:-tmp/worldfoundry_unified_env.sh}"
CONDA_ENVIRONMENT_FILE="${WORLDFOUNDRY_CONDA_ENVIRONMENT_FILE:-environment.yml}"
WRITE_ENV_FILE=1
INSTALL_PRESET="${WORLDFOUNDRY_INSTALL_PRESET:-max-infer}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/unified_install.sh [options]

Install the default unified WorldFoundry GPU environment. This is the recommended
one-command setup for local inference on modern NVIDIA hosts.

Default stack:
  CUDA tier: auto (cu128 on CUDA 12.8+ drivers, cu124 on 12.4+, cu121 on 12.1+)
  install preset: max-infer
  env prefix: ${WORLDFOUNDRY_HOME:-$HOME/.cache/worldfoundry}/conda_envs/worldfoundry-unified-<tier>

Options:
  --cuda TIER           auto, cu128, cu124, or cu121. Default: auto.
  --environment-file PATH
                        Conda environment YAML. Default: environment.yml.
  --env-name NAME       Override unified env name.
  --prefix PATH         Override env prefix.
  --home PATH           Runtime state root. Default: ${XDG_CACHE_HOME:-$HOME/.cache}/worldfoundry.
  --env-root PATH       Conda envs root. Default: ${WORLDFOUNDRY_HOME}/conda_envs.
  --data-root PATH      Benchmark/data root. Default: ${WORLDFOUNDRY_HOME}/data.
  --model-root PATH     Model/checkpoint root. Default: ${WORLDFOUNDRY_HOME}/models.
  --artifact-root PATH  Generated artifact root. Default: ${WORLDFOUNDRY_HOME}/artifacts.
  --env-file PATH       Write sourceable exports. Default: tmp/worldfoundry_unified_env.sh.
  --no-env-file         Do not write the env export file.
  --preset NAME         max-infer or slim. Default: max-infer.
                        Both currently install requirements/worldfoundry-unified.txt.
  --pytorch-bundle NAME Accepted for older command lines; currently ignored.
  --transformers NAME   Accepted for older command lines; currently ignored.
  --skip-flash-attn     Skip flash-attn install.
  --torch SPEC          Torch package spec. Default: selected CUDA tier range.
  --torchvision SPEC    Torchvision package spec. Default: selected tier range.
  --torchaudio SPEC     Torchaudio package spec. Default: selected tier range.
  --allow-no-cuda       Do not fail verification when CUDA is not visible.
  --verify-only         Only verify imports/CUDA in the env.
  -h, --help            Show this help.

After install:
  source tmp/worldfoundry_unified_env.sh
  conda activate <printed env prefix>
EOF
}

INSTALL_ARGS=()

while (($#)); do
  case "$1" in
    --cuda)
      CUDA_TIER_REQUEST="$2"
      shift 2
      ;;
    --environment-file)
      CONDA_ENVIRONMENT_FILE="$2"
      shift 2
      ;;
    --env-name)
      ENV_NAME_OVERRIDE="$2"
      shift 2
      ;;
    --prefix)
      ENV_PREFIX_OVERRIDE="$2"
      shift 2
      ;;
    --home)
      HOME_ROOT="$2"
      shift 2
      ;;
    --env-root)
      ENV_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --model-root)
      MODEL_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      WRITE_ENV_FILE=1
      shift 2
      ;;
    --no-env-file)
      WRITE_ENV_FILE=0
      shift
      ;;
    --preset)
      INSTALL_PRESET="$2"
      shift 2
      ;;
    --pytorch-bundle)
      INSTALL_ARGS+=(--pytorch-bundle "$2")
      shift 2
      ;;
    --transformers)
      INSTALL_ARGS+=(--transformers "$2")
      shift 2
      ;;
    --skip-flash-attn)
      INSTALL_ARGS+=(--skip-flash-attn)
      shift
      ;;
    --torch)
      INSTALL_ARGS+=(--torch "$2")
      shift 2
      ;;
    --torchvision)
      INSTALL_ARGS+=(--torchvision "$2")
      shift 2
      ;;
    --torchaudio)
      INSTALL_ARGS+=(--torchaudio "$2")
      shift 2
      ;;
    --allow-no-cuda)
      INSTALL_ARGS+=(--allow-no-cuda)
      shift
      ;;
    --verify-only)
      INSTALL_ARGS+=(--verify-only)
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
done

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "python is required to resolve CUDA tier. Install Python or set PYTHON." >&2
    exit 1
  fi
fi

CUDA_REPORT="$(PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT" "$PYTHON_BIN" -m worldfoundry.runtime.cuda_tiers --requested "$CUDA_TIER_REQUEST" --field json)"
CUDA_TIER="$(printf '%s' "$CUDA_REPORT" | "$PYTHON_BIN" -c 'import json, sys; print(json.load(sys.stdin)["tier"])')"
DETECTED_DRIVER_CUDA="$(printf '%s' "$CUDA_REPORT" | "$PYTHON_BIN" -c 'import json, sys; print(json.load(sys.stdin).get("driver_cuda") or "")')"

ENV_NAME="${ENV_NAME_OVERRIDE:-worldfoundry-unified-${CUDA_TIER}}"
ENV_ROOT="${ENV_ROOT_OVERRIDE:-${HOME_ROOT}/conda_envs}"
ENV_PREFIX="${ENV_PREFIX_OVERRIDE:-${ENV_ROOT}/${ENV_NAME}}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-${HOME_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT_OVERRIDE:-${HOME_ROOT}/models}"
ARTIFACT_ROOT="${ARTIFACT_ROOT_OVERRIDE:-${HOME_ROOT}/artifacts}"
CACHE_ROOT="${WORLDFOUNDRY_CACHE_DIR:-${HOME_ROOT}/cache}"
HF_HOME_ROOT="${HF_HOME:-${HOME_ROOT}/huggingface}"
HFD_DATASET_ROOT="${WORLDFOUNDRY_HFD_DATASET_ROOT:-${DATA_ROOT}/datasets}"
HFD_ROOT="${WORLDFOUNDRY_HFD_ROOT:-${MODEL_ROOT}/checkpoints/hfd}"

case "$CUDA_TIER" in
  cu121|cu124|cu128) ;;
  *)
    echo "Unsupported resolved CUDA tier: ${CUDA_TIER}." >&2
    exit 2
    ;;
esac

mkdir -p "$CACHE_ROOT" "$DATA_ROOT" "$HFD_DATASET_ROOT" "$MODEL_ROOT" "${MODEL_ROOT}/checkpoints" "$HFD_ROOT" "$ARTIFACT_ROOT" "$HF_HOME_ROOT"

if [[ "$WRITE_ENV_FILE" == "1" ]]; then
  mkdir -p "$(dirname "$ENV_FILE")"
  cat >"$ENV_FILE" <<EOF
# Source this file before running WorldFoundry commands.
export WORLDFOUNDRY_HOME="${HOME_ROOT}"
export WORLDFOUNDRY_CACHE_DIR="${CACHE_ROOT}"
export WORLDFOUNDRY_DATA_DIR="${DATA_ROOT}"
export WORLDFOUNDRY_BENCHMARK_DATA_ROOT="${DATA_ROOT}/datasets"
export WORLDFOUNDRY_HFD_DATASET_ROOT="${HFD_DATASET_ROOT}"
export WORLDFOUNDRY_MODEL_DIR="${MODEL_ROOT}"
export WORLDFOUNDRY_CKPT_DIR="${MODEL_ROOT}/checkpoints"
export WORLDFOUNDRY_HFD_ROOT="${HFD_ROOT}"
export WORLDFOUNDRY_ARTIFACT_DIR="${ARTIFACT_ROOT}"
export WORLDFOUNDRY_CONDA_ENVS_ROOT="${ENV_ROOT}"
export WORLDFOUNDRY_CONDA_ENV_ROOT="${ENV_ROOT}"
export WORLDFOUNDRY_UNIFIED_ENV_NAME="${ENV_NAME}"
export WORLDFOUNDRY_UNIFIED_ENV_PREFIX="${ENV_PREFIX}"
export WORLDFOUNDRY_CONDA_ENV_PREFIX="${ENV_PREFIX}"
export WORLDFOUNDRY_CUDA_PROFILE="${CUDA_TIER}"
export WORLDFOUNDRY_CUDA_TIER="${CUDA_TIER}"
export WORLDFOUNDRY_USE_UNIFIED_ENV=1
export WORLDFOUNDRY_REPO_ROOT="${ROOT}"
export WORLDFOUNDRY_BENCH_ROOT="${ROOT}"
export WORLDFOUNDRY_WORKSPACE_ROOT="${HOME_ROOT}"
export CONDA_PREFIX="${ENV_PREFIX}"
export PATH="${ENV_PREFIX}/bin:\${PATH}"
export PYTHON="${ENV_PREFIX}/bin/python"
export HF_HOME="${HF_HOME_ROOT}"
export HF_HUB_CACHE="${HF_HOME_ROOT}/hub"
export HF_DATASETS_CACHE="${HF_HOME_ROOT}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME_ROOT}/transformers"
EOF
fi

export WORLDFOUNDRY_HOME="$HOME_ROOT"
export WORLDFOUNDRY_CACHE_DIR="$CACHE_ROOT"
export WORLDFOUNDRY_DATA_DIR="$DATA_ROOT"
export WORLDFOUNDRY_HFD_DATASET_ROOT="$HFD_DATASET_ROOT"
export WORLDFOUNDRY_MODEL_DIR="$MODEL_ROOT"
export WORLDFOUNDRY_ARTIFACT_DIR="$ARTIFACT_ROOT"
export WORLDFOUNDRY_HFD_ROOT="$HFD_ROOT"
export WORLDFOUNDRY_CONDA_ENVS_ROOT="$ENV_ROOT"
export WORLDFOUNDRY_CONDA_ENV_ROOT="$ENV_ROOT"
export WORLDFOUNDRY_UNIFIED_ENV_NAME="$ENV_NAME"
export WORLDFOUNDRY_UNIFIED_ENV_PREFIX="$ENV_PREFIX"
export WORLDFOUNDRY_CONDA_ENV_PREFIX="$ENV_PREFIX"
export WORLDFOUNDRY_CUDA_PROFILE="$CUDA_TIER"
export WORLDFOUNDRY_CUDA_TIER="$CUDA_TIER"
export WORLDFOUNDRY_DETECTED_DRIVER_CUDA="$DETECTED_DRIVER_CUDA"
export WORLDFOUNDRY_USE_UNIFIED_ENV="${WORLDFOUNDRY_USE_UNIFIED_ENV:-1}"
export WORLDFOUNDRY_REPO_ROOT="$ROOT"
export WORLDFOUNDRY_BENCH_ROOT="$ROOT"
export WORLDFOUNDRY_WORKSPACE_ROOT="$HOME_ROOT"

echo "WorldFoundry unified CUDA report: ${CUDA_REPORT}"
echo "WorldFoundry unified env prefix: ${ENV_PREFIX}"
if [[ "$WRITE_ENV_FILE" == "1" ]]; then
  echo "WorldFoundry env exports: ${ENV_FILE}"
fi

bash "$ROOT/scripts/setup/conda_install.sh" \
  --prefix "$ENV_PREFIX" \
  --cuda "$CUDA_TIER" \
  --environment-file "$CONDA_ENVIRONMENT_FILE" \
  --preset "$INSTALL_PRESET" \
  "${INSTALL_ARGS[@]}"

install_worldfoundry_wrapper() {
  local name="$1"
  local module="$2"
  local wrapper="${ENV_PREFIX}/bin/${name}"
  mkdir -p "${ENV_PREFIX}/bin"
  cat >"$wrapper" <<EOF
#!/usr/bin/env bash
export WORLDFOUNDRY_REPO_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}\${PYTHONPATH:+:\${PYTHONPATH}}"
exec "${ENV_PREFIX}/bin/python" -m ${module} "\$@"
EOF
  chmod +x "$wrapper"
}

install_worldfoundry_wrapper worldfoundry worldfoundry.cli
install_worldfoundry_wrapper worldfoundry-eval worldfoundry.cli
install_worldfoundry_wrapper worldfoundry-mcp worldfoundry.mcp
install_worldfoundry_wrapper worldfoundry-studio worldfoundry.studio.cli
install_worldfoundry_wrapper worldfoundry-tui worldfoundry.cli.tui

echo
echo "WorldFoundry unified environment is ready."
if [[ "$WRITE_ENV_FILE" == "1" ]]; then
  echo "Run: source ${ENV_FILE}"
fi
echo "Run: conda activate ${ENV_PREFIX}"
