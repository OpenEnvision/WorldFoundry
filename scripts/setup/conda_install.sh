#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORLDFOUNDRY_SOURCE_ROOT="$ROOT"

ENV_NAME="${WORLDFOUNDRY_CONDA_ENV_NAME:-worldfoundry}"
ENV_PREFIX="${WORLDFOUNDRY_CONDA_ENV_PREFIX:-}"
CUDA_PROFILE="${WORLDFOUNDRY_CUDA_PROFILE:-auto}"
CONDA_EXE_PATH="${CONDA_EXE:-conda}"
CONDA_ENVIRONMENT_FILE="${WORLDFOUNDRY_CONDA_ENVIRONMENT_FILE:-environment.yml}"
INSTALL_PRESET="${WORLDFOUNDRY_INSTALL_PRESET:-max-infer}"
INSTALL_FLASH_ATTN=1
FLASH_ATTN_BUCKET="${WORLDFOUNDRY_FLASH_ATTN_BUCKET:-flash_attn_fa25}"
TORCH_SPEC="${WORLDFOUNDRY_TORCH_SPEC:-torch>=2.7,<2.12.0}"
TORCHVISION_SPEC="${WORLDFOUNDRY_TORCHVISION_SPEC:-torchvision>=0.22,<0.27.0}"
TORCHAUDIO_SPEC="${WORLDFOUNDRY_TORCHAUDIO_SPEC:-torchaudio>=2.7,<2.12.0}"
VERIFY_ONLY=0
ALLOW_NO_CUDA="${WORLDFOUNDRY_ALLOW_NO_CUDA:-0}"
CUDA_NVCC_DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/conda_install.sh [options]

Creates or updates the open-source WorldFoundry GPU conda environment.

Options:
  --env-name NAME       Conda environment name. Default: worldfoundry.
  --prefix PATH         Conda environment prefix instead of --env-name.
  --cuda PROFILE        PyTorch CUDA wheel bucket: auto, cu128, cu124, or cu121.
                       Default: auto; modern CUDA 12.8 hosts resolve to cu128.
  --environment-file PATH
                       Conda environment YAML. Default: environment.yml.
  --preset NAME         max-infer or slim. Default: max-infer.
                       Both install requirements/worldfoundry-unified.txt; slim
                       is retained as a compatibility alias.
  --pytorch-bundle NAME Legacy compatibility option; ignored.
  --transformers NAME   Legacy compatibility option; ignored.
  --three-d-core        Legacy compatibility option; ignored.
  --skip-three-d-core   Legacy compatibility option; ignored.
  --flash-attn BUCKET   flash-attn bucket: flash_attn_fa25 or flash_attn_fa28.
  --skip-flash-attn     Do not install flash-attn.
  --torch SPEC          Torch package spec. Default: torch>=2.7,<2.12.0.
  --torchvision SPEC    Torchvision package spec. Default: torchvision>=0.22,<0.27.0.
  --torchaudio SPEC     Torchaudio package spec. Default: torchaudio>=2.7,<2.12.0.
  --verify-only         Only run import and CUDA verification in the env.
  --allow-no-cuda       Do not fail verification when CUDA is not visible.
  --cuda-nvcc-dry-run   Print the tier-aware nvcc install command and exit.
  -h, --help            Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --prefix)
      ENV_PREFIX="$2"
      shift 2
      ;;
    --cuda)
      CUDA_PROFILE="$2"
      shift 2
      ;;
    --environment-file)
      CONDA_ENVIRONMENT_FILE="$2"
      shift 2
      ;;
    --preset)
      INSTALL_PRESET="$2"
      shift 2
      ;;
    --pytorch-bundle)
      shift 2
      ;;
    --transformers)
      shift 2
      ;;
    --three-d-core)
      shift
      ;;
    --skip-three-d-core)
      shift
      ;;
    --flash-attn)
      FLASH_ATTN_BUCKET="$2"
      INSTALL_FLASH_ATTN=1
      shift 2
      ;;
    --skip-flash-attn)
      INSTALL_FLASH_ATTN=0
      shift
      ;;
    --torch)
      TORCH_SPEC="$2"
      shift 2
      ;;
    --torchvision)
      TORCHVISION_SPEC="$2"
      shift 2
      ;;
    --torchaudio)
      TORCHAUDIO_SPEC="$2"
      shift 2
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --allow-no-cuda)
      ALLOW_NO_CUDA=1
      shift
      ;;
    --cuda-nvcc-dry-run)
      CUDA_NVCC_DRY_RUN=1
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

case "$INSTALL_PRESET" in
  max-infer|slim) ;;
  *)
    echo "Unsupported --preset: ${INSTALL_PRESET}. Use max-infer or slim." >&2
    exit 2
    ;;
esac

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

REQUESTED_CUDA_PROFILE="$CUDA_PROFILE"
if ! CUDA_REPORT="$(PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT" "$PYTHON_BIN" -m worldfoundry.runtime.cuda_tiers --requested "$REQUESTED_CUDA_PROFILE" --field json)"; then
  echo "Failed to resolve --cuda ${REQUESTED_CUDA_PROFILE}." >&2
  exit 2
fi
CUDA_PROFILE="$(printf '%s' "$CUDA_REPORT" | "$PYTHON_BIN" -c 'import json, sys; print(json.load(sys.stdin)["tier"])')"
DETECTED_DRIVER_CUDA="$(printf '%s' "$CUDA_REPORT" | "$PYTHON_BIN" -c 'import json, sys; print(json.load(sys.stdin).get("driver_cuda") or "")')"

case "$CUDA_PROFILE" in
  cu121) CUDA_NVCC_VERSION="12.1" ;;
  cu124) CUDA_NVCC_VERSION="12.4" ;;
  cu128) CUDA_NVCC_VERSION="12.8" ;;
  *)
    echo "Unsupported resolved CUDA profile: ${CUDA_PROFILE}. Use auto, cu121, cu124, or cu128." >&2
    exit 2
    ;;
esac

if [[ "$CUDA_NVCC_DRY_RUN" == "1" ]]; then
  if [[ -n "$ENV_PREFIX" ]]; then
    target=(--prefix "$ENV_PREFIX")
  else
    target=(--name "$ENV_NAME")
  fi
  printf '%q ' "$CONDA_EXE_PATH" install "${target[@]}" --yes --channel nvidia "cuda-nvcc=${CUDA_NVCC_VERSION}"
  printf '\n'
  exit 0
fi

case "$FLASH_ATTN_BUCKET" in
  flash_attn_fa25|flash_attn_fa28) ;;
  *)
    echo "Unsupported --flash-attn bucket: ${FLASH_ATTN_BUCKET}. Use flash_attn_fa25 or flash_attn_fa28." >&2
    exit 2
    ;;
esac

if ! command -v "$CONDA_EXE_PATH" >/dev/null 2>&1; then
  echo "conda executable not found. Install Miniconda/Anaconda or set CONDA_EXE." >&2
  exit 1
fi

echo "WorldFoundry install preset: ${INSTALL_PRESET}"
echo "WorldFoundry CUDA tier: ${CUDA_REPORT}"

export WORLDFOUNDRY_CONDA_ENV_NAME="$ENV_NAME"
export WORLDFOUNDRY_CONDA_ENV_PREFIX="$ENV_PREFIX"
export WORLDFOUNDRY_ALLOW_NO_CUDA="$ALLOW_NO_CUDA"
export WORLDFOUNDRY_DETECTED_DRIVER_CUDA="$DETECTED_DRIVER_CUDA"
export CONDA_EXE="$CONDA_EXE_PATH"

env_selector() {
  if [[ -n "$ENV_PREFIX" ]]; then
    printf '%s\n' "-p" "$ENV_PREFIX"
  else
    printf '%s\n' "-n" "$ENV_NAME"
  fi
}

conda_run() {
  local args=()
  local env_prefix=()
  mapfile -t args < <(env_selector)
  if [[ -n "${PIP_CONFIG_FILE:-}" ]]; then
    env_prefix=(env "PIP_CONFIG_FILE=${PIP_CONFIG_FILE}")
  fi
  "${env_prefix[@]}" "$CONDA_EXE_PATH" run "${args[@]}" "$@"
}

ensure_matching_cuda_nvcc() {
  local current_release=""
  local nvcc_output=""
  if nvcc_output="$(conda_run nvcc --version 2>/dev/null)"; then
    current_release="$(printf '%s\n' "$nvcc_output" | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
  fi
  if [[ "$current_release" == "$CUDA_NVCC_VERSION" ]]; then
    echo "Matching CUDA nvcc already installed: ${current_release}"
    return 0
  fi

  local selector=()
  mapfile -t selector < <(env_selector)
  echo "Installing CUDA nvcc ${CUDA_NVCC_VERSION} from the NVIDIA conda channel."
  run_cmd "$CONDA_EXE_PATH" install "${selector[@]}" --yes --channel nvidia "cuda-nvcc=${CUDA_NVCC_VERSION}"
}

run_cmd() {
  "$@"
}

env_exists() {
  if [[ -n "$ENV_PREFIX" ]]; then
    [[ -x "$ENV_PREFIX/bin/python" ]]
  else
    "$CONDA_EXE_PATH" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
  fi
}

if [[ "$VERIFY_ONLY" != "1" ]]; then
  selector=()
  mapfile -t selector < <(env_selector)
  if env_exists; then
    run_cmd "$CONDA_EXE_PATH" env update "${selector[@]}" -f "$CONDA_ENVIRONMENT_FILE" --prune
  else
    run_cmd "$CONDA_EXE_PATH" env create "${selector[@]}" -f "$CONDA_ENVIRONMENT_FILE"
  fi

  ensure_matching_cuda_nvcc

  PIP_CONFIG_FILE="${WORLDFOUNDRY_PIP_CONFIG_FILE:-/dev/null}" conda_run \
    python -m pip install --no-cache-dir --upgrade pip

  TORCH_INDEX_URL="${WORLDFOUNDRY_TORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_PROFILE}}"
  PYPI_INDEX_URL="${WORLDFOUNDRY_PYPI_INDEX_URL:-https://pypi.org/simple}"
  PIP_CONFIG_FILE="${WORLDFOUNDRY_PIP_CONFIG_FILE:-/dev/null}" conda_run \
    python -m pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" --extra-index-url "$PYPI_INDEX_URL" \
    "$TORCH_SPEC" "$TORCHVISION_SPEC" "$TORCHAUDIO_SPEC"

  PIP_CONFIG_FILE="${WORLDFOUNDRY_PIP_CONFIG_FILE:-/dev/null}" conda_run \
    python -m pip install --no-cache-dir --index-url "$PYPI_INDEX_URL" \
    -r requirements/worldfoundry-unified.txt

  if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
    run_cmd bash scripts/setup/install_flash_attn.sh "$FLASH_ATTN_BUCKET"
  fi
fi

conda_run python - <<'PY'
import importlib
import json
import os

mods = [
    "cv2",
    "h5py",
    "matplotlib",
    "numpy",
    "PIL",
    "pyarrow",
    "torch",
    "torchvision",
    "transformers",
    "diffusers",
    "worldfoundry",
    "worldfoundry.cli",
    "xml.parsers.expat",
]
out = {}
for name in mods:
    module = importlib.import_module(name)
    out[name] = getattr(module, "__version__", "ok")

import torch

out["torch_version"] = torch.__version__
out["torch_cuda"] = torch.version.cuda
out["cuda_available"] = torch.cuda.is_available()
out["cuda_device_count"] = torch.cuda.device_count()
allow_no_cuda = os.environ.get("WORLDFOUNDRY_ALLOW_NO_CUDA") == "1"
if not torch.cuda.is_available() and not allow_no_cuda:
    raise SystemExit("CUDA is not available in the WorldFoundry conda environment")

print(json.dumps(out, sort_keys=True))
PY
