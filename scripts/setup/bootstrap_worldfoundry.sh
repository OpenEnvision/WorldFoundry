#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORLDFOUNDRY_SOURCE_ROOT="$ROOT"

CUDA_TIER="${WORLDFOUNDRY_CUDA_PROFILE:-${WORLDFOUNDRY_CUDA_TIER:-auto}}"
HOME_ROOT="${WORLDFOUNDRY_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/worldfoundry}"
ENV_ROOT="${WORLDFOUNDRY_CONDA_ENVS_ROOT:-${WORLDFOUNDRY_CONDA_ENV_ROOT:-}}"
DATA_ROOT="${WORLDFOUNDRY_DATA_DIR:-}"
MODEL_ROOT="${WORLDFOUNDRY_MODEL_DIR:-}"
ARTIFACT_ROOT="${WORLDFOUNDRY_ARTIFACT_DIR:-}"
ENV_FILE="${WORLDFOUNDRY_ENV_FILE:-tmp/worldfoundry_unified_env.sh}"
WORKSPACE_PORT="${WORLDFOUNDRY_WORKSPACE_PORT:-7870}"
WORKSPACE_HOST="${WORLDFOUNDRY_WORKSPACE_HOST:-127.0.0.1}"
MAX_JOBS="${WORLDFOUNDRY_WORKSPACE_MAX_JOBS:-8}"
INSTALL_UNIFIED=1
VERIFY_ONLY=0
VERIFY=1
SKIP_FLASH_ATTN=0
ALLOW_NO_CUDA=0
DOWNLOAD_MODEL_ASSETS=0
START_WORKSPACE=0
MODELS=()
PREPARE_MODELS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/bootstrap_worldfoundry.sh [options]

Bootstrap a fresh WorldFoundry checkout. The default path creates the
single unified GPU conda environment, writes tmp/worldfoundry_unified_env.sh, and
verifies imports plus the model catalog. Model-specific environments and
checkpoint preparation are opt-in.

Common examples:
  bash scripts/setup/bootstrap_worldfoundry.sh
  bash scripts/setup/bootstrap_worldfoundry.sh --cuda cu128 --skip-flash-attn
  bash scripts/setup/bootstrap_worldfoundry.sh --with-model matrix-game-2 --prepare-model matrix-game-2
  bash scripts/setup/bootstrap_worldfoundry.sh --with-model lingbot-world --prepare-model lingbot-world
  bash scripts/setup/bootstrap_worldfoundry.sh --with-model evalcrafter
  bash scripts/setup/bootstrap_worldfoundry.sh --prepare-model skyreels-v3 --download-model-assets

Options:
  --cuda TIER              auto, cu128, cu124, or cu121. Default: auto.
  --home PATH              WorldFoundry runtime root. Default: ${XDG_CACHE_HOME:-$HOME/.cache}/worldfoundry.
  --env-root PATH          Conda envs root. Default: ${WORLDFOUNDRY_HOME}/conda_envs.
  --data-root PATH         Data root forwarded to unified_install.sh.
  --model-root PATH        Model/checkpoint root forwarded to unified_install.sh.
  --artifact-root PATH     Artifact root forwarded to unified_install.sh.
  --env-file PATH          Sourceable env exports. Default: tmp/worldfoundry_unified_env.sh.
  --with-model MODEL       Install or verify a model/benchmark resolved conda env. May be repeated.
  --prepare-model MODEL    Print checkpoint/HF preparation plan. May be repeated.
  --download-model-assets  Execute public model asset downloads for --prepare-model.
                            Gated models still require accepted terms and HF_TOKEN.
  --workspace-port PORT    Port printed or used for workspace start. Default: 7870.
  --workspace-host HOST    Host printed or used for workspace start. Default: 127.0.0.1.
  --max-jobs N             Workspace max jobs. Default: 8.
  --start-workspace        Start the workspace after setup. This blocks the shell.
  --skip-unified           Do not install the unified env; still source --env-file.
  --verify-only            Verify existing envs; do not install packages.
  --no-verify              Skip post-install import/catalog verification.
  --skip-flash-attn        Forward to unified and model env installers.
  --allow-no-cuda          Do not fail verification when CUDA is not visible.
  -h, --help               Show this help.

Policy:
  Use the unified env first. Add --with-model only for profiles that declare a
  real compatibility reason for isolation, including benchmark metric stacks
  such as EvalCrafter full official scoring. Use --prepare-model without
  --download-model-assets to review storage and authentication requirements.
  On CPU-only hosts, bootstrap skips flash-attn and allows CUDA verification to
  be absent while retaining cu128 as the explicit PyTorch wheel-index tier.
EOF
}

while (($#)); do
  case "$1" in
    --cuda)
      CUDA_TIER="$2"
      shift 2
      ;;
    --home)
      HOME_ROOT="$2"
      shift 2
      ;;
    --env-root)
      ENV_ROOT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --model-root)
      MODEL_ROOT="$2"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --with-model)
      MODELS+=("$2")
      shift 2
      ;;
    --prepare-model)
      PREPARE_MODELS+=("$2")
      shift 2
      ;;
    --download-model-assets)
      DOWNLOAD_MODEL_ASSETS=1
      shift
      ;;
    --workspace-port)
      WORKSPACE_PORT="$2"
      shift 2
      ;;
    --workspace-host)
      WORKSPACE_HOST="$2"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS="$2"
      shift 2
      ;;
    --start-workspace)
      START_WORKSPACE=1
      shift
      ;;
    --skip-unified)
      INSTALL_UNIFIED=0
      shift
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --no-verify)
      VERIFY=0
      shift
      ;;
    --skip-flash-attn)
      SKIP_FLASH_ATTN=1
      shift
      ;;
    --allow-no-cuda)
      ALLOW_NO_CUDA=1
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

# Avoid an auto-to-cu128 flash-attn source build when no NVIDIA driver command
# is available. The explicit tier selects the wheel index only; verification is
# allowed to run without a visible CUDA device.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [[ "$ALLOW_NO_CUDA" != "1" || "$SKIP_FLASH_ATTN" != "1" ]]; then
    echo "WARNING: nvidia-smi not found; enabling --allow-no-cuda --skip-flash-attn for CPU bootstrap." >&2
  fi
  ALLOW_NO_CUDA=1
  SKIP_FLASH_ATTN=1
  if [[ "$CUDA_TIER" == "auto" ]]; then
    echo "WARNING: no driver detected; using --cuda cu128 only as the torch wheel index (CPU verification allowed)." >&2
    CUDA_TIER=cu128
  fi
fi

source_env_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    return 0
  fi
  # shellcheck disable=SC1090
  source "$ENV_FILE"
}

unified_args=(bash "$ROOT/scripts/setup/unified_install.sh" --cuda "$CUDA_TIER" --home "$HOME_ROOT" --env-file "$ENV_FILE")
[[ -n "$ENV_ROOT" ]] && unified_args+=(--env-root "$ENV_ROOT")
[[ -n "$DATA_ROOT" ]] && unified_args+=(--data-root "$DATA_ROOT")
[[ -n "$MODEL_ROOT" ]] && unified_args+=(--model-root "$MODEL_ROOT")
[[ -n "$ARTIFACT_ROOT" ]] && unified_args+=(--artifact-root "$ARTIFACT_ROOT")
[[ "$VERIFY_ONLY" == "1" ]] && unified_args+=(--verify-only)
[[ "$SKIP_FLASH_ATTN" == "1" ]] && unified_args+=(--skip-flash-attn)
[[ "$ALLOW_NO_CUDA" == "1" ]] && unified_args+=(--allow-no-cuda)

if [[ "$INSTALL_UNIFIED" == "1" ]]; then
  "${unified_args[@]}"
fi

source_env_file

ENV_PREFIX="${WORLDFOUNDRY_UNIFIED_ENV_PREFIX:-${WORLDFOUNDRY_CONDA_ENV_PREFIX:-}}"
if [[ -z "$ENV_PREFIX" && "$INSTALL_UNIFIED" != "1" ]]; then
  echo "Missing WORLDFOUNDRY_UNIFIED_ENV_PREFIX. Source an env file or omit --skip-unified." >&2
  exit 1
fi

if [[ "$VERIFY" == "1" && -n "$ENV_PREFIX" ]]; then
  conda run -p "$ENV_PREFIX" python - <<'PY'
import json
import importlib
import torch

mods = ["worldfoundry", "torch", "diffusers", "transformers", "cv2", "PIL"]
result = {name: getattr(importlib.import_module(name), "__version__", "ok") for name in mods}
result["torch_cuda"] = torch.version.cuda
result["cuda_available"] = torch.cuda.is_available()
result["cuda_device_count"] = torch.cuda.device_count()
print(json.dumps(result, sort_keys=True))
PY
  conda run -p "$ENV_PREFIX" worldfoundry-eval zoo models --json >/dev/null
fi

for model_id in "${MODELS[@]}"; do
  args=(bash "$ROOT/scripts/setup/model_env_install.sh" --model "$model_id" --cuda "$CUDA_TIER" --home "$HOME_ROOT")
  [[ -n "$ENV_ROOT" ]] && args+=(--env-root "$ENV_ROOT")
  [[ "$VERIFY_ONLY" == "1" ]] && args+=(--verify-only)
  [[ "$SKIP_FLASH_ATTN" == "1" ]] && args+=(--skip-flash-attn)
  [[ "$ALLOW_NO_CUDA" == "1" ]] && args+=(--allow-no-cuda)
  "${args[@]}"
done

for model_id in "${PREPARE_MODELS[@]}"; do
  args=(bash "$ROOT/scripts/inference/prepare_model_infer.sh" "$model_id" --skip-env)
  if [[ "$DOWNLOAD_MODEL_ASSETS" == "1" ]]; then
    args+=(--download)
  fi
  [[ "$ALLOW_NO_CUDA" == "1" ]] && args+=(--allow-no-cuda)
  if [[ -n "$ENV_PREFIX" ]]; then
    env PYTHON="$ENV_PREFIX/bin/python" PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}" "${args[@]}"
  else
    "${args[@]}"
  fi
done

cat <<EOF

WorldFoundry setup summary
  env exports: ${ENV_FILE}
  unified env: ${ENV_PREFIX:-not resolved}
  checkpoint root: ${WORLDFOUNDRY_CKPT_DIR:-source ${ENV_FILE} to resolve}
  Hugging Face cache: ${HF_HUB_CACHE:-source ${ENV_FILE} to resolve}
  artifact root: ${WORLDFOUNDRY_ARTIFACT_DIR:-source ${ENV_FILE} to resolve}

Next shell commands:
  source ${ENV_FILE}
  conda activate "\${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}"
  bash scripts/setup/link_hf_checkpoints.sh --ckpt-dir "\${WORLDFOUNDRY_CKPT_DIR}" --hfd-root "\${WORLDFOUNDRY_HFD_ROOT}" --hf-hub-cache "\${HF_HUB_CACHE}" --default-world
  PYTHONPATH=${WORLDFOUNDRY_SOURCE_ROOT} WORLDFOUNDRY_WORKSPACE_MAX_JOBS=${MAX_JOBS} python -m worldfoundry.studio.serving.workspace --host ${WORKSPACE_HOST} --port ${WORKSPACE_PORT}

Open:
  http://${WORKSPACE_HOST}:${WORKSPACE_PORT}/
EOF

if [[ "$START_WORKSPACE" == "1" ]]; then
  if [[ -z "$ENV_PREFIX" ]]; then
    echo "--start-workspace requires a resolved unified env prefix." >&2
    exit 1
  fi
  export PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export WORLDFOUNDRY_WORKSPACE_MAX_JOBS="$MAX_JOBS"
  exec conda run -p "$ENV_PREFIX" python -m worldfoundry.studio.serving.workspace --host "$WORKSPACE_HOST" --port "$WORKSPACE_PORT"
fi
