#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORLDFOUNDRY_SOURCE_ROOT="$ROOT"

CONDA_EXE_PATH="${CONDA_EXE:-conda}"
CUDA_TIER_REQUEST="${WORLDFOUNDRY_CUDA_PROFILE:-${WORLDFOUNDRY_CUDA_TIER:-auto}}"
HOME_ROOT="${WORLDFOUNDRY_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/worldfoundry}"
ENV_ROOT="${WORLDFOUNDRY_CONDA_ENVS_ROOT:-${WORLDFOUNDRY_CONDA_ENV_ROOT:-}}"
VERIFY_ONLY=0
ALLOW_NO_CUDA="${WORLDFOUNDRY_ALLOW_NO_CUDA:-0}"
SKIP_FLASH_ATTN=0
LIST_ONLY=0
MODELS=()

canonical_model_id() {
  case "$1" in
    lyra1)
      printf '%s\n' "lyra-1"
      ;;
    cosmos3-nano|cosmos3-super|cosmos-3|cosmos-3-nano|cosmos-3-super)
      printf '%s\n' "cosmos3"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/model_env_install.sh --model MODEL [--model MODEL ...] [options]

Install the conda environment required by one or more WorldFoundry model or
benchmark runtime profiles. The script uses the profile records in
worldfoundry/data/models/runtime/environments and routes
compatible profiles into the unified WorldFoundry env by default.

Common examples:
  bash scripts/setup/model_env_install.sh --model wan2.1-t2v-1.3b
  bash scripts/setup/model_env_install.sh --model ltx-2.3-i2v
  bash scripts/setup/model_env_install.sh --model hunyuanvideo-1.5-t2v
  bash scripts/setup/model_env_install.sh --model evalcrafter
  bash scripts/setup/model_env_install.sh --list

Options:
  --model MODEL         Model or benchmark runtime profile id. May be repeated.
  --list                Print the model-to-env mapping and exit. Includes
                        runtime-profile models that default to the unified env.
  --cuda TIER           auto, cu128, cu124, or cu121. Default: auto.
  --home PATH           Runtime state root. Default: ${XDG_CACHE_HOME:-$HOME/.cache}/worldfoundry.
  --env-root PATH       Conda envs directory. Default: the same WorldFoundry path resolver used at runtime.
  --verify-only         Verify imports in the resolved env; do not install.
  --skip-flash-attn     Forwarded when installing the unified env.
  --allow-no-cuda       Forwarded when installing/verifying the unified env.
  -h, --help            Show this help.

Policy:
  The unified env is the default for open-source inference. Dedicated envs are
  installed only when the runtime profile has a real compatibility reason such
  as JAX/TensorFlow isolation, exact ABI pins, or an older diffusers stack.
EOF
}

while (($#)); do
  case "$1" in
    --model)
      MODELS+=("$(canonical_model_id "$2")")
      shift 2
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    --cuda)
      CUDA_TIER_REQUEST="$2"
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
    --verify-only)
      VERIFY_ONLY=1
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

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "python is required to resolve model runtime profiles. Install Python or set PYTHON." >&2
    exit 1
  fi
fi

if ! command -v "$CONDA_EXE_PATH" >/dev/null 2>&1 && [[ "$LIST_ONLY" != "1" ]]; then
  echo "conda executable not found. Install Miniconda/Anaconda or set CONDA_EXE." >&2
  exit 1
fi

CUDA_REPORT="$(PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT" "$PYTHON_BIN" -m worldfoundry.runtime.cuda_tiers --requested "$CUDA_TIER_REQUEST" --field json)"
CUDA_TIER="$(printf '%s' "$CUDA_REPORT" | "$PYTHON_BIN" -c 'import json, sys; print(json.load(sys.stdin)["tier"])')"
DETECTED_DRIVER_CUDA="$(printf '%s' "$CUDA_REPORT" | "$PYTHON_BIN" -c 'import json, sys; print(json.load(sys.stdin).get("driver_cuda") or "")')"
if [[ -z "$ENV_ROOT" ]]; then
  ENV_ROOT="$(WORLDFOUNDRY_HOME="$HOME_ROOT" PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT" "$PYTHON_BIN" <<'PY'
from worldfoundry.core.io.paths import conda_envs_root_path

print(conda_envs_root_path())
PY
)"
fi

export WORLDFOUNDRY_HOME="$HOME_ROOT"
export WORLDFOUNDRY_CONDA_ENVS_ROOT="$ENV_ROOT"
export WORLDFOUNDRY_CONDA_ENV_ROOT="$ENV_ROOT"
export WORLDFOUNDRY_CUDA_PROFILE="$CUDA_TIER"
export WORLDFOUNDRY_CUDA_TIER="$CUDA_TIER"
export WORLDFOUNDRY_DETECTED_DRIVER_CUDA="$DETECTED_DRIVER_CUDA"
export WORLDFOUNDRY_USE_UNIFIED_ENV=1

json_field() {
  local json_text="$1"
  local field="$2"
  JSON_TEXT="$json_text" "$PYTHON_BIN" - "$field" <<'PY'
import json
import os
import sys

field = sys.argv[1]
payload = json.loads(os.environ["JSON_TEXT"])
value = payload.get(field)
if value is None:
    raise SystemExit(0)
if isinstance(value, (dict, list)):
    print(json.dumps(value, sort_keys=True))
else:
    print(value)
PY
}

json_array_lines() {
  local json_text="$1"
  local field="$2"
  JSON_TEXT="$json_text" "$PYTHON_BIN" - "$field" <<'PY'
import json
import os
import sys

field = sys.argv[1]
payload = json.loads(os.environ["JSON_TEXT"])
for value in payload.get(field) or []:
    print(value)
PY
}

spec_json_for_model() {
  local model_id="$1"
  PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT" "$PYTHON_BIN" - "$model_id" <<'PY'
import json
import sys

from worldfoundry.runtime.conda import load_runtime_conda_env_spec
from worldfoundry.runtime.cuda_tiers import resolve_install_tier, unified_env_name
from worldfoundry.core.io.paths import conda_envs_root_path

model_id = sys.argv[1]
spec = load_runtime_conda_env_spec(model_id)
if spec is None:
    tier = resolve_install_tier()
    root = conda_envs_root_path()
    print(json.dumps({
        "model_id": model_id,
        "env_name": unified_env_name(tier),
        "resolved_env_name": unified_env_name(tier),
        "env_prefix": str(root / unified_env_name(tier)),
        "python": "3.11",
        "cuda_profile": tier,
        "driver_status": "compatible_unified_default",
        "conda_packages": [],
        "pip_packages": [],
        "pip_extra_index_url": "",
        "pip_find_links": [],
        "validation_imports": ["torch", "diffusers", "transformers", "worldfoundry"],
        "source_requirement_files": [],
        "editable_install_dirs": [],
        "pythonpath_dirs": [],
        "notes": ["no per-model conda profile; using unified WorldFoundry env"],
        "exists": False,
    }, sort_keys=True))
else:
    print(json.dumps(spec.to_dict(check_exists=False), sort_keys=True))
PY
}

print_mapping() {
  PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT" "$PYTHON_BIN" <<'PY'
from worldfoundry.runtime.conda import load_runtime_conda_env_specs_with_overrides
from worldfoundry.runtime.cuda_tiers import resolve_install_tier, unified_env_name
from worldfoundry.core.io.paths import conda_envs_root_path
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profiles

specs = load_runtime_conda_env_specs_with_overrides()
profiles = load_runtime_profiles(check_conda_env_exists=False)
tier = resolve_install_tier()
unified = unified_env_name(tier)
root = conda_envs_root_path()
rows = {}
for model_id, spec in specs.items():
    rows[model_id] = (
        model_id,
        spec.resolved_env_name,
        spec.cuda_profile,
        spec.python,
        "explicit-conda-profile",
    )
for model_id, profile in profiles.items():
    profile_env = profile.conda_env or {}
    if profile_env:
        resolved_env = profile_env.get("resolved_env_name") or profile_env.get("env_name") or unified
        profile_tier = profile_env.get("cuda_profile") or tier
        profile_python = profile_env.get("python") or "3.11"
        source_model = profile_env.get("model_id") or model_id
        source = f"runtime-profile-environment:{source_model}"
    else:
        resolved_env = unified
        profile_tier = tier
        profile_python = "3.11"
        source = f"runtime-profile-default:{root / unified}"
    rows.setdefault(
        model_id,
        (
            model_id,
            resolved_env,
            profile_tier,
            profile_python,
            source,
        ),
    )
for row in sorted(rows.values(), key=lambda item: item[0]):
    print("\t".join(str(item) for item in row))
PY
}

install_unified_env() {
  local env_prefix="$1"
  local cuda_profile="$2"
  local args=(bash "$ROOT/scripts/setup/unified_install.sh" --cuda "$cuda_profile" --prefix "$env_prefix" --home "$HOME_ROOT")
  if [[ "$VERIFY_ONLY" == "1" ]]; then
    args+=(--verify-only)
  fi
  if [[ "$SKIP_FLASH_ATTN" == "1" ]]; then
    args+=(--skip-flash-attn)
  fi
  if [[ "$ALLOW_NO_CUDA" == "1" ]]; then
    args+=(--allow-no-cuda)
  fi
  "${args[@]}"
}

verify_env_imports() {
  local env_prefix="$1"
  shift
  local imports=("$@")
  if [[ "${#imports[@]}" == "0" ]]; then
    imports=(worldfoundry)
  fi
  local python_bin="$env_prefix/bin/python"
  if [[ ! -x "$python_bin" ]]; then
    echo "Missing env Python: $python_bin" >&2
    exit 1
  fi
  local import_csv
  import_csv="$(IFS=,; printf '%s' "${imports[*]}")"
  CUDA_HOME="$env_prefix" \
  LD_LIBRARY_PATH="$(runtime_ld_library_path "$env_prefix")" \
  PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" - "$import_csv" <<'PY'
import importlib
import json
import sys

imports = [item for item in sys.argv[1].split(",") if item]
result = {}
for name in imports:
    try:
        importlib.import_module(name)
        result[name] = "ok"
    except Exception as exc:
        result[name] = f"FAIL: {type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True))
failed = [name for name, value in result.items() if value != "ok"]
raise SystemExit(1 if failed else 0)
PY
}

verify_env_runtime_tools() {
  local model_id="$1"
  local env_prefix="$2"
  if [[ "$model_id" != "cosmos3" ]]; then
    return 0
  fi
  CUDA_HOME="$env_prefix" \
  LD_LIBRARY_PATH="$(runtime_ld_library_path "$env_prefix")" \
  PATH="$env_prefix/bin:$PATH" \
  "$env_prefix/bin/python" <<'PY'
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

candidates = [Path(__import__("sys").prefix) / "bin" / "ffmpeg", Path(imageio_ffmpeg.get_ffmpeg_exe())]
results = {}
for candidate in dict.fromkeys(candidates):
    try:
        completed = subprocess.run(
            [str(candidate), "-version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        results[str(candidate)] = f"FAIL: {type(exc).__name__}: {exc}"
    else:
        results[str(candidate)] = "ok" if completed.returncode == 0 else f"FAIL: {completed.stderr.strip()}"
print(json.dumps({"cosmos3_ffmpeg_candidates": results}, sort_keys=True))
if not any(value == "ok" for value in results.values()):
    raise SystemExit("No usable ffmpeg executable is available in the Cosmos3 environment.")
PY
}

runtime_ld_library_path() {
  local env_prefix="$1"
  local dirs=()
  [[ -d "$env_prefix/lib" ]] && dirs+=("$env_prefix/lib")
  local path
  for path in "$env_prefix"/lib/python*/site-packages/torch/lib; do
    [[ -d "$path" ]] && dirs+=("$path")
  done
  for path in "$env_prefix"/lib/python*/site-packages/nvidia/*/lib; do
    [[ -d "$path" ]] && dirs+=("$path")
  done
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    dirs+=("$LD_LIBRARY_PATH")
  fi
  local IFS=:
  printf '%s' "${dirs[*]}"
}

filter_bootstrap_conda_packages() {
  local package
  for package in "$@"; do
    case "$package" in
      python|python=*|python==*|pip|pip=*|pip==*)
        continue
        ;;
      *)
        printf '%s\n' "$package"
        ;;
    esac
  done
}

bootstrap_pip_package() {
  local package resolved="pip"
  for package in "$@"; do
    case "$package" in
      pip==*)
        resolved="pip=${package#pip==}"
        ;;
      pip=*)
        resolved="$package"
        ;;
    esac
  done
  printf '%s\n' "$resolved"
}

append_pip_index_options() {
  local -n target_args="$1"
  target_args+=(--index-url "${WORLDFOUNDRY_PIP_INDEX_URL:-https://pypi.org/simple}")
  if [[ -n "${WORLDFOUNDRY_PIP_TRUSTED_HOST:-}" ]]; then
    target_args+=(--trusted-host "$WORLDFOUNDRY_PIP_TRUSTED_HOST")
  fi
}

append_pip_find_links_options() {
  local -n target_args="$1"
  shift
  local link
  for link in "$@"; do
    [[ -n "$link" ]] || continue
    target_args+=(--find-links "$link")
  done
}

patch_transformer_engine_links() {
  local env_prefix="$1"
  local include_dir py_include_dir
  include_dir="$env_prefix/include"
  py_include_dir="$env_prefix/include/python3.10"
  mkdir -p "$include_dir" "$py_include_dir"
  local include_root header versioned_lib
  for include_root in "$env_prefix"/lib/python*/site-packages/nvidia/*/include; do
    [[ -d "$include_root" ]] || continue
    for header in "$include_root"/*; do
      [[ -e "$header" ]] || continue
      ln -sf "$header" "$include_dir/"
      ln -sf "$header" "$py_include_dir/"
    done
  done

  for versioned_lib in "$env_prefix"/lib/python*/site-packages/nvidia/cudnn/lib/libcudnn.so.*; do
    [[ -f "$versioned_lib" ]] || continue
    local lib_dir
    lib_dir="$(dirname "$versioned_lib")"
    [[ -e "$lib_dir/libcudnn.so" ]] || ln -sf "$(basename "$versioned_lib")" "$lib_dir/libcudnn.so"
  done
}

transformer_engine_import_ok() {
  local env_prefix="$1"
  CUDA_HOME="$env_prefix" \
  LD_LIBRARY_PATH="$(runtime_ld_library_path "$env_prefix")" \
  "$env_prefix/bin/python" - <<'PY' >/dev/null 2>&1
import transformer_engine
PY
}

install_transformer_engine_official() {
  local env_prefix="$1"
  if transformer_engine_import_ok "$env_prefix"; then
    echo "==> transformer-engine already importable in ${env_prefix}"
    return
  fi
  local pip_args=("$CONDA_EXE_PATH" run -p "$env_prefix" python -m pip install)
  append_pip_index_options pip_args
  pip_args+=(--no-cache-dir "transformer-engine[pytorch]==1.12.0")
  CUDA_HOME="$env_prefix" \
  LD_LIBRARY_PATH="$(runtime_ld_library_path "$env_prefix")" \
  "${pip_args[@]}"
}

four_d_worldbench_aux_imports_ok() {
  local env_prefix="$1"
  "$env_prefix/bin/python" - <<'PY' >/dev/null 2>&1
import keye_vl_utils
import openai
import qwen_vl_utils
import skvideo
PY
}

install_four_d_worldbench_aux_stack() {
  local env_prefix="$1"
  if four_d_worldbench_aux_imports_ok "$env_prefix"; then
    echo "==> 4dworldbench: auxiliary Python packages already importable in ${env_prefix}"
    return
  fi

  local pip_args=("$CONDA_EXE_PATH" run -p "$env_prefix" python -m pip install --no-cache-dir)
  append_pip_index_options pip_args
  pip_args+=(keye-vl-utils qwen-vl-utils openai scikit-video)
  "${pip_args[@]}"
}

install_gamecraft_flash_attn() {
  local env_prefix="$1"
  if CUDA_HOME="$env_prefix" \
    LD_LIBRARY_PATH="$(runtime_ld_library_path "$env_prefix")" \
    "$env_prefix/bin/python" - <<'PY' >/dev/null 2>&1
import flash_attn
PY
  then
    echo "==> Hunyuan GameCraft: flash-attn already importable"
    return
  fi
  local pip_args=("$CONDA_EXE_PATH" run -p "$env_prefix" python -m pip install)
  append_pip_index_options pip_args
  pip_args+=(--no-cache-dir --no-build-isolation "flash-attn==2.6.3")
  CUDA_HOME="$env_prefix" \
  LD_LIBRARY_PATH="$(runtime_ld_library_path "$env_prefix")" \
  "${pip_args[@]}"
}

post_install_model_env() {
  local model_id="$1"
  local env_prefix="$2"
  case "$model_id" in
    4dworldbench)
      echo "==> 4dworldbench: installing auxiliary metric runtime packages"
      install_four_d_worldbench_aux_stack "$env_prefix"
      ;;
    gen3c|lyra-1)
      echo "==> ${model_id}: applying the Cosmos Predict1 Transformer Engine setup"
      patch_transformer_engine_links "$env_prefix"
      install_transformer_engine_official "$env_prefix"
      ;;
    hunyuan-game-craft)
      echo "==> Hunyuan GameCraft: building the pinned released flash-attn package"
      install_gamecraft_flash_attn "$env_prefix"
      ;;
  esac
}

install_dedicated_env() {
  local spec_json="$1"
  local model_id env_name env_prefix python_version pip_extra_index
  model_id="$(json_field "$spec_json" model_id)"
  env_name="$(json_field "$spec_json" resolved_env_name)"
  env_prefix="$(json_field "$spec_json" env_prefix)"
  python_version="$(json_field "$spec_json" python)"
  pip_extra_index="$(json_field "$spec_json" pip_extra_index_url)"
  mapfile -t conda_packages < <(json_array_lines "$spec_json" conda_packages)
  mapfile -t pip_packages < <(json_array_lines "$spec_json" pip_packages)
  mapfile -t requirement_files < <(json_array_lines "$spec_json" source_requirement_files)
  mapfile -t editable_dirs < <(json_array_lines "$spec_json" editable_install_dirs)
  mapfile -t pythonpath_dirs < <(json_array_lines "$spec_json" pythonpath_dirs)
  mapfile -t validation_imports < <(json_array_lines "$spec_json" validation_imports)
  mapfile -t pip_find_links < <(json_array_lines "$spec_json" pip_find_links)
  mapfile -t channels < <(json_array_lines "$spec_json" channels)
  mapfile -t resolved_conda_packages < <(filter_bootstrap_conda_packages "${conda_packages[@]}")
  local pip_package
  pip_package="$(bootstrap_pip_package "${conda_packages[@]}")"
  local conda_channel_args=()
  for channel in "${channels[@]}"; do
    conda_channel_args+=(-c "$channel")
  done

  if [[ "$VERIFY_ONLY" == "1" ]]; then
    echo "==> ${model_id}: verifying ${env_name} at ${env_prefix}"
  else
    echo "==> ${model_id}: installing ${env_name} at ${env_prefix}"
  fi
  local installing_marker="${env_prefix}.worldfoundry-installing"
  local ready_marker="${env_prefix}/.worldfoundry-ready"
  if [[ "$VERIFY_ONLY" != "1" ]]; then
    mkdir -p "$(dirname "$env_prefix")"
    touch "$installing_marker"
    rm -f "$ready_marker"
    if [[ ! -x "$env_prefix/bin/python" ]]; then
      local create_args=("$CONDA_EXE_PATH" create -y "${conda_channel_args[@]}" -p "$env_prefix" "python=${python_version}" "$pip_package")
      if [[ "${#resolved_conda_packages[@]}" -gt 0 ]]; then
        create_args+=("${resolved_conda_packages[@]}")
      fi
      "${create_args[@]}"
    elif [[ -d "$env_prefix/conda-meta" ]]; then
      local install_args=("$CONDA_EXE_PATH" install -y "${conda_channel_args[@]}" -p "$env_prefix" "python=${python_version}" "$pip_package")
      if [[ "${#resolved_conda_packages[@]}" -gt 0 ]]; then
        install_args+=("${resolved_conda_packages[@]}")
      fi
      "${install_args[@]}"
    elif [[ ! -d "$env_prefix/conda-meta" ]]; then
      echo "Existing runtime at ${env_prefix} is not a conda environment. Remove it or choose a different --env-root." >&2
      exit 1
    fi
    if [[ "$pip_package" == "pip" ]]; then
      local pip_upgrade_args=("$CONDA_EXE_PATH" run -p "$env_prefix" python -m pip install)
      append_pip_index_options pip_upgrade_args
      pip_upgrade_args+=(--upgrade pip)
      "${pip_upgrade_args[@]}"
    fi
    if [[ "${#pip_packages[@]}" -gt 0 ]]; then
      local pip_args=("$CONDA_EXE_PATH" run -p "$env_prefix" python -m pip install --no-cache-dir)
      append_pip_index_options pip_args
      append_pip_find_links_options pip_args "${pip_find_links[@]}"
      if [[ -n "$pip_extra_index" ]]; then
        pip_args+=(--extra-index-url "$pip_extra_index")
      fi
      pip_args+=("${pip_packages[@]}")
      "${pip_args[@]}"
    fi
    for req in "${requirement_files[@]}"; do
      local req_pip_args=("$CONDA_EXE_PATH" run -p "$env_prefix" python -m pip install --no-cache-dir)
      append_pip_index_options req_pip_args
      append_pip_find_links_options req_pip_args "${pip_find_links[@]}"
      req_pip_args+=(-r "$req")
      "${req_pip_args[@]}"
    done
    for edit_dir in "${editable_dirs[@]}"; do
      local edit_pip_args=("$CONDA_EXE_PATH" run -p "$env_prefix" python -m pip install --no-cache-dir)
      append_pip_index_options edit_pip_args
      append_pip_find_links_options edit_pip_args "${pip_find_links[@]}"
      edit_pip_args+=(-e "$edit_dir")
      "${edit_pip_args[@]}"
    done
    if [[ "${#pythonpath_dirs[@]}" -gt 0 || -d "$WORLDFOUNDRY_SOURCE_ROOT" ]]; then
      local site_packages
      site_packages="$(find "$env_prefix/lib" -name site-packages -type d | head -1)"
      if [[ -n "$site_packages" ]]; then
        {
          printf '%s\n' "$WORLDFOUNDRY_SOURCE_ROOT"
          for path in "${pythonpath_dirs[@]}"; do
            printf '%s\n' "$ROOT/$path"
          done
        } >"$site_packages/worldfoundry.pth"
      fi
    fi
    post_install_model_env "$model_id" "$env_prefix"
  fi
  verify_env_imports "$env_prefix" "${validation_imports[@]}"
  verify_env_runtime_tools "$model_id" "$env_prefix"
  if [[ "$VERIFY_ONLY" != "1" ]]; then
    rm -f "$installing_marker" "$env_prefix/.worldfoundry-installing"
    touch "$ready_marker"
  fi
}

install_model_env() {
  local model_id="$1"
  local spec_json env_name env_prefix cuda_profile
  spec_json="$(spec_json_for_model "$model_id")"
  env_name="$(json_field "$spec_json" resolved_env_name)"
  env_prefix="$(json_field "$spec_json" env_prefix)"
  cuda_profile="$(json_field "$spec_json" cuda_profile)"
  echo "==> ${model_id}: resolved env=${env_name} cuda=${cuda_profile} prefix=${env_prefix}"
  if [[ "$env_name" == worldfoundry-unified-* ]]; then
    install_unified_env "$env_prefix" "$cuda_profile"
    if [[ "$VERIFY_ONLY" != "1" ]]; then
      post_install_model_env "$model_id" "$env_prefix"
    fi
  else
    install_dedicated_env "$spec_json"
  fi
}

if [[ "$LIST_ONLY" == "1" ]]; then
  print_mapping
  exit 0
fi

if [[ "${#MODELS[@]}" == "0" ]]; then
  echo "At least one --model is required unless --list is used." >&2
  usage >&2
  exit 2
fi

mkdir -p "$ENV_ROOT"
for model_id in "${MODELS[@]}"; do
  install_model_env "$model_id"
done
