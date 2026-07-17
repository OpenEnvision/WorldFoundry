#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORLDFOUNDRY_SOURCE_ROOT="$ROOT"

MODEL_ID="${1:-}"
if [[ -z "$MODEL_ID" || "$MODEL_ID" == "-h" || "$MODEL_ID" == "--help" ]]; then
  cat >&2 <<'EOF'
Usage: bash scripts/inference/prepare_model_infer.sh <model-id> [options]

Options:
  --download             Download missing public model checkpoints.
  --skip-env             Do not install or verify the model conda profile.
  --verify-env-only      Verify the resolved conda env; do not install/update it.
  --home PATH            Runtime state root forwarded to model_env_install.sh.
  --env-root PATH        Conda envs directory forwarded to model_env_install.sh.
  --cache-dir PATH       Hugging Face hub cache root used by `hf download`.
  --output-dir PATH      Suggested inference output directory.
  --skip-flash-attn      Forwarded to the unified env installer.
  --allow-no-cuda        Forwarded to the unified env installer.

Example:
  bash scripts/inference/prepare_model_infer.sh matrix-game-2
  bash scripts/inference/prepare_model_infer.sh matrix-game-2 --download

RollingForcing downloads prefer ModelScope when its Python package is available.
Set WORLDFOUNDRY_HF_ENDPOINT (or HF_ENDPOINT) to override its Hugging Face
fallback endpoint; the fallback defaults to https://hf-mirror.com.
EOF
  if [[ -z "$MODEL_ID" ]]; then
    exit 2
  fi
  exit 0
fi
shift

DOWNLOAD=0
SKIP_ENV=0
VERIFY_ENV_ONLY=0
SKIP_FLASH_ATTN=0
ALLOW_NO_CUDA=0
HOME_ROOT="${WORLDFOUNDRY_HOME:-}"
ENV_ROOT="${WORLDFOUNDRY_CONDA_ENVS_ROOT:-${WORLDFOUNDRY_CONDA_ENV_ROOT:-}}"
HF_HOME_VALUE="${HF_HOME:-${HOME}/.cache/huggingface}"
CACHE_DIR="${HF_HUB_CACHE:-${HF_HOME_VALUE}/hub}"
OUTPUT_DIR="tmp/worldfoundry_infer/${MODEL_ID}"
OUTPUT_DIR_EXPLICIT=0
COSMOS3_NANO_REVISION="411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
COSMOS3_SUPER_REVISION="e0262be9d8f7586bc24c069a2aed2b665bdff266"
COSMOS3_REPO_ID=""
COSMOS3_EXPECTED_REVISION=""
COSMOS3_PINNED_MODEL_REF=""
COSMOS3_CHECKPOINT_REPORT=""

while (($#)); do
  case "$1" in
    --download)
      DOWNLOAD=1
      shift
      ;;
    --skip-env)
      SKIP_ENV=1
      shift
      ;;
    --verify-env-only)
      VERIFY_ENV_ONLY=1
      shift
      ;;
    --home)
      HOME_ROOT="$2"
      shift 2
      ;;
    --env-root)
      ENV_ROOT="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      OUTPUT_DIR_EXPLICIT=1
      shift 2
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
      exec bash "$0"
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    esac
done

REQUESTED_MODEL_ID="$MODEL_ID"
case "$MODEL_ID" in
  cogvideox_2b_t2v)
    MODEL_ID="cogvideox-2b-t2v"
    ;;
  cogvideox_5b_t2v)
    MODEL_ID="cogvideox-5b-t2v"
    ;;
  cogvideox_5b_i2v)
    MODEL_ID="cogvideox-5b-i2v"
    ;;
  dynamicrafter|dynamicrafter_512_i2v)
    MODEL_ID="dynamicrafter-512-i2v"
    ;;
  dynamicrafter_1024_i2v)
    MODEL_ID="dynamicrafter-1024-i2v"
    ;;
  easyanimate|easyanimate_i2v)
    MODEL_ID="easyanimate-i2v"
    ;;
  motion-ctrl|motion_ctrl)
    MODEL_ID="motionctrl"
    ;;
  hy-worldplay|hunyuanworldplay|hunyuan-world-play|worldplay)
    MODEL_ID="hunyuan-worldplay"
    ;;
  longcat|longcat_video)
    MODEL_ID="longcat-video"
    ;;
  helios-distilled|helios_distilled|helios-base|helios_base|helios-mid|helios_mid)
    MODEL_ID="helios"
    ;;
  gammaworld|gamma_world|gamma-world-causal|gamma-world-causal-few-step)
    MODEL_ID="gamma-world"
    ;;
  lingbot_video|lingbotvideo|lingbot-video-dense|lingbot-video-moe)
    MODEL_ID="lingbot-video"
    ;;
  lingbot-world-v2|lingbot_world_v2|lingbot-v2|lingbot-world-infinity)
    MODEL_ID="lingbot-world-v2"
    ;;
  self_forcing|selfforcing)
    MODEL_ID="self-forcing"
    ;;
  causal_forcing|causalforcing)
    MODEL_ID="causal-forcing"
    ;;
  rolling_forcing|rollingforcing|rolling)
    MODEL_ID="rolling-forcing"
    ;;
  allegro|allegro_ti2v)
    MODEL_ID="allegro-ti2v"
    ;;
  animate-diff|animate_diff)
    MODEL_ID="animatediff"
    ;;
  modelscope|modelscope_t2v)
    MODEL_ID="modelscope-t2v"
    ;;
  wan-2p2|wan2.2|wan22|wan2.2-ti2v|wan2.2-ti2v-5b)
    MODEL_ID="wan2.2-ti2v-5b"
    ;;
  zero-scope|zero_scope)
    MODEL_ID="zeroscope"
    ;;
  matrixgame|matrixgame1|matrix-game1|matrix_game_1)
    MODEL_ID="matrix-game-1"
    ;;
  svc|seva)
    MODEL_ID="stable-virtual-camera"
    ;;
  pusa|pusa-v1|pusa_v1|pusa-wan2.2-v1|pusa-wan22-v1)
    MODEL_ID="pusa-vidgen"
    ;;
  hunyuanvideo|hunyuan-video|hunyuanvideo_t2v|hunyuan-video-t2v)
    MODEL_ID="hunyuanvideo-t2v"
    ;;
  hunyuanvideo_i2v|hunyuan-video-i2v)
    MODEL_ID="hunyuanvideo-i2v"
    ;;
  lyra1)
    MODEL_ID="lyra-1"
    ;;
  cosmos-3|cosmos-3-nano|cosmos3_nano)
    MODEL_ID="cosmos3-nano"
    ;;
  cosmos-3-super|cosmos3_super)
    MODEL_ID="cosmos3-super"
    ;;
esac
if [[ "$MODEL_ID" != "$REQUESTED_MODEL_ID" ]]; then
  echo "Normalized model id: ${REQUESTED_MODEL_ID} -> ${MODEL_ID}"
fi
if [[ "$OUTPUT_DIR_EXPLICIT" != "1" ]]; then
  OUTPUT_DIR="tmp/worldfoundry_infer/${MODEL_ID}"
fi
INFER_MODEL_ID="$MODEL_ID"
ZOO_MODEL_ID="$MODEL_ID"
case "$MODEL_ID" in
  allegro-ti2v)
    INFER_MODEL_ID="allegro_ti2v"
    ZOO_MODEL_ID="allegro"
    ;;
  wan2.2-ti2v-5b|wan2.2-ti2v-5b-1280x704-121f)
    INFER_MODEL_ID="wan-2p2"
    ZOO_MODEL_ID="wan2.2"
    ;;
  dynamicrafter-512-i2v|dynamicrafter-1024-i2v)
    ZOO_MODEL_ID="dynamicrafter"
    ;;
  easyanimate-i2v)
    INFER_MODEL_ID="easyanimate_i2v"
    ZOO_MODEL_ID="easyanimate"
    ;;
    hunyuan-worldplay)
      ZOO_MODEL_ID="hy-worldplay"
      ;;
    longcat-video)
      ZOO_MODEL_ID="longcat-video"
      ;;
    lingbot-video)
      ZOO_MODEL_ID="lingbot-video"
      ;;
    self-forcing|causal-forcing|rolling-forcing)
      ZOO_MODEL_ID="$MODEL_ID"
      ;;
    cosmos3|cosmos3-nano|cosmos3-super)
      INFER_MODEL_ID="cosmos3"
      ZOO_MODEL_ID="cosmos3"
      ;;
esac

PYTHON_BIN="${PYTHON:-python}"
PYTHON_DIR="$(dirname "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)' 2>/dev/null || command -v "$PYTHON_BIN" || printf '%s' "$PYTHON_BIN")")"
if [[ -d "$PYTHON_DIR" ]]; then
  export PATH="$PYTHON_DIR:$PATH"
fi
export PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WORLDFOUNDRY_HOME="${WORLDFOUNDRY_HOME:-${HOME}/.cache/worldfoundry}"
export WORLDFOUNDRY_CKPT_DIR="${WORLDFOUNDRY_CKPT_DIR:-${WORLDFOUNDRY_HOME}/checkpoints}"
export WORLDFOUNDRY_HFD_ROOT="${WORLDFOUNDRY_HFD_ROOT:-${WORLDFOUNDRY_CKPT_DIR}/hfd}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_DIR}"

link_if_present() {
  local source_path="$1"
  local target_path="$2"
  if [[ -e "$target_path" || -L "$target_path" ]]; then
    return 0
  fi
  if [[ -e "$source_path" ]]; then
    mkdir -p "$(dirname "$target_path")"
    ln -s "$source_path" "$target_path"
  fi
}

download_modelscope_snapshot() {
  local repo_id="$1"
  local target_root="$2"
  shift 2

  if ! "$PYTHON_BIN" -c 'import modelscope' >/dev/null 2>&1; then
    return 1
  fi

  MODELSCOPE_DOWNLOAD_PARALLELS="${MODELSCOPE_DOWNLOAD_PARALLELS:-8}" \
    "$PYTHON_BIN" - "$repo_id" "$target_root" "$@" <<'PY'
import sys

from modelscope.hub.snapshot_download import snapshot_download

repo_id, local_dir, *allow_patterns = sys.argv[1:]
snapshot_download(
    repo_id,
    revision="master",
    local_dir=local_dir,
    allow_patterns=allow_patterns or None,
    max_workers=4,
)
PY
}

download_hf_with_mirror() {
  local endpoint="${WORLDFOUNDRY_HF_ENDPOINT:-${HF_ENDPOINT:-https://hf-mirror.com}}"
  HF_ENDPOINT="$endpoint" HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" hf download "$@"
}

verify_rolling_forcing_checkpoint() {
  local checkpoint_path="$1"
  local expected_size="17028919541"
  local expected_sha256="08448992460d85ef1b992dd30585d5724d098d805a48399730c8e717027a6d9d"
  local actual_size
  local actual_sha256

  if [[ ! -f "$checkpoint_path" ]]; then
    echo "RollingForcing checkpoint is missing: ${checkpoint_path}" >&2
    return 1
  fi
  actual_size="$(stat -c '%s' "$checkpoint_path")"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "RollingForcing checkpoint size mismatch: expected ${expected_size}, got ${actual_size}" >&2
    return 1
  fi
  actual_sha256="$(sha256sum "$checkpoint_path" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "RollingForcing checkpoint SHA256 mismatch: expected ${expected_sha256}, got ${actual_sha256}" >&2
    return 1
  fi
}

download_rolling_forcing_assets() {
  local checkpoint_root="${WORLDFOUNDRY_CKPT_DIR}/RollingForcing"
  local checkpoint_path="${checkpoint_root}/checkpoints/rolling_forcing_dmd.pt"
  local wan_root="${WORLDFOUNDRY_CKPT_DIR}/Wan2.1-T2V-1.3B"

  if ! verify_rolling_forcing_checkpoint "$checkpoint_path"; then
    echo "Downloading TencentARC/RollingForcing from ModelScope ..."
    if download_modelscope_snapshot \
      TencentARC/RollingForcing \
      "$checkpoint_root" \
      checkpoints/rolling_forcing_dmd.pt \
      && verify_rolling_forcing_checkpoint "$checkpoint_path"; then
      :
    else
      echo "ModelScope download failed or was invalid; falling back to Hugging Face mirror." >&2
      rm -f "$checkpoint_path"
      download_hf_with_mirror \
        TencentARC/RollingForcing \
        checkpoints/rolling_forcing_dmd.pt \
        --local-dir "$checkpoint_root"
      verify_rolling_forcing_checkpoint "$checkpoint_path"
    fi
  fi

  if [[ ! -f "${wan_root}/config.json" ]]; then
    echo "Downloading Wan-AI/Wan2.1-T2V-1.3B from ModelScope ..."
    if ! download_modelscope_snapshot Wan-AI/Wan2.1-T2V-1.3B "$wan_root"; then
      echo "ModelScope base-model download failed; falling back to Hugging Face mirror." >&2
      download_hf_with_mirror Wan-AI/Wan2.1-T2V-1.3B --local-dir "$wan_root"
    fi
  fi
}

prepare_hunyuanvideo_t2v_layout() {
  local target_root="${WORLDFOUNDRY_CKPT_DIR}/HunyuanVideo"
  if [[ "$DOWNLOAD" == "1" ]]; then
    hf download tencent/HunyuanVideo --local-dir "$target_root"
  fi

  link_if_present "${WORLDFOUNDRY_HFD_ROOT}/tencent--HunyuanVideo" "$target_root"

  cat <<EOF

HunyuanVideo T2V checkpoint layout expected by the in-tree official runtime:
  ${target_root}
    hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt
    hunyuan-video-t2v-720p/vae/
    text_encoder/
    text_encoder_2/

Environment:
  This original HunyuanVideo T2V runtime uses the dedicated HunyuanVideo conda
  profile because it is pinned to the official diffusers 0.31 stack:
    bash scripts/setup/model_env_install.sh --model hunyuanvideo-t2v

Validated Studio demo:
  model-id=hunyuanvideo-t2v, torchrun --nproc_per_node=8, 129 frames, 720x1280,
  50 steps, prompt "A cat walks on the grass, realistic style.", seed 42.

EOF
}

prepare_hunyuanvideo_i2v_layout() {
  local target_root="${WORLDFOUNDRY_CKPT_DIR}/HunyuanVideo-I2V"
  if [[ "$DOWNLOAD" == "1" ]]; then
    hf download tencent/HunyuanVideo-I2V --local-dir "$target_root"
    hf download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir "$target_root/text_encoder_i2v"
    hf download openai/clip-vit-large-patch14 --local-dir "$target_root/text_encoder_2"
  fi

  link_if_present "${WORLDFOUNDRY_HFD_ROOT}/tencent--HunyuanVideo-I2V" "$target_root"
  link_if_present "${WORLDFOUNDRY_HFD_ROOT}/xtuner--llava-llama-3-8b-v1_1-transformers" "$target_root/text_encoder_i2v"
  link_if_present "${WORLDFOUNDRY_HFD_ROOT}/openai--clip-vit-large-patch14" "$target_root/text_encoder_2"

  cat <<EOF

HunyuanVideo I2V checkpoint layout expected by the in-tree official runtime:
  ${target_root}
    hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt
    hunyuan-video-i2v-720p/vae/
    text_encoder_i2v/
    text_encoder_2/

Environment:
  HunyuanVideo I2V reuses the dedicated HunyuanVideo conda profile with xfuser
  available for the official 8-rank sequence-parallel demo:
    bash scripts/setup/model_env_install.sh --model hunyuanvideo-i2v

Validated Studio demo target:
  model-id=hunyuanvideo-i2v, torchrun --nproc_per_node=8, official stability
  recipe, 129 frames, 720p, 50 steps, flow_shift=7.0, seed 0, input
  worldfoundry/data/test_cases/hunyuanvideo_i2v/0.jpg.

EOF
}

prepare_hunyuanvideo15_layout() {
  local target_root="${WORLDFOUNDRY_CKPT_DIR}/HunyuanVideo-1.5"
  if [[ "$DOWNLOAD" == "1" ]]; then
    hf download tencent/HunyuanVideo-1.5 --local-dir "$target_root"
    hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$target_root/text_encoder/llm"
    hf download google/byt5-small --local-dir "$target_root/text_encoder/byt5-small"
    hf download black-forest-labs/FLUX.1-Redux-dev --local-dir "$target_root/vision_encoder/siglip"
    "$PYTHON_BIN" - <<PY
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('AI-ModelScope/Glyph-SDXL-v2', local_dir='${target_root}/text_encoder/Glyph-SDXL-v2')
PY
  fi

  link_if_present "${WORLDFOUNDRY_CKPT_DIR}/Qwen2.5-VL-7B-Instruct" "$target_root/text_encoder/llm"
  link_if_present "${WORLDFOUNDRY_CKPT_DIR}/byt5-small" "$target_root/text_encoder/byt5-small"
  link_if_present "${WORLDFOUNDRY_CKPT_DIR}/Glyph-SDXL-v2" "$target_root/text_encoder/Glyph-SDXL-v2"
  link_if_present "${WORLDFOUNDRY_CKPT_DIR}/FLUX.1-Redux-dev" "$target_root/vision_encoder/siglip"

  cat <<EOF

HunyuanVideo-1.5 checkpoint layout expected by the in-tree official runtime:
  ${target_root}
    text_encoder/llm              # Qwen/Qwen2.5-VL-7B-Instruct
    text_encoder/byt5-small       # google/byt5-small
    text_encoder/Glyph-SDXL-v2    # AI-ModelScope/Glyph-SDXL-v2
    vision_encoder/siglip         # black-forest-labs/FLUX.1-Redux-dev
EOF
}

prepare_yume_layout() {
  local repo_id target_name
  case "$MODEL_ID" in
    yume-1p5|yume-1.5|yume1.5)
      repo_id="stdstu123/Yume-5B-720P"
      target_name="Yume-5B-720P"
      ;;
    *)
      repo_id="stdstu123/Yume-I2V-540P"
      target_name="Yume-I2V-540P"
      ;;
  esac
  if [[ "$DOWNLOAD" == "1" ]]; then
    hf download "$repo_id" --cache-dir "$CACHE_DIR"
  fi
  cat <<EOF

YUME checkpoint layout:
  Native Hugging Face cache is supported for ${repo_id}.
  Direct compatibility aliases are also accepted:
    ${WORLDFOUNDRY_CKPT_DIR}/${target_name}
    ${WORLDFOUNDRY_HFD_ROOT}/${repo_id//\//--}
    ${WORLDFOUNDRY_HFD_ROOT}/models--${repo_id//\//--}/snapshots/<revision>

Studio default demo parameters are recorded in:
  worldfoundry/data/models/runtime/profiles/${MODEL_ID}.yaml
EOF
}

prepare_cosmos3_layout() {
  local repo_id="nvidia/Cosmos3-Nano"
  local revision="$COSMOS3_NANO_REVISION"
  if [[ "$MODEL_ID" == "cosmos3-super" ]]; then
    repo_id="nvidia/Cosmos3-Super"
    revision="$COSMOS3_SUPER_REVISION"
  fi
  cat <<EOF

Cosmos3 checkpoint layout:
  Repository: ${repo_id}
  Pinned revision: ${revision}
  The repository already has an official Diffusers layout; no conversion is required.
  This script downloads/checks the exact revision in the native Hugging Face cache:
    ${CACHE_DIR}/models--${repo_id//\//--}/snapshots/${revision}/model_index.json
  Direct HFD aliases are accepted only when .hfd/repo_metadata.json records the same revision.
  An older directory may still be usable for compatibility testing, but is never reported as current.

Environment:
  bash scripts/setup/model_env_install.sh --model ${MODEL_ID}

Safety checker:
  The official default requires cosmos-guardrail and approved Hugging Face access to
  nvidia/Cosmos-1.0-Guardrail. Set an authorized HF_TOKEN after accepting its access terms.
  For an explicitly unscreened run, set enable_safety_checker=false at load time and
  enable_safety_check=false at call time; Workspace manifests preserve both settings.
EOF
}

inspect_cosmos3_checkpoint_revisions() {
  local repo_id="$1"
  local expected_revision="$2"
  REPO_ID="$repo_id" EXPECTED_REVISION="$expected_revision" \
  CACHE_DIR_VALUE="$CACHE_DIR" HFD_ROOT_VALUE="$WORLDFOUNDRY_HFD_ROOT" \
  CKPT_DIR_VALUE="$WORLDFOUNDRY_CKPT_DIR" REPORT_PATH_VALUE="$COSMOS3_CHECKPOINT_REPORT" \
  "$PYTHON_BIN" <<'PY'
import json
import os
import sys
from pathlib import Path

from worldfoundry.base_models.diffusion_model.video.cosmos3.worldfoundry_runtime import Cosmos3Runtime

repo_id = os.environ["REPO_ID"]
expected = os.environ["EXPECTED_REVISION"]
cache_dir = Path(os.environ["CACHE_DIR_VALUE"]).expanduser()
hfd_root = Path(os.environ["HFD_ROOT_VALUE"]).expanduser()
ckpt_dir = Path(os.environ["CKPT_DIR_VALUE"]).expanduser()
namespace = repo_id.replace("/", "--")
basename = repo_id.rsplit("/", 1)[-1]

candidates = [
    ("pinned_hf_snapshot", cache_dir / f"models--{namespace}" / "snapshots" / expected, expected),
    ("cache_direct_hfd", cache_dir / namespace, None),
    ("worldfoundry_hfd", hfd_root / namespace, None),
    ("worldfoundry_hfd_basename", hfd_root / basename, None),
    ("worldfoundry_hfd_pinned", hfd_root / f"{basename}-{expected[:8]}", None),
    ("workspace_hfd", ckpt_dir / namespace, None),
    ("workspace_checkpoint", ckpt_dir / basename, None),
    ("workspace_pinned_checkpoint", ckpt_dir / f"{basename}-{expected[:8]}", None),
]
seen = set()
current = []
rows = []
for label, path, known_revision in candidates:
    try:
        key = path.resolve() if path.exists() else path.absolute()
    except OSError:
        key = path.absolute()
    if key in seen or not path.exists():
        continue
    seen.add(key)
    actual = known_revision
    metadata_path = path / ".hfd" / "repo_metadata.json"
    if actual is None and metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        value = payload.get("sha") or payload.get("revision") or payload.get("commit")
        actual = str(value).strip() if value else None
    layout_plan = Cosmos3Runtime.plan(model_path=str(path))
    has_layout = not layout_plan["blocked"]
    if not has_layout:
        state = "incomplete"
    elif actual == expected:
        state = "current"
        current.append(path.resolve())
    elif actual:
        state = "stale"
    else:
        state = "unverified"
    print(
        f"Cosmos3 revision check: {state}: {path} "
        f"(actual={actual or 'unknown'}, expected={expected})",
        file=sys.stderr,
    )
    if layout_plan["blocked"]:
        for blocker in layout_plan["blockers"]:
            print(f"  layout blocker: {blocker}", file=sys.stderr)
    rows.append(
        {
            "label": label,
            "path": str(path),
            "state": state,
            "actual_revision": actual,
            "expected_revision": expected,
            "layout_blockers": list(layout_plan["blockers"]),
        }
    )

if current:
    selected_path = current[0]
else:
    selected_path = cache_dir / f"models--{namespace}" / "snapshots" / expected
    print(
        "Cosmos3 revision check: no complete checkpoint matching the pinned revision is ready; "
        f"expected {selected_path}",
        file=sys.stderr,
    )

report_path = Path(os.environ["REPORT_PATH_VALUE"])
try:
    report = json.loads(report_path.read_text(encoding="utf-8"))
except (FileNotFoundError, OSError, json.JSONDecodeError):
    report = {
        "schema_version": "worldfoundry-cosmos3-checkpoint-revision-report-v1",
        "ok": bool(current),
    }
report["model_download_check_ok"] = bool(report.get("ok"))
report["ok"] = bool(current)
report["cosmos3_revision_selection"] = {
    "repo_id": repo_id,
    "expected_revision": expected,
    "current_revision_ready": bool(current),
    "selected_model_ref": str(selected_path),
    "candidates": rows,
}
report_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary_path.replace(report_path)

# Preserve the exact future target so the printed demo command cannot silently use a stale directory.
print(selected_path)
PY
}

prepare_lyra_layout() {
  cat <<EOF

Lyra checkpoint layout:
  Lyra-1 expects local Lyra-1/Cosmos assets. The Studio profile uses:
    ${WORLDFOUNDRY_CKPT_DIR}/lyra-1
    ${WORLDFOUNDRY_CKPT_DIR}/cosmos-gen3c
  Studio demo: static single-image SDG, trajectory=zoom_in, 121 frames,
  1280x704, 24 fps, num_steps=35, seed=1.

  Lyra-2 expects the official Lyra-2.0 directory with:
    checkpoints/model/
    checkpoints/text_encoder/negative_prompt.pt
    checkpoints/recon/model.pt

  Accepted roots include:
    ${WORLDFOUNDRY_HFD_ROOT}/Lyra-2.0
    ${WORLDFOUNDRY_HFD_ROOT}/nvidia--Lyra-2.0
    ${WORLDFOUNDRY_CKPT_DIR}/Lyra-2.0
    ${WORLDFOUNDRY_CKPT_DIR}/Lyra-2

EOF
}

prepare_fantasyworld_layout() {
  if [[ "$DOWNLOAD" == "1" ]]; then
    case "$MODEL_ID" in
      fantasyworld-wan21)
        hf download acvlab/FantasyWorld-Wan2.1-I2V-14B-480P --cache-dir "$CACHE_DIR"
        hf download Wan-AI/Wan2.1-I2V-14B-480P --cache-dir "$CACHE_DIR"
        ;;
      *)
        hf download acvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera --cache-dir "$CACHE_DIR"
        hf download alibaba-pai/Wan2.2-Fun-A14B-Control-Camera --cache-dir "$CACHE_DIR"
        hf download alibaba-pai/Wan2.2-Fun-Reward-LoRAs --cache-dir "$CACHE_DIR"
        ;;
    esac
    hf download Ruicheng/moge-2-vitl-normal --cache-dir "$CACHE_DIR"
  fi
  cat <<EOF

FantasyWorld checkpoint layout:
  Wan2.1 variant:
    acvlab/FantasyWorld-Wan2.1-I2V-14B-480P
    Wan-AI/Wan2.1-I2V-14B-480P
    Ruicheng/moge-2-vitl-normal

  Wan2.2 variant:
    acvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera
    alibaba-pai/Wan2.2-Fun-A14B-Control-Camera
    alibaba-pai/Wan2.2-Fun-Reward-LoRAs
    Ruicheng/moge-2-vitl-normal

EOF
}

prepare_lingbot_world_layout() {
  if [[ "$DOWNLOAD" == "1" ]]; then
    hf download robbyant/lingbot-world-base-cam --cache-dir "$CACHE_DIR"
  fi
  cat <<EOF

LingBot-World checkpoint layout:
  Native Hugging Face cache is supported for robbyant/lingbot-world-base-cam.
  Direct HFD alias is also accepted:
    ${WORLDFOUNDRY_HFD_ROOT}/robbyant--lingbot-world-base-cam

Environment:
  LingBot-World currently uses the dedicated lingbot-world conda profile:
    bash scripts/setup/model_env_install.sh --model lingbot-world

EOF
}

prepare_lingbot_world_v2_layout() {
  if [[ "$DOWNLOAD" == "1" ]]; then
    hf download robbyant/lingbot-world-v2-14b-causal-fast --cache-dir "$CACHE_DIR"
  fi
  cat <<EOF

LingBot-World-V2 causal-fast checkpoint layout:
  Native Hugging Face cache is supported for:
    robbyant/lingbot-world-v2-14b-causal-fast
  Direct HFD alias is also accepted:
    ${WORLDFOUNDRY_HFD_ROOT}/robbyant--lingbot-world-v2-14b-causal-fast

Required files:
  models_t5_umt5-xxl-enc-bf16.pth
  google/umt5-xxl/
  Wan2.1_VAE.pth
  transformers/config.json

Action input must be a directory containing poses.npy [F,4,4] and
intrinsics.npy [F,4]. The default public recipe uses eight GPUs:
  bash scripts/setup/model_env_install.sh --model lingbot-world-v2

License: CC BY-NC-SA 4.0 (non-commercial, share-alike).

EOF
}

prepare_matrix_game_1_layout() {
  if [[ "$DOWNLOAD" == "1" ]]; then
    hf download Skywork/Matrix-Game --cache-dir "$CACHE_DIR"
  fi
  cat <<EOF

Matrix-Game-1 checkpoint layout:
  Native Hugging Face cache is supported for Skywork/Matrix-Game.
  Offline aliases accepted by the in-tree runner include:
    ${WORLDFOUNDRY_HFD_ROOT}/Skywork--Matrix-Game
    ${WORLDFOUNDRY_HFD_ROOT}/models--Skywork--Matrix-Game/snapshots/<revision>
    ${WORLDFOUNDRY_CKPT_DIR}/Matrix-Game

Environment:
  Use worldfoundry-unified-cu128 on modern NVIDIA hosts. The upstream environment
  file is kept at worldfoundry/data/models/runtime/configs/matrix_game_1/environment.yml
  for provenance, but the Studio/API runner has been exercised in the unified env.

EOF
}

prepare_solaris_layout() {
  local pretrained_root="${WORLDFOUNDRY_CKPT_DIR}/solaris"
  local dataset_root="${pretrained_root}/datasets"
  if [[ "$DOWNLOAD" == "1" ]]; then
    hf download nyu-visionx/solaris --local-dir "$pretrained_root"
    hf download nyu-visionx/solaris-eval-datasets --repo-type dataset --local-dir "$dataset_root"
  fi

  cat <<EOF

Solaris checkpoint and eval-data layout:
  Runtime: in-tree inference-only runtime under
    worldfoundry/synthesis/visual_generation/solaris/solaris_runtime

  Native Hugging Face sources:
    nyu-visionx/solaris
    nyu-visionx/solaris-eval-datasets

  Default local mirror layout:
    ${pretrained_root}/solaris.pt
    ${pretrained_root}/clip.pt
    ${pretrained_root}/vae.pt
    ${dataset_root}/translationEval

Environment:
  Solaris is validated in the unified CUDA 12.8 environment. It uses JAX/Flax
  Orbax checkpoints and needs one generated sample per visible GPU. For an
  eight-GPU official-style Studio job, use eval_num_samples=8.

EOF
}

prepare_astra_layout() {
  local astra_repo="EvanEternal/Astra"
  local wan_repo="Wan-AI/Wan2.1-T2V-1.3B"
  if [[ "$DOWNLOAD" == "1" ]]; then
    hf download "$astra_repo" --cache-dir "$CACHE_DIR"
    hf download "$wan_repo" --cache-dir "$CACHE_DIR"
  fi

  cat <<EOF

Astra checkpoint layout:
  Native Hugging Face cache/download is the default:
    ${astra_repo}
    ${wan_repo}

  The in-tree runtime accepts those repo ids directly. Shared local mirrors can
  be linked into the Hugging Face cache with:
    bash scripts/setup/link_hf_checkpoints.sh --default-world

  Local mirror layout used by the official-style demo:
    ${WORLDFOUNDRY_CKPT_DIR}/Astra/models/Astra/checkpoints/diffusion_pytorch_model.ckpt
    ${WORLDFOUNDRY_CKPT_DIR}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
    ${WORLDFOUNDRY_CKPT_DIR}/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
    ${WORLDFOUNDRY_CKPT_DIR}/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth

Environment:
  Astra is validated in worldfoundry-unified-cu128 with the in-tree
  inference-only runtime under worldfoundry/synthesis/visual_generation/kling/astra_runtime.

EOF
}

prepare_open_sora_plan_layout() {
  local model_target="${WORLDFOUNDRY_HFD_ROOT}/LanguageBind--Open-Sora-Plan-v1.3.0"
  local mt5_target="${WORLDFOUNDRY_CKPT_DIR}/mt5-xxl"
  if [[ "$DOWNLOAD" == "1" ]]; then
    hf download LanguageBind/Open-Sora-Plan-v1.3.0 --local-dir "$model_target"
    hf download google/mt5-xxl --local-dir "$mt5_target" --exclude "tf_model.h5"
  fi
  cat <<EOF

Open-Sora-Plan checkpoint layout:
  The validated in-tree v1.3 GPU path expects:
    ${model_target}
      any93x640x640/
      vae/
    ${mt5_target}

  The text encoder must be official google/mt5-xxl. Do not substitute Wan-style
  google/umt5-xxl; it produces visually invalid Open-Sora-Plan outputs.

Validated Studio demo:
  model-id=open-sora-plan, torchrun --nproc_per_node=8, 93 frames, 352x640,
  fps 18, 100 EulerAncestralDiscrete steps, guidance_scale 7.5, seed 1234.

EOF
}

prepare_cogvideox_layout() {
  local repo_id target_name
  case "$MODEL_ID" in
    cogvideox-2b-t2v)
      repo_id="THUDM/CogVideoX-2b"
      target_name="CogVideoX-2b"
      ;;
    cogvideox-5b-t2v)
      repo_id="THUDM/CogVideoX-5b"
      target_name="CogVideoX-5b"
      ;;
    cogvideox-5b-i2v)
      repo_id="THUDM/CogVideoX-5b-I2V"
      target_name="CogVideoX-5b-I2V"
      ;;
    *)
      repo_id="THUDM/CogVideoX-5b"
      target_name="CogVideoX-5b"
      ;;
  esac
  if [[ "$DOWNLOAD" == "1" ]]; then
    hf download "$repo_id" --cache-dir "$CACHE_DIR"
  fi
  cat <<EOF

CogVideoX checkpoint layout:
  The in-tree Diffusers runtime defaults to native Hugging Face loading from:
    ${repo_id}

  For offline mirrors, pass model_path in the Studio/API job or symlink a local
  mirror to one of these conventional roots:
    ${WORLDFOUNDRY_CKPT_DIR}/${target_name}
    ${WORLDFOUNDRY_HFD_ROOT}/models--${repo_id//\//--}/snapshots/<revision>

EOF
}

prepare_recent_video_layout() {
  case "$MODEL_ID" in
    i2vgen-xl)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download ali-vilab/i2vgen-xl --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

I2VGen-XL checkpoint layout:
  Native Hugging Face cache is supported for ali-vilab/i2vgen-xl.
  Offline aliases are also accepted when passed as model_path:
    ${WORLDFOUNDRY_CKPT_DIR}/i2vgen-xl
    ${WORLDFOUNDRY_HFD_ROOT}/ali-vilab--i2vgen-xl

EOF
      ;;
    dynamicrafter-512-i2v)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download Doubiiu/DynamiCrafter_512 --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

DynamiCrafter 512 checkpoint layout:
  Native Hugging Face cache is supported for Doubiiu/DynamiCrafter_512.
  The in-tree runtime also accepts the upstream-style local checkpoint:
    ${WORLDFOUNDRY_CKPT_DIR}/DynamiCrafter_512/model.ckpt

EOF
      ;;
    dynamicrafter-1024-i2v)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download Doubiiu/DynamiCrafter_1024 --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

DynamiCrafter 1024 checkpoint layout:
  Native Hugging Face cache target: Doubiiu/DynamiCrafter_1024.
  Expected upstream-style local checkpoint:
    ${WORLDFOUNDRY_CKPT_DIR}/DynamiCrafter_1024/model.ckpt

EOF
      ;;
    allegro-ti2v)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download rhymes-ai/Allegro-TI2V --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

Allegro TI2V checkpoint layout:
  Native Hugging Face cache is supported for rhymes-ai/Allegro-TI2V.
  Offline aliases can be staged under:
    ${WORLDFOUNDRY_CKPT_DIR}/Allegro-TI2V
    ${WORLDFOUNDRY_HFD_ROOT}/rhymes-ai--Allegro-TI2V

EOF
      ;;
    wan2.2-ti2v-5b|wan2.2-ti2v-5b-1280x704-121f)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download Wan-AI/Wan2.2-TI2V-5B --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

Wan2.2 TI2V 5B checkpoint layout:
  Native Hugging Face cache is supported for Wan-AI/Wan2.2-TI2V-5B.
  Offline aliases can be staged under:
    ${WORLDFOUNDRY_CKPT_DIR}/Wan2.2-TI2V-5B
    ${WORLDFOUNDRY_HFD_ROOT}/Wan-AI--Wan2.2-TI2V-5B

EOF
      ;;
    stable-virtual-camera)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download stabilityai/stable-virtual-camera --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

Stable Virtual Camera checkpoint layout:
  Hugging Face repo stabilityai/stable-virtual-camera is gated. Users must accept
  the model terms and provide HF authentication before --download can succeed.
  The in-tree adapter expects or can be pointed at:
    ${WORLDFOUNDRY_CKPT_DIR}/stable-virtual-camera/modelv1.1.safetensors

EOF
      ;;
    skyreels-v2)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download Skywork/SkyReels-V2-DF-1.3B-540P --cache-dir "$CACHE_DIR"
        hf download Skywork/SkyReels-V2-I2V-1.3B-540P --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

SkyReels-V2 checkpoint layout:
  Small-route native Hugging Face checkpoints:
    Skywork/SkyReels-V2-DF-1.3B-540P
    Skywork/SkyReels-V2-I2V-1.3B-540P

EOF
      ;;
    zeroscope)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download cerspense/zeroscope_v2_576w --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

ZeroScope checkpoint layout:
  Native Hugging Face cache is supported for cerspense/zeroscope_v2_576w.
  Offline aliases can be staged under:
    ${WORLDFOUNDRY_HFD_ROOT}/cerspense--zeroscope_v2_576w
    ${WORLDFOUNDRY_HFD_ROOT}/models--cerspense--zeroscope_v2_576w/snapshots/<revision>

EOF
      ;;
    modelscope-t2v)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download ali-vilab/modelscope-damo-text-to-video-synthesis --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

ModelScopeT2V checkpoint layout:
  Native Hugging Face cache is supported for ali-vilab/modelscope-damo-text-to-video-synthesis.
  Offline aliases can be staged under:
    ${WORLDFOUNDRY_CKPT_DIR}/modelscope-damo-text-to-video-synthesis
    ${WORLDFOUNDRY_HFD_ROOT}/ali-vilab--modelscope-damo-text-to-video-synthesis
    ${WORLDFOUNDRY_HFD_ROOT}/models--ali-vilab--modelscope-damo-text-to-video-synthesis/snapshots/<revision>

EOF
      ;;
    animatediff)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download guoyww/animatediff --cache-dir "$CACHE_DIR"
        hf download guoyww/animatediff_t2i_backups --cache-dir "$CACHE_DIR"
        hf download stable-diffusion-v1-5/stable-diffusion-v1-5 --cache-dir "$CACHE_DIR"
        hf download openai/clip-vit-large-patch14 --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

AnimateDiff checkpoint layout:
  Native Hugging Face cache is supported for guoyww/animatediff and
  guoyww/animatediff_t2i_backups. The default Studio profile also needs the
  SD1.5 base and CLIP text encoder:
    ${WORLDFOUNDRY_CKPT_DIR}/animatediff/mm_sd_v15_v2.ckpt
    ${WORLDFOUNDRY_CKPT_DIR}/animatediff_t2i_backups/realisticVisionV60B1_v51VAE.safetensors
    ${WORLDFOUNDRY_CKPT_DIR}/stable-diffusion-v1-5
    ${WORLDFOUNDRY_HFD_ROOT}/models--openai--clip-vit-large-patch14/snapshots/<revision>

EOF
      ;;
    krea-realtime-video)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download krea/krea-realtime-video --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

Krea Realtime Video checkpoint layout:
  Native Hugging Face cache is supported for krea/krea-realtime-video.
  Offline alias:
    ${WORLDFOUNDRY_CKPT_DIR}/krea-realtime-video

EOF
      ;;
    framepack)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download lllyasviel/FramePackI2V_HY --cache-dir "$CACHE_DIR"
        hf download hunyuanvideo-community/HunyuanVideo --cache-dir "$CACHE_DIR"
        hf download lllyasviel/flux_redux_bfl --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

FramePack checkpoint layout:
  Native Hugging Face cache is supported for lllyasviel/FramePackI2V_HY.
  The in-tree wrapper also needs the HunyuanVideo support repo and FLUX Redux
  support assets. Offline aliases:
    ${WORLDFOUNDRY_CKPT_DIR}/FramePackI2V_HY
    ${WORLDFOUNDRY_HFD_ROOT}/hub/models--hunyuanvideo-community--HunyuanVideo
    ${WORLDFOUNDRY_CKPT_DIR}/FLUX.1-Redux-dev

EOF
      ;;
    motionctrl)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download TencentARC/MotionCtrl --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

MotionCtrl checkpoint layout:
  Native Hugging Face cache is supported for TencentARC/MotionCtrl.
  Offline aliases can be staged under:
    ${WORLDFOUNDRY_CKPT_DIR}/MotionCtrl/motionctrl.pth
    ${WORLDFOUNDRY_HFD_ROOT}/TencentARC--MotionCtrl/motionctrl.pth

EOF
      ;;
    easyanimate-i2v)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download alibaba-pai/EasyAnimateV5.1-7b-zh-InP --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

EasyAnimate I2V checkpoint layout:
  Native Hugging Face cache is supported for alibaba-pai/EasyAnimateV5.1-7b-zh-InP.
  Official EasyAnimate V5.1 I2V demos use the InP checkpoint family; the
  non-InP alibaba-pai/EasyAnimateV5.1-7b-zh checkpoint is text-to-video.
  The Studio default also accepts the local HFD-style mirror:
    ${WORLDFOUNDRY_CKPT_DIR}/hfd/alibaba-pai--EasyAnimateV5.1-7b-zh-InP
    ${WORLDFOUNDRY_HFD_ROOT}/alibaba-pai--EasyAnimateV5.1-7b-zh-InP

EOF
      ;;
    hunyuan-worldplay)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download tencent/HY-WorldPlay --cache-dir "$CACHE_DIR"
        prepare_hunyuanvideo15_layout
      fi
      cat <<EOF

HY-WorldPlay checkpoint layout:
  Native Hugging Face cache is supported for tencent/HY-WorldPlay.
  The runtime also resolves the HunyuanVideo-1.5 video-model assets through
  the default Hugging Face cache or the local aliases below.
  Offline aliases can be staged under:
    ${WORLDFOUNDRY_CKPT_DIR}/HY-WorldPlay
    ${WORLDFOUNDRY_HFD_ROOT}/tencent--HY-WorldPlay
    ${WORLDFOUNDRY_CKPT_DIR}/HunyuanVideo-1.5
    ${WORLDFOUNDRY_HFD_ROOT}/tencent--HunyuanVideo-1.5

  On remote or shared storage, optional local staging avoids every rank
  cold-reading the same large safetensors file:
    export WORLDFOUNDRY_HY_WORLDPLAY_LOCAL_CKPT_CACHE_DIR=/local/fast/worldfoundry_ckpt

EOF
      ;;
    longcat-video)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download meituan-longcat/LongCat-Video --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

LongCat-Video checkpoint layout:
  Native Hugging Face cache is supported for meituan-longcat/LongCat-Video.
  The Studio default also accepts local aliases:
    ${WORLDFOUNDRY_CKPT_DIR}/LongCat-Video
    ${WORLDFOUNDRY_HFD_ROOT}/meituan-longcat--LongCat-Video

EOF
      ;;
    helios)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download BestWishYsh/Helios-Distilled --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

Helios checkpoint layout:
  The default WorldFoundry route uses BestWishYsh/Helios-Distilled.
  Accepted local aliases:
    ${WORLDFOUNDRY_CKPT_DIR}/Helios-Distilled
    ${WORLDFOUNDRY_HFD_ROOT}/BestWishYsh--Helios-Distilled

  Install the pinned official inference environment with:
    bash scripts/setup/model_env_install.sh --model helios

  Base and Mid checkpoints can be selected by overriding checkpoint_path and
  setting variant to base or mid in the inference call JSON.

EOF
      ;;
    gamma-world)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download chijw/Gamma-World --cache-dir "$CACHE_DIR"
        hf download nvidia/Cosmos-Reason1-7B --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

Gamma-World checkpoint layout:
  World model and tokenizer:
    ${WORLDFOUNDRY_CKPT_DIR}/Gamma-World
    ${WORLDFOUNDRY_HFD_ROOT}/chijw--Gamma-World
  Gated text encoder:
    ${WORLDFOUNDRY_CKPT_DIR}/Cosmos-Reason1-7B
    ${WORLDFOUNDRY_HFD_ROOT}/nvidia--Cosmos-Reason1-7B

  Runtime source:
    ${WORLDFOUNDRY_SOURCE_ROOT}/worldfoundry/synthesis/visual_generation/gamma_world

  Accept the NVIDIA Open Model License for Cosmos-Reason1-7B and authenticate
  with Hugging Face before downloading. The default mode is causal_few_step.

EOF
      ;;
    lingbot-video)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download robbyant/lingbot-video-dense-1.3b --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

LingBot-Video checkpoint layout:
  Dense default:
    ${WORLDFOUNDRY_HFD_ROOT}/robbyant--lingbot-video-dense-1.3b
  Optional MoE + refiner variant:
    ${WORLDFOUNDRY_HFD_ROOT}/robbyant--lingbot-video-moe-30b-a3b

  Install the dedicated inference environment before running:
    bash scripts/setup/model_env_install.sh --model lingbot-video

  The runtime consumes structured JSON captions. The prompt rewriter is not
  included in this inference-only integration.

EOF
      ;;
    self-forcing)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download gdhe17/Self-Forcing --cache-dir "$CACHE_DIR"
        hf download Wan-AI/Wan2.1-T2V-1.3B --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

Self-Forcing checkpoint layout:
  Runtime: in-tree infer-only runtime under
    worldfoundry/synthesis/visual_generation/forcing/self_forcing_runtime
  Native Hugging Face cache is supported for gdhe17/Self-Forcing.
  Wan2.1 base weights are resolved from native HF cache or local aliases under:
    ${WORLDFOUNDRY_CKPT_DIR}/Wan2.1-T2V-1.3B
    ${WORLDFOUNDRY_CKPT_DIR}/Wan2.1-T2V-14B
  Local official checkpoint alias:
    ${WORLDFOUNDRY_CKPT_DIR}/Self-Forcing/checkpoints/self_forcing_dmd.pt

EOF
      ;;
    causal-forcing)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download zhuhz22/Causal-Forcing --cache-dir "$CACHE_DIR"
        hf download Wan-AI/Wan2.1-T2V-1.3B --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

Causal-Forcing checkpoint layout:
  Runtime: in-tree infer-only runtime under
    worldfoundry/synthesis/visual_generation/forcing/causal_forcing_runtime
  Native Hugging Face cache is supported for zhuhz22/Causal-Forcing.
  Wan2.1 base weights are resolved from native HF cache or local aliases under:
    ${WORLDFOUNDRY_CKPT_DIR}/Wan2.1-T2V-1.3B
    ${WORLDFOUNDRY_CKPT_DIR}/Wan2.1-T2V-14B
  Local chunk-wise checkpoint alias:
    ${WORLDFOUNDRY_CKPT_DIR}/Causal-Forcing/chunkwise/causal_forcing.pt

EOF
      ;;
    rolling-forcing)
      if [[ "$DOWNLOAD" == "1" ]]; then
        download_rolling_forcing_assets
      fi
      cat <<EOF

RollingForcing checkpoint layout:
  Runtime: flat, in-tree inference-only package under
    worldfoundry/synthesis/visual_generation/rolling_forcing
  Runtime source never depends on an external RollingForcing checkout.
  Downloads prefer the official ModelScope mirror and fall back to Hugging Face.
  Native Hugging Face cache is also supported for TencentARC/RollingForcing.
  Local official checkpoint alias:
    ${WORLDFOUNDRY_CKPT_DIR}/RollingForcing/checkpoints/rolling_forcing_dmd.pt
  Wan2.1 base weights:
    ${WORLDFOUNDRY_CKPT_DIR}/Wan2.1-T2V-1.3B

  License restriction: academic use only; commercial and production use are prohibited.

EOF
      ;;
    worldgen)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download LeoXie/WorldGen --cache-dir "$CACHE_DIR"
        hf download black-forest-labs/FLUX.1-dev --cache-dir "$CACHE_DIR"
        hf download black-forest-labs/FLUX.1-Fill-dev --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

WorldGen checkpoint layout:
  WorldGen LoRA assets come from LeoXie/WorldGen. The standard non-low-vram path
  also needs FLUX.1-dev and FLUX.1-Fill-dev access through native Hugging Face
  cache or local aliases:
    ${WORLDFOUNDRY_CKPT_DIR}/WorldGen/models--WorldGen-Flux-Lora/
    ${WORLDFOUNDRY_CKPT_DIR}/FLUX.1-dev
    ${WORLDFOUNDRY_CKPT_DIR}/FLUX.1-Fill-dev

EOF
      ;;
    dualcamctrl)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download FayeHongfeiZhang/DualCamCtrl checkpoints/dualcamctrl_diffusion_transformer.pt --cache-dir "$CACHE_DIR"
        hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera --cache-dir "$CACHE_DIR"
        hf download Wan-AI/Wan2.1-T2V-1.3B --cache-dir "$CACHE_DIR"
        hf download Wan-AI/Wan2.1-I2V-14B-480P --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

DualCamCtrl checkpoint layout:
  Runtime: in-tree infer-only runtime under
    worldfoundry/synthesis/visual_generation/dualcamctrl
  Native Hugging Face cache is supported for:
    FayeHongfeiZhang/DualCamCtrl
    alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera
    Wan-AI/Wan2.1-T2V-1.3B
    Wan-AI/Wan2.1-I2V-14B-480P
  Local pre-staged snapshots can be reused by setting WORLDFOUNDRY_MODEL_DIR.
  The default Studio recipe uses the in-tree seaside RGB/depth/camera fixtures
  with 61 frames, 320x480, 50 steps, fps 10, seed 42.

EOF
      ;;
    pusa-vidgen)
      if [[ "$DOWNLOAD" == "1" ]]; then
        hf download RaphaelLiu/Pusa-Wan2.2-V1 --cache-dir "$CACHE_DIR"
        hf download Wan-AI/Wan2.2-T2V-A14B --cache-dir "$CACHE_DIR"
        hf download lightx2v/Wan2.2-Lightning --cache-dir "$CACHE_DIR"
      fi
      cat <<EOF

Pusa VidGen V1 checkpoint layout:
  Runtime: in-tree Pusa V1 Wan2.2 runner under
    worldfoundry/synthesis/visual_generation/pusa_vidgen
  Native Hugging Face cache is supported for:
    RaphaelLiu/Pusa-Wan2.2-V1
    Wan-AI/Wan2.2-T2V-A14B
    lightx2v/Wan2.2-Lightning
  Offline aliases accepted by the Studio default profile:
    ${WORLDFOUNDRY_HFD_ROOT}/RaphaelLiu--Pusa-Wan2.2-V1
    ${WORLDFOUNDRY_CKPT_DIR}/Wan2.2-T2V-A14B
    ${WORLDFOUNDRY_CKPT_DIR}/Wan2.2-Lightning

EOF
      ;;
  esac
}

if [[ "$SKIP_ENV" != "1" ]]; then
  env_args=(bash "$ROOT/scripts/setup/model_env_install.sh" --model "$MODEL_ID")
  [[ -n "$HOME_ROOT" ]] && env_args+=(--home "$HOME_ROOT")
  [[ -n "$ENV_ROOT" ]] && env_args+=(--env-root "$ENV_ROOT")
  [[ "$VERIFY_ENV_ONLY" == "1" ]] && env_args+=(--verify-only)
  [[ "$SKIP_FLASH_ATTN" == "1" ]] && env_args+=(--skip-flash-attn)
  [[ "$ALLOW_NO_CUDA" == "1" ]] && env_args+=(--allow-no-cuda)
  "${env_args[@]}"
fi

download_args=("$PYTHON_BIN" -m worldfoundry.cli zoo model-download --model-id "$ZOO_MODEL_ID" --cache-dir "$CACHE_DIR" --check-local --json)
if [[ "$MODEL_ID" == "helios" ]]; then
  download_args+=(--repo-id BestWishYsh/Helios-Distilled)
elif [[ "$MODEL_ID" == "gamma-world" ]]; then
  download_args+=(--repo-id chijw/Gamma-World)
elif [[ "$MODEL_ID" == "lingbot-video" ]]; then
  download_args+=(--repo-id robbyant/lingbot-video-dense-1.3b)
elif [[ "$MODEL_ID" == "cosmos3-super" ]]; then
  COSMOS3_REPO_ID="nvidia/Cosmos3-Super"
  COSMOS3_EXPECTED_REVISION="$COSMOS3_SUPER_REVISION"
  download_args+=(--repo-id "$COSMOS3_REPO_ID")
elif [[ "$MODEL_ID" == "cosmos3" || "$MODEL_ID" == "cosmos3-nano" ]]; then
  COSMOS3_REPO_ID="nvidia/Cosmos3-Nano"
  COSMOS3_EXPECTED_REVISION="$COSMOS3_NANO_REVISION"
  download_args+=(--repo-id "$COSMOS3_REPO_ID")
fi
if [[ -n "$COSMOS3_REPO_ID" ]]; then
  mkdir -p "$OUTPUT_DIR"
  COSMOS3_CHECKPOINT_REPORT="${OUTPUT_DIR}/cosmos3-checkpoint-report.json"
  download_args+=(--report-path "$COSMOS3_CHECKPOINT_REPORT")
fi
if [[ "$DOWNLOAD" == "1" ]]; then
  download_args+=(--execute --disable-xet)
fi

case "$MODEL_ID" in
  hunyuanvideo-t2v)
    prepare_hunyuanvideo_t2v_layout
    ;;
  hunyuanvideo-i2v)
    prepare_hunyuanvideo_i2v_layout
    ;;
  hunyuanvideo-1.5|hunyuanvideo-1.5-t2v|hunyuanvideo-1.5-i2v)
    prepare_hunyuanvideo15_layout
    ;;
  yume|yume-1p5|yume-1.5|yume1.5)
    prepare_yume_layout
    ;;
  cosmos3|cosmos3-nano|cosmos3-super)
    prepare_cosmos3_layout
    ;;
  lyra|lyra-1|lyra-2)
    prepare_lyra_layout
    ;;
  fantasyworld|fantasyworld-wan21|fantasyworld-wan22)
    prepare_fantasyworld_layout
    ;;
  lingbot-world)
    prepare_lingbot_world_layout
    ;;
  lingbot-world-v2)
    prepare_lingbot_world_v2_layout
    ;;
  matrix-game-1)
    prepare_matrix_game_1_layout
    ;;
  astra)
    prepare_astra_layout
    ;;
  solaris)
    prepare_solaris_layout
    ;;
  open-sora-plan|opensora-plan)
    prepare_open_sora_plan_layout
    ;;
  cogvideox|cogvideox-2b-t2v|cogvideox-5b-t2v|cogvideox-5b-i2v)
    prepare_cogvideox_layout
    ;;
  i2vgen-xl|dynamicrafter-512-i2v|dynamicrafter-1024-i2v|allegro-ti2v|wan2.2-ti2v-5b|wan2.2-ti2v-5b-1280x704-121f|stable-virtual-camera|skyreels-v2|zeroscope|modelscope-t2v|animatediff|krea-realtime-video|framepack|motionctrl|easyanimate-i2v|hunyuan-worldplay|longcat-video|helios|gamma-world|lingbot-video|self-forcing|causal-forcing|rolling-forcing|worldgen|pusa-vidgen|dualcamctrl)
    prepare_recent_video_layout
    ;;
esac

set +e
"${download_args[@]}"
download_status=$?
set -e

if [[ "$MODEL_ID" == "cosmos3" || "$MODEL_ID" == "cosmos3-nano" || "$MODEL_ID" == "cosmos3-super" ]]; then
  COSMOS3_PINNED_MODEL_REF="$(inspect_cosmos3_checkpoint_revisions "$COSMOS3_REPO_ID" "$COSMOS3_EXPECTED_REVISION")"
  if "$PYTHON_BIN" - "$COSMOS3_CHECKPOINT_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ready = bool(report.get("cosmos3_revision_selection", {}).get("current_revision_ready"))
raise SystemExit(0 if ready else 1)
PY
  then
    # A verified exact-revision checkpoint on shared CKPT/HFD storage is just
    # as runnable as a native Hugging Face cache snapshot.
    download_status=0
  fi
  cosmos3_variant="cosmos3-nano"
  if [[ "$MODEL_ID" == "cosmos3-super" ]]; then
    cosmos3_variant="cosmos3-super"
  fi
  cat <<EOF

Suggested inference command:
  PYTHONPATH=${WORLDFOUNDRY_SOURCE_ROOT} python -m worldfoundry.studio.workspace_job infer \
    --model-id cosmos3 \
    --variant-id ${cosmos3_variant} \
    --model-ref '${COSMOS3_PINNED_MODEL_REF}' \
    --prompt 'A robot arm carefully cleans a plate in a bright kitchen.' \
    --call-json '{"num_frames":189,"height":720,"width":1280,"fps":24,"num_inference_steps":35,"guidance_scale":6.0,"seed":0}' \
    --output-path ${OUTPUT_DIR}/cosmos3.mp4 \
    --output-dir ${OUTPUT_DIR} \
    --device cuda

Checkpoint verification report:
  ${COSMOS3_CHECKPOINT_REPORT}
EOF
elif [[ "$MODEL_ID" == "lingbot-world-v2" ]]; then
  cat <<EOF

Suggested inference command:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTHONPATH=${WORLDFOUNDRY_SOURCE_ROOT} python -m worldfoundry.studio.workspace_job infer \
    --model-id lingbot-world-v2 \
    --input-path /path/to/image.jpg \
    --prompt "A cinematic first-person journey through a detailed world." \
    --call-json '{"action_path":"/path/to/action_directory","frame_num":361,"chunk_size":4,"local_attn_size":18,"sink_size":6,"nproc_per_node":8,"return_dict":true}' \
    --output-path ${OUTPUT_DIR}/lingbot-world-v2.mp4 \
    --output-dir ${OUTPUT_DIR} \
    --device cuda
EOF
elif [[ "$MODEL_ID" == "dualcamctrl" ]]; then
  cat <<EOF

Suggested inference command:
  PYTHONPATH=${WORLDFOUNDRY_SOURCE_ROOT} python -m worldfoundry.studio.workspace_job infer \\
    --model-id dualcamctrl \\
    --prompt "High aerial view over a British seaside town on a sunny afternoon." \\
    --call-json '{"demo_name":"seaside","num_frames":61,"height":320,"width":480,"num_inference_steps":50,"fps":10,"seed":42,"cfg_scale":5.0,"original_height":360,"original_width":640,"tiled":true,"return_control_latents":true}' \\
    --output-path ${OUTPUT_DIR}/seaside.mp4 \\
    --output-dir ${OUTPUT_DIR} \\
    --device cuda
EOF
elif [[ "$MODEL_ID" == "gen3c" ]]; then
  cat <<EOF

Suggested inference command:
  PYTHONPATH=${WORLDFOUNDRY_SOURCE_ROOT} python -m worldfoundry.studio.workspace_job infer \\
    --model-id gen3c \\
    --input-path worldfoundry/data/test_cases/gen3c/image.png \\
    --interactions left \\
    --call-json '{"trajectory":"left","camera_rotation":"center_facing","movement_distance":0.3,"guidance":1.0,"num_steps":35,"num_video_frames":121,"fps":24,"height":704,"width":1280,"seed":1,"num_gpus":8,"noise_aug_strength":0.0,"filter_points_threshold":0.05,"foreground_masking":true,"disable_prompt_upsampler":true,"disable_guardrail":true,"return_dict":true}' \\
    --output-dir ${OUTPUT_DIR} \\
    --device cuda
EOF
elif [[ "$MODEL_ID" == "astra" ]]; then
  cat <<EOF

Suggested inference command:
  ASTRA_PROMPT='A sunlit European street lined with historic buildings and vibrant greenery creates a warm, charming, and inviting atmosphere. The scene shows a picturesque open square paved with red bricks, surrounded by classic narrow townhouses featuring tall windows, gabled roofs, and dark-painted facades. On the right side, a lush arrangement of potted plants and blooming flowers adds rich color and texture to the foreground. A vintage-style streetlamp stands prominently near the center-right, contributing to the timeless character of the street. Mature trees frame the background, their leaves glowing in the warm afternoon sunlight. Bicycles are visible along the edges of the buildings, reinforcing the urban yet leisurely feel. The sky is bright blue with scattered clouds, and soft sun flares enter the frame from the left, enhancing the scene's inviting, peaceful mood.'
  PYTHONPATH=${WORLDFOUNDRY_SOURCE_ROOT} python -m worldfoundry.studio.workspace_job infer \\
    --model-id astra \\
    --input-path worldfoundry/data/test_cases/astra/condition_images/garden_1.png \\
    --interactions forward_left \\
    --prompt "\$ASTRA_PROMPT" \\
    --fps 20 \\
    --call-json '{"frames_per_generation":8,"total_frames_to_generate":24,"num_inference_steps":50,"start_frame":0,"initial_condition_frames":1,"modality_type":"sekai","return_dict":true}' \\
    --output-dir ${OUTPUT_DIR} \\
    --device cuda
EOF
elif [[ "$MODEL_ID" == "solaris" ]]; then
  cat <<EOF

Suggested inference command:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
  PYTHONPATH=${WORLDFOUNDRY_SOURCE_ROOT} python -m worldfoundry.studio.workspace_job infer \\
    --model-id solaris \\
    --call-json '{"eval_types":"translation","eval_num_samples":8,"num_workers":8,"return_dict":true}' \\
    --output-dir ${OUTPUT_DIR} \\
    --device cuda
EOF
elif [[ "$MODEL_ID" == "hunyuanworld-mirror" || "$MODEL_ID" == "hunyuan-mirror" ]]; then
  cat <<EOF

Suggested inference command:
  PYTHONPATH=${WORLDFOUNDRY_SOURCE_ROOT} python -m worldfoundry.studio.workspace_job infer \\
    --model-id hunyuanworld-mirror \\
    --input-path worldfoundry/data/test_cases/vggt/examples/kitchen/images \\
    --call-json '{"save_pointmap":true,"save_depth":true,"save_normal":true,"save_gs":true,"save_rendered":false,"save_colmap":true,"return_dict":true}' \\
    --output-dir ${OUTPUT_DIR} \\
    --device cuda
EOF
else
  cat <<EOF

Suggested inference command:
  PYTHONPATH=${WORLDFOUNDRY_SOURCE_ROOT} python -m worldfoundry.studio.workspace_job infer \\
    --model-id ${INFER_MODEL_ID} \\
    --output-dir ${OUTPUT_DIR} \\
    --device cuda

For Matrix-Game-style navigation demos:
  bash scripts/inference/test_nav_video_gen.sh ${INFER_MODEL_ID} --output-dir ${OUTPUT_DIR}
EOF
fi

if [[ "$download_status" != "0" ]]; then
  if [[ "$DOWNLOAD" != "1" ]]; then
    cat <<EOF

Local asset check completed. Some required assets are missing from:
  ${CACHE_DIR}

Run again with --download after confirming storage, license, and gated-access
requirements. If the weights already exist on shared storage, run
scripts/setup/link_hf_checkpoints.sh to create no-copy HF/HFD aliases.
EOF
    exit 0
  fi
  cat >&2 <<EOF

Model setup is not fully ready yet.
Use --download to fetch public Hugging Face checkpoints into HF_HUB_CACHE, or
run scripts/setup/link_hf_checkpoints.sh to reuse existing local weights.
EOF
  exit 1
fi
