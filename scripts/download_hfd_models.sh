#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_CKPT_ROOT="${WORLDFOUNDRY_CKPT_DIR:-${HOME}/.cache/worldfoundry/checkpoints}"
DOWNLOAD_ROOT="${WORLDFOUNDRY_HFD_ROOT:-${DEFAULT_CKPT_ROOT}/hfd}"
MAX_PARALLEL=4
HF_USERNAME="${HF_USERNAME:-}"
HF_TOKEN="${HF_TOKEN:-}"
LIST_ONLY=0
PYTHON_BIN="${PYTHON:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    fi
fi

HF_DOWNLOAD_BACKEND="python"
declare -a HF_CLI=()
if command -v hf >/dev/null 2>&1; then
    HF_CLI=(hf download)
    HF_DOWNLOAD_BACKEND="cli"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI=(huggingface-cli download)
    HF_DOWNLOAD_BACKEND="cli"
fi

declare -a SELECTIONS=()
declare -a FAILED_REPOS=()
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_REPOS=()

# group|model_alias|component_key|repo_id|revision(optional)
MODEL_COMPONENTS=(
    "navigation|matrix-game-2|pretrained_model_path|Skywork/Matrix-Game-2.0"
    "navigation|hunyuan-game-craft|pretrained_model_path|tencent/Hunyuan-GameCraft-1.0"
    "navigation|hunyuan-world-voyager|pretrained_model_path|tencent/HunyuanWorld-Voyager"
    "navigation|hunyuan-world-voyager|represent_model_path|Ruicheng/moge-vitl"
    "navigation|astra|pretrained_model_path|EvanEternal/Astra"
    "navigation|astra|wan_model_path|Wan-AI/Wan2.1-T2V-1.3B"
    "navigation|yume-1.5|pretrained_model_path|stdstu123/Yume-5B-720P"
    "navigation|infinite-world|pretrained_model_path|MeiGen-AI/Infinite-World"
    "navigation|worldcam|pretrained_model_path|worldcam/worldcam"
    "navigation|worldcam|wan_model_path|Wan-AI/Wan2.1-T2V-1.3B"
    "navigation|vmem|pretrained_model_path|liguang0115/vmem"
    "navigation|vmem|surfel_model_path|liguang0115/cut3r"
    "navigation|neoverse|pretrained_model_path|Yuppie1204/NeoVerse"
    "navigation|lingbot-world|pretrained_model_path|robbyant/lingbot-world-base-cam"
    "navigation|lingbot-world-v2|pretrained_model_path|robbyant/lingbot-world-v2-14b-causal-fast"
    "video|wan-2.2|pretrained_model_path|Wan-AI/Wan2.2-TI2V-5B"
    "video|wow|pretrained_model_path|WoW-world-model/WoW-1-Wan-14B-600k"
    "video|cosmos-predict-2.5|pretrained_model_path|nvidia/Cosmos-Predict2.5-2B"
    "video|cosmos-predict-2.5|text_encoder_model_path|nvidia/Cosmos-Reason1-7B"
    "video|cosmos-predict-2.5|vae_model_path|Wan-AI/Wan2.1-T2V-1.3B"
    "video|cosmos3|pretrained_model_path|nvidia/Cosmos3-Nano|411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
    "video|cosmos3-super|pretrained_model_path|nvidia/Cosmos3-Super|e0262be9d8f7586bc24c069a2aed2b665bdff266"
    "video|recammaster|pretrained_model_path|KlingTeam/ReCamMaster-Wan2.1"
    "video|recammaster|wan_model_path|Wan-AI/Wan2.1-T2V-1.3B"
    "video|inspatio-world|pretrained_model_path|inspatio/world"
    "video|inspatio-world|wan_model_path|Wan-AI/Wan2.1-T2V-1.3B"
    "video|inspatio-world|da3_model_path|depth-anything/DA3NESTED-GIANT-LARGE"
    "video|inspatio-world|florence_model_path|microsoft/Florence-2-large"
    "video|gen3c|pretrained_model_path|nvidia/GEN3C-Cosmos-7B"
    "video|gen3c|tokenizer_model_path|nvidia/Cosmos-Tokenize1-CV8x8x8-720p"
    "video|gen3c|text_encoder_model_path|google-t5/t5-11b"
    "video|gen3c|moge_pretrained|Ruicheng/moge-vitl"
    "video|pusa-vidgen|pretrained_model_path|RaphaelLiu/Pusa-Wan2.2-V1"
    "video|pusa-vidgen|wan_model_path|Wan-AI/Wan2.2-T2V-A14B"
    "video|pusa-vidgen|lightx2v_path|lightx2v/Wan2.2-Lightning"
    "video|lingbot-video|dense_model_path|robbyant/lingbot-video-dense-1.3b"
    "video|lingbot-video|moe_model_path|robbyant/lingbot-video-moe-30b-a3b"
    "3d|hy-world-2p0|pretrained_model_path|tencent/HY-World-2.0"
    "3d|depth-anything-v2|pretrained_model_path|depth-anything/Depth-Anything-V2-Large"
    "3d|vggt|representation_path|facebook/VGGT-1B"
    "3d|infinite-vggt|pretrained_model_path|lch01/StreamVGGT"
    "3d|flashworld|pretrained_model_path|imlixinyang/FlashWorld"
    "3d|flashworld|wan_model_path|Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    "3d|cut3r|representation_path|liguang0115/cut3r"
    "video-official|step-video-t2v|pretrained_model_path|stepfun-ai/stepvideo-t2v"
    "video-official|open-magvit2|pretrained_model_path|TencentARC/Open-MAGVIT2"
    "video-official|show-o|pretrained_model_path|showlab/show-o"
    "video-official|animatediff|pretrained_model_path|guoyww/animatediff"
    "video-official|animatediff|motion_adapter_path|guoyww/animatediff-motion-adapter-v1-5-2"
    "video-official|animatediff|sd15_path|stable-diffusion-v1-5/stable-diffusion-v1-5"
    "video-official|zeroscope|pretrained_model_path|cerspense/zeroscope_v2_576w"
    "video-official|zeroscope|xl_model_path|cerspense/zeroscope_v2_XL"
    "world-official|cameractrl|pretrained_model_path|hehao13/CameraCtrl"
    "world-official|cameractrl|svd_checkpoint_path|hehao13/CameraCtrl_SVD_ckpts"
    "world-official|cameractrl|sd15_path|stable-diffusion-v1-5/stable-diffusion-v1-5"
    "world-official|motionctrl|pretrained_model_path|TencentARC/MotionCtrl"
    "world-official|dreamdojo|pretrained_model_path|nvidia/DreamDojo"
    "world-official|dreamzero|pretrained_model_path|GEAR-Dreams/DreamZero-DROID"
    "world-official|dreamzero|agibot_checkpoint_path|GEAR-Dreams/DreamZero-AgiBot"
    "embodied-action|giga-brain-0|pretrained_model_path|open-gigaai/GigaBrain-0-3.5B-Base"
    "embodied-action|giga-brain-0|pretrained_model_0p1_path|open-gigaai/GigaBrain-0.1-3.5B-Base"
    "embodied-action|giga-brain-0|tokenizer_model_path|google/paligemma-3b-pt-224"
    "embodied-action|giga-brain-0|fast_tokenizer_path|physical-intelligence/fast"
    "embodied-action|gr00t|policy_checkpoint_path|nvidia/GR00T-N1.7-LIBERO"
    "embodied-action|gr00t|reasoner_model_path|nvidia/Cosmos-Reason2-2B"
    "embodied-action|lingbot-va|base_checkpoint_path|robbyant/lingbot-va-base"
    "embodied-action|lingbot-va|robotwin_checkpoint_path|robbyant/lingbot-va-posttrain-robotwin"
    "embodied-action|lingbot-va|libero_long_checkpoint_path|robbyant/lingbot-va-posttrain-libero-long"
    "embodied-action|cogact|llm_backbone_path|meta-llama/Llama-2-7b-hf"
    "embodied-action|cogact|small_checkpoint_path|CogACT/CogACT-Small"
    "embodied-action|cogact|base_checkpoint_path|CogACT/CogACT-Base"
    "embodied-action|cogact|large_checkpoint_path|CogACT/CogACT-Large"
    "embodied-action|db-cogact|libero_checkpoint_path|Dexmal/libero-db-cogact"
    "embodied-action|db-cogact|calvin_checkpoint_path|Dexmal/calvin-db-cogact"
    "embodied-action|db-cogact|simpler_checkpoint_path|Dexmal/simpler-db-cogact"
    "embodied-action|db-cogact|single_arm_checkpoint_path|Dexmal/Dexbotic-CogACT-SArm"
    "embodied-action|db-cogact|dual_arm_checkpoint_path|Dexmal/Dexbotic-CogACT-HArm"
    "embodied-action|vlanext|checkpoint_collection_path|DravenALG/VLANeXt"
    "embodied-action|molmobot|droid_checkpoint_path|allenai/MolmoBot-DROID"
    "embodied-action|molmobot|img_droid_checkpoint_path|allenai/MolmoBot-Img-DROID"
    "embodied-action|molmobot|rby1_multitask_checkpoint_path|allenai/MolmoBot-RBY1Multitask"
    "embodied-action|molmobot|pi0_droid_checkpoint_path|allenai/MolmoBot-Pi0-DROID"
    "embodied-action|mme-vla|perceptual_framesamp_modul_path|Yinpei/perceptual-framesamp-modul"
    "embodied-action|mme-vla|mme_vla_suite_path|Yinpei/mme_vla_suite"
    "embodied-action|mme-vla|pi05_baseline_path|Yinpei/pi05_baseline"
    "embodied-action|mme-vla|vlm_subgoal_predictor_path|Yinpei/vlm_subgoal_predictor"
    "3d-official|splatt3r|pretrained_model_path|brandonsmart/splatt3r_v1.0"
    "3d-official|pixelsplat|pretrained_model_path|dylanebert/pixelSplat"
)

usage() {
    cat <<EOF
Usage:
  bash scripts/download_hfd_models.sh [options] [selection...]

Description:
  Download the repo's commonly used world models in parallel via Hugging Face Hub.
  Default download root: ${DOWNLOAD_ROOT}

Selections:
  all                 Download everything (default)
  navigation          Matrix-Game-2, Hunyuan-GameCraft, Hunyuan-World-Voyager, Astra, YUME-1.5, Infinite-World, WorldCam, VMem, LingBot-World/V2
                      NeoVerse
  video               Wan-2.2, WoW, Cosmos-Predict-2.5, Cosmos3 Nano/Super, ReCamMaster, InSpatio-World, GEN3C, Pusa VidGen, LingBot-Video
  3d                  HY-World-2.0, Depth-Anything-V2, VGGT, Infinite-VGGT, FlashWorld, CUT3R
  video-official      Step-Video-T2V, Open-MAGVIT2, Show-o, AnimateDiff, ZeroScope
  world-official      CameraCtrl, MotionCtrl, DreamDojo, DreamZero
  embodied-action     GigaBrain-0, GR00T, LingBot-VA, CogACT, DB-CogACT, VLANeXt, MolmoBot, MME-VLA
  3d-official         Splatt3R, pixelSplat
  matrix-game-2
  hunyuan-game-craft
  hunyuan-world-voyager
  astra
  yume-1.5
  infinite-world
  worldcam
  vmem
  neoverse
  lingbot-world
  lingbot-world-v2
  wan-2.2
  wow
  cosmos-predict-2.5
  cosmos3             Cosmos3 Nano (default)
  cosmos3-super       Cosmos3 Super
  recammaster
  inspatio-world
  gen3c
  pusa-vidgen
  lingbot-video
  hy-world-2p0
  depth-anything-v2
  vggt
  infinite-vggt
  flashworld
  cut3r
  dreamdojo
  dreamzero
  giga-brain-0
  gr00t
  lingbot-va
  cogact
  db-cogact
  vlanext
  molmobot
  mme-vla

Options:
  --download-root PATH   Override download root (default: ${DOWNLOAD_ROOT})
  --parallel N           Max concurrent repos to download (default: 4)
  --hf_username USER     Hugging Face username for gated repos
  --hf_token TOKEN       Deprecated; prefer HF_TOKEN env var so tokens do not appear in process listings
  --list                 Print model/component mapping and exit
  -h, --help             Show this help

Notes:
  - The script uses `hf download` when available and falls back to Python `huggingface_hub.snapshot_download`.
  - Cosmos3 Nano/Super generator checkpoints are public. Its default safety checker separately
    downloads gated nvidia/Cosmos-1.0-Guardrail plus public Qwen/Qwen3Guard-Gen-0.6B at first use,
    so fresh machines need an approved HF_TOKEN unless safety is explicitly disabled.
  - Some older Cosmos and Llama/CogACT repos may also require --hf_username plus HF_TOKEN.
  - CogACT needs access to the gated meta-llama/Llama-2-7b-hf backbone in addition to CogACT checkpoints.
  - A manifest is written to <download-root>/model_paths.tsv after selection.
EOF
}

sanitize_repo_id() {
    local repo_id="$1"
    echo "${repo_id//\//--}"
}

repo_dir() {
    local repo_id="$1"
    echo "${DOWNLOAD_ROOT}/$(sanitize_repo_id "$repo_id")"
}

repo_log() {
    local repo_id="$1"
    echo "${DOWNLOAD_ROOT}/logs/$(sanitize_repo_id "$repo_id").log"
}

has_selection() {
    local needle="$1"
    local item
    for item in "${SELECTIONS[@]}"; do
        if [[ "$item" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

is_selected_record() {
    local group="$1"
    local model_alias="$2"

    if ((${#SELECTIONS[@]} == 0)); then
        return 0
    fi

    if has_selection "all" || has_selection "$group" || has_selection "$model_alias"; then
        return 0
    fi

    if [[ "$model_alias" == "cosmos3" ]] && has_selection "cosmos3-nano"; then
        return 0
    fi

    return 1
}

print_model_mapping() {
    local record group model_alias component_key repo_id revision
    printf "group\tmodel\tcomponent_key\trepo_id\trevision\tlocal_dir\n"
    for record in "${MODEL_COMPONENTS[@]}"; do
        IFS='|' read -r group model_alias component_key repo_id revision <<< "$record"
        if ! is_selected_record "$group" "$model_alias"; then
            continue
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$group" \
            "$model_alias" \
            "$component_key" \
            "$repo_id" \
            "${revision:-}" \
            "$(repo_dir "$repo_id")"
    done
}

write_manifest() {
    local manifest_path="${DOWNLOAD_ROOT}/model_paths.tsv"
    local record group model_alias component_key repo_id revision

    mkdir -p "${DOWNLOAD_ROOT}"
    {
        printf "group\tmodel\tcomponent_key\trepo_id\trevision\tlocal_dir\n"
        for record in "${MODEL_COMPONENTS[@]}"; do
            IFS='|' read -r group model_alias component_key repo_id revision <<< "$record"
            if is_selected_record "$group" "$model_alias"; then
                printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
                    "$group" \
                    "$model_alias" \
                    "$component_key" \
                    "$repo_id" \
                    "${revision:-}" \
                    "$(repo_dir "$repo_id")"
            fi
        done
    } > "${manifest_path}"
    echo "${manifest_path}"
}

record_download_revision() {
    local repo_id="$1"
    local revision="$2"
    local local_dir="$3"
    [[ -n "$revision" ]] || return 0

    mkdir -p "${local_dir}/.hfd"
    if [[ -n "${PYTHON_BIN}" ]]; then
        REPO_ID="$repo_id" REVISION="$revision" LOCAL_DIR="$local_dir" "${PYTHON_BIN}" <<'PY'
import json
import os
from pathlib import Path

metadata_path = Path(os.environ["LOCAL_DIR"]) / ".hfd" / "repo_metadata.json"
try:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError, OSError):
    payload = {}
payload.update(
    {
        "id": os.environ["REPO_ID"],
        "revision": os.environ["REVISION"],
        "sha": os.environ["REVISION"],
        "worldfoundry_pinned_download": True,
    }
)
temporary_path = metadata_path.with_suffix(".json.tmp")
temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary_path.replace(metadata_path)
PY
    else
        printf '{"id":"%s","revision":"%s","sha":"%s","worldfoundry_pinned_download":true}\n' \
            "$repo_id" "$revision" "$revision" >"${local_dir}/.hfd/repo_metadata.json"
    fi
}

download_repo() {
    local repo_id="$1"
    local revision="${2:-}"
    local local_dir
    local log_path
    local_dir="$(repo_dir "$repo_id")"
    log_path="$(repo_log "$repo_id")"

    mkdir -p "${DOWNLOAD_ROOT}/logs"

    if [[ -d "${local_dir}" ]] && find "${local_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        echo "[resume] ${repo_id} at ${local_dir}"
    fi

    echo "[start] ${repo_id}${revision:+@${revision}} -> ${local_dir}"
    if [[ -n "${HF_TOKEN}" ]]; then
        export HF_TOKEN
    fi
    if [[ "${HF_DOWNLOAD_BACKEND}" == "cli" ]]; then
        local -a cmd=(
            "${HF_CLI[@]}"
            "${repo_id}"
            --repo-type model
            --local-dir "${local_dir}"
        )
        if [[ -n "$revision" ]]; then
            cmd+=(--revision "$revision")
        fi
        env -u all_proxy -u ALL_PROXY "${cmd[@]}" > "${log_path}" 2>&1
    else
        if [[ -z "${PYTHON_BIN}" ]]; then
            echo "No Hugging Face CLI found and PYTHON is unset." > "${log_path}"
            return 1
        fi
        REPO_ID="${repo_id}" REVISION="${revision}" LOCAL_DIR="${local_dir}" "${PYTHON_BIN}" <<'PY' > "${log_path}" 2>&1
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["REPO_ID"],
    repo_type="model",
    local_dir=os.environ["LOCAL_DIR"],
    revision=os.environ.get("REVISION") or None,
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
)
PY
    fi
    record_download_revision "$repo_id" "$revision" "$local_dir" >>"${log_path}" 2>&1
}

wait_for_oldest() {
    local pid="${ACTIVE_PIDS[0]}"
    local repo_id="${ACTIVE_REPOS[0]}"
    local log_path
    log_path="$(repo_log "$repo_id")"

    if wait "${pid}"; then
        echo "[done]  ${repo_id}"
    else
        echo "[fail]  ${repo_id} (see ${log_path})"
        FAILED_REPOS+=("${repo_id}")
    fi

    ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
    ACTIVE_REPOS=("${ACTIVE_REPOS[@]:1}")
}

while (($# > 0)); do
    case "$1" in
        --download-root)
            DOWNLOAD_ROOT="$2"
            shift 2
            ;;
        --parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --hf_username)
            HF_USERNAME="$2"
            shift 2
            ;;
        --hf_token)
            HF_TOKEN="$2"
            export HF_TOKEN
            shift 2
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            SELECTIONS+=("$1")
            shift
            ;;
    esac
done

declare -A UNIQUE_REPOS=()
declare -A REPO_REVISIONS=()
declare -a ORDERED_REPOS=()
declare -A SELECTED_MODELS=()

for record in "${MODEL_COMPONENTS[@]}"; do
    IFS='|' read -r group model_alias component_key repo_id revision <<< "$record"
    if ! is_selected_record "$group" "$model_alias"; then
        continue
    fi

    SELECTED_MODELS["${model_alias}"]=1
    if [[ -z "${UNIQUE_REPOS[${repo_id}]+x}" ]]; then
        UNIQUE_REPOS["${repo_id}"]=1
        REPO_REVISIONS["${repo_id}"]="${revision:-}"
        ORDERED_REPOS+=("${repo_id}")
    elif [[ -n "${revision:-}" && -n "${REPO_REVISIONS[${repo_id}]:-}" && "${revision}" != "${REPO_REVISIONS[${repo_id}]}" ]]; then
        echo "[error] Conflicting pinned revisions for ${repo_id}: ${REPO_REVISIONS[${repo_id}]} vs ${revision}" >&2
        exit 1
    elif [[ -n "${revision:-}" ]]; then
        REPO_REVISIONS["${repo_id}"]="$revision"
    fi
done

if ((${#ORDERED_REPOS[@]} == 0)); then
    echo "[error] No models matched the requested selection: ${SELECTIONS[*]:-<empty>}" >&2
    usage
    exit 1
fi

MANIFEST_PATH="$(write_manifest)"

if ((LIST_ONLY == 1)); then
    echo "Manifest      : ${MANIFEST_PATH}"
    print_model_mapping
    exit 0
fi

if [[ "${HF_DOWNLOAD_BACKEND}" == "python" ]]; then
    if [[ -z "${PYTHON_BIN}" ]] || ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import huggingface_hub  # noqa: F401
PY
    then
        echo "[error] Hugging Face CLI not found and Python package huggingface_hub is unavailable." >&2
        echo "        Install huggingface_hub[cli] or run this script from the WorldFoundry conda env." >&2
        exit 1
    fi
fi

echo "Download root : ${DOWNLOAD_ROOT}"
echo "Manifest      : ${MANIFEST_PATH}"
echo "Parallel jobs : ${MAX_PARALLEL}"
if [[ "${HF_DOWNLOAD_BACKEND}" == "cli" ]]; then
    echo "HF backend    : ${HF_CLI[*]}"
else
    echo "HF backend    : ${PYTHON_BIN} -c huggingface_hub.snapshot_download"
fi
echo "Models        : ${#SELECTED_MODELS[@]}"
echo "Unique repos  : ${#ORDERED_REPOS[@]}"

if [[ -z "${HF_USERNAME}" || -z "${HF_TOKEN}" ]]; then
    for repo_id in "${ORDERED_REPOS[@]}"; do
        if [[ "${repo_id}" == nvidia/Cosmos-* ]]; then
            echo "[note] Cosmos repos are selected. If access is gated, pass --hf_username and set HF_TOKEN."
            break
        fi
    done
fi

echo "Selected repos:"
for repo_id in "${ORDERED_REPOS[@]}"; do
    revision="${REPO_REVISIONS[${repo_id}]:-}"
    echo "  - ${repo_id}${revision:+@${revision}} -> $(repo_dir "${repo_id}")"
done

for repo_id in "${ORDERED_REPOS[@]}"; do
    download_repo "${repo_id}" "${REPO_REVISIONS[${repo_id}]:-}" &
    ACTIVE_PIDS+=("$!")
    ACTIVE_REPOS+=("${repo_id}")

    if ((${#ACTIVE_PIDS[@]} >= MAX_PARALLEL)); then
        wait_for_oldest
    fi
done

while ((${#ACTIVE_PIDS[@]} > 0)); do
    wait_for_oldest
done

if ((${#FAILED_REPOS[@]} > 0)); then
    echo "[error] ${#FAILED_REPOS[@]} repo downloads failed:" >&2
    for repo_id in "${FAILED_REPOS[@]}"; do
        echo "  - ${repo_id} (log: $(repo_log "${repo_id}"))" >&2
    done
    exit 1
fi

echo "All downloads finished successfully."
echo "You can inspect local paths in: ${MANIFEST_PATH}"
