#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORLDFOUNDRY_HOME_DEFAULT="${XDG_CACHE_HOME:-$HOME/.cache}/worldfoundry"
WORLDFOUNDRY_HOME_VALUE="${WORLDFOUNDRY_HOME:-$WORLDFOUNDRY_HOME_DEFAULT}"
MODEL_ROOT_VALUE="${WORLDFOUNDRY_MODEL_DIR:-${WORLDFOUNDRY_HOME_VALUE}/models}"
CKPT_DIR="${WORLDFOUNDRY_CKPT_DIR:-${MODEL_ROOT_VALUE}/checkpoints}"
HFD_ROOT="${WORLDFOUNDRY_HFD_ROOT:-${CKPT_DIR}/hfd}"
HF_HOME_VALUE="${HF_HOME:-${WORLDFOUNDRY_HOME_VALUE}/huggingface}"
HF_HUB_CACHE_VALUE="${HF_HUB_CACHE:-${HF_HOME_VALUE}/hub}"
FORCE=0
HFD_ROOT_EXPLICIT=0
HFD_ROOT_FROM_ENV=0
REPO_SPECS=()

if [[ -n "${WORLDFOUNDRY_HFD_ROOT:-}" ]]; then
  HFD_ROOT_FROM_ENV=1
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/link_hf_checkpoints.sh [options]

Create no-copy checkpoint aliases for machines that already have model weights.
The script links each local checkpoint directory into both supported layouts:

  1. HFD-style alias:
     ${WORLDFOUNDRY_HFD_ROOT}/OWNER--REPO -> ${WORLDFOUNDRY_CKPT_DIR}/LOCAL_DIR

  2. Native Hugging Face cache alias:
     ${HF_HUB_CACHE}/models--OWNER--REPO/snapshots/<40-hex-local-revision> -> LOCAL_DIR
     ${HF_HUB_CACHE}/models--OWNER--REPO/refs/main = <40-hex-local-revision>

New users can ignore this script and let Hugging Face download normally.
Cluster users with shared checkpoints can run it to reuse existing weights
without copying large directories.

Options:
  --ckpt-dir PATH       Existing checkpoint root. Default: $WORLDFOUNDRY_CKPT_DIR.
  --hfd-root PATH       HFD alias root. Default: $WORLDFOUNDRY_HFD_ROOT or <ckpt-dir>/hfd.
  --hf-hub-cache PATH   Native Hugging Face hub cache root. Default: $HF_HUB_CACHE.
  --repo REPO=DIR       Add one repo mapping, for example
                        --repo THUDM/CogVideoX-5b-I2V=CogVideoX-5b-I2V.
                        DIR may be absolute or relative to --ckpt-dir.
  --force               Replace existing symlinks/files owned by this layout.
  --default-world       Link common WorldFoundry world/video checkpoints if present.
  -h, --help            Show this help.
EOF
}

add_default_world_repos() {
  REPO_SPECS+=(
    "THUDM/CogVideoX-2b=CogVideoX-2b"
    "THUDM/CogVideoX-5b=CogVideoX-5b"
    "THUDM/CogVideoX-5b-I2V=CogVideoX-5b-I2V"
    "THUDM/CogVideoX1.5-5B=CogVideoX1.5-5B"
    "THUDM/CogVideoX1.5-5B-I2V=CogVideoX1.5-5B-I2V"
    "Skywork/SkyReels-V3-R2V-14B=SkyReels-V3-R2V-14B"
    "Skywork/SkyReels-V2-DF-1.3B-540P=SkyReels-V2-DF-1.3B-540P"
    "Skywork/SkyReels-V2-T2V-14B-720P=SkyReels-V2-T2V-14B-720P"
    "Skywork/SkyReels-V2-I2V-1.3B-540P=SkyReels-V2-I2V-1.3B-540P"
    "Skywork/SkyReels-V2-I2V-14B-720P=SkyReels-V2-I2V-14B-720P"
    "Wan-AI/Wan2.1-T2V-1.3B=Wan2.1-T2V-1.3B"
    "Wan-AI/Wan2.1-T2V-14B=Wan2.1-T2V-14B"
    "Wan-AI/Wan2.1-I2V-14B-480P=Wan2.1-I2V-14B-480P"
    "Wan-AI/Wan2.1-I2V-14B-720P=Wan2.1-I2V-14B-720P"
    "Wan-AI/Wan2.1-VACE-14B=Wan2.1-VACE-14B"
    "Wan-AI/Wan2.1-VACE-1.3B-diffusers=Wan2.1-VACE-1.3B-diffusers"
    "FayeHongfeiZhang/DualCamCtrl=DualCamCtrl"
    "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera=Wan2.1-Fun-V1.1-1.3B-Control-Camera"
    "Wan-AI/Wan2.2-TI2V-5B=Wan2.2-TI2V-5B"
    "Wan-AI/Wan2.2-T2V-A14B=Wan2.2-T2V-A14B"
    "Wan-AI/Wan2.2-I2V-A14B=Wan2.2-I2V-A14B"
    "Wan-AI/Wan2.2-Lightning=Wan2.2-Lightning"
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers=Wan2.2-TI2V-5B-Diffusers"
    "Skywork/Matrix-Game-2.0=Matrix-Game-2.0"
    "Skywork/Matrix-Game-3.0=Matrix-Game-3.0"
    "tencent/HY-WorldPlay=HY-WorldPlay"
    "tencent/Hunyuan-GameCraft-1.0=Hunyuan-GameCraft-1.0"
    "tencent/HunyuanVideo-1.5=HunyuanVideo-1.5"
    "tencent/HunyuanWorld-Voyager=HunyuanWorld-Voyager"
    "tencent/HY-World-2.0=HY-World-2.0"
    "tencent/HunyuanWorld-1=HunyuanWorld-1"
    "tencent/HunyuanWorld-Mirror=HunyuanWorld-Mirror"
    "Qwen/Qwen2.5-VL-7B-Instruct=Qwen2.5-VL-7B-Instruct"
    "google/byt5-small=byt5-small"
    "google-t5/t5-11b=t5-11b"
    "black-forest-labs/FLUX.1-Redux-dev=FLUX.1-Redux-dev"
    "black-forest-labs/FLUX.1-dev=FLUX.1-dev"
    "black-forest-labs/FLUX.1-Fill-dev=FLUX.1-Fill-dev"
    "google/siglip-base-patch16-224=siglip-base-patch16-224"
    "stdstu123/Yume-I2V-540P=Yume-I2V-540P"
    "stdstu123/Yume-5B-720P=Yume-5B-720P"
    "Yuppie1204/NeoVerse=NeoVerse"
    "liguang0115/vmem=vmem"
    "liguang0115/cut3r=cut3r"
    "worldcam/worldcam=worldcam"
    "MeiGen-AI/Infinite-World=Infinite-World"
    "KwaiVGI/Astra=Astra"
    "EvanEternal/Astra=Astra"
    "WoW-world-model/WoW-1-Wan-14B-600k=WoW-1-Wan-14B-600k"
    "nyu-visionx/solaris=solaris"
    "nvidia/GEN3C-Cosmos-7B=GEN3C-Cosmos-7B"
    "nvidia/Cosmos-Tokenize1-CV8x8x8-720p=Cosmos-Tokenize1-CV8x8x8-720p"
    "nvidia/Cosmos-Predict2.5-2B=Cosmos-Predict2.5-2B"
    "nvidia/Cosmos-Predict2.5-14B=Cosmos-Predict2.5-14B"
    "nvidia/Cosmos-Reason1-7B=Cosmos-Reason1-7B"
    "nvidia/Cosmos-Reason2-2B=Cosmos-Reason2-2B"
    "nvidia/Cosmos-Transfer2.5-2B=Cosmos-Transfer2.5-2B"
    "nvidia/Cosmos3-Nano=Cosmos3-Nano"
    "nvidia/Cosmos3-Super=Cosmos3-Super"
    "Ruicheng/moge-vitl=moge-vitl"
    "KlingTeam/ReCamMaster-Wan2.1=ReCamMaster-Wan2.1"
    "inspatio/world=world"
    "depth-anything/DA3NESTED-GIANT-LARGE=DA3NESTED-GIANT-LARGE"
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1=DA3NESTED-GIANT-LARGE-1.1"
    "microsoft/Florence-2-large=Florence-2-large"
    "RaphaelLiu/Pusa-Wan2.2-V1=Pusa-Wan2.2-V1"
    "lightx2v/Wan2.2-Lightning=Wan2.2-Lightning"
    "facebook/VGGT-1B=VGGT-1B"
    "lch01/StreamVGGT=StreamVGGT"
    "imlixinyang/FlashWorld=FlashWorld"
    "depth-anything/Depth-Anything-V2-Large=Depth-Anything-V2-Large"
    "depth-anything/Depth-Anything-V2-Giant=Depth-Anything-V2-Giant"
    "stepfun-ai/stepvideo-t2v=stepvideo-t2v"
    "TencentARC/Open-MAGVIT2=Open-MAGVIT2"
    "TencentARC/MotionCtrl=MotionCtrl"
    "showlab/show-o=show-o"
    "guoyww/animatediff=animatediff"
    "guoyww/animatediff-motion-adapter-v1-5-2=animatediff-motion-adapter-v1-5-2"
    "stable-diffusion-v1-5/stable-diffusion-v1-5=stable-diffusion-v1-5"
    "cerspense/zeroscope_v2_576w=zeroscope_v2_576w"
    "cerspense/zeroscope_v2_XL=zeroscope_v2_XL"
    "hehao13/CameraCtrl=CameraCtrl"
    "hehao13/CameraCtrl_SVD_ckpts=CameraCtrl_SVD_ckpts"
    "nvidia/DreamDojo=DreamDojo"
    "GEAR-Dreams/DreamZero-DROID=DreamZero-DROID"
    "GEAR-Dreams/DreamZero-AgiBot=DreamZero-AgiBot"
    "open-gigaai/GigaBrain-0-3.5B-Base=GigaBrain-0-3.5B-Base"
    "open-gigaai/GigaBrain-0.1-3.5B-Base=GigaBrain-0.1-3.5B-Base"
    "google/paligemma-3b-pt-224=paligemma-3b-pt-224"
    "physical-intelligence/fast=fast"
    "nvidia/GR00T-N1.7-LIBERO=GR00T-N1.7-LIBERO"
    "robbyant/lingbot-world-base-cam=lingbot-world-base-cam"
    "robbyant/lingbot-world-base-act-preview=lingbot-world-base-act-preview"
    "robbyant/lingbot-world-fast=lingbot-world-fast"
    "robbyant/lingbot-world-v2-14b-causal-fast=lingbot-world-v2-14b-causal-fast"
    "robbyant/lingbot-video-dense-1.3b=lingbot-video-dense-1.3b"
    "robbyant/lingbot-video-moe-30b-a3b=lingbot-video-moe-30b-a3b"
    "robbyant/lingbot-va-base=lingbot-va-base"
    "robbyant/lingbot-va-posttrain-robotwin=lingbot-va-posttrain-robotwin"
    "robbyant/lingbot-va-posttrain-libero-long=lingbot-va-posttrain-libero-long"
    "brandonsmart/splatt3r_v1.0=splatt3r_v1.0"
    "dylanebert/pixelSplat=pixelSplat"
  )
}

while (($#)); do
  case "$1" in
    --ckpt-dir)
      CKPT_DIR="$2"
      shift 2
      ;;
    --hfd-root)
      HFD_ROOT="$2"
      HFD_ROOT_EXPLICIT=1
      shift 2
      ;;
    --hf-hub-cache)
      HF_HUB_CACHE_VALUE="$2"
      shift 2
      ;;
    --repo)
      REPO_SPECS+=("$2")
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --default-world)
      add_default_world_repos
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

if [[ "$HFD_ROOT_EXPLICIT" == "0" && "$HFD_ROOT_FROM_ENV" == "0" ]]; then
  HFD_ROOT="${CKPT_DIR}/hfd"
fi

if [[ "${#REPO_SPECS[@]}" == "0" ]]; then
  add_default_world_repos
fi

mkdir -p "$HFD_ROOT" "$HF_HUB_CACHE_VALUE"

repo_cache_name() {
  local repo_id="$1"
  printf 'models--%s\n' "${repo_id//\//--}"
}

hfd_name() {
  local repo_id="$1"
  printf '%s\n' "${repo_id//\//--}"
}

hf_local_revision() {
  local repo_id="$1"
  printf 'worldfoundry-local:%s' "$repo_id" | sha1sum | awk '{print $1}'
}

resolve_local_dir() {
  local repo_id="$1"
  local local_ref="$2"
  local hfd_ref
  hfd_ref="$(hfd_name "$repo_id")"
  local cache_ref
  cache_ref="$(repo_cache_name "$repo_id")"
  local repo_basename="${repo_id##*/}"
  local candidates=()
  if [[ "$local_ref" = /* ]]; then
    candidates+=("$local_ref")
  else
    candidates+=("${CKPT_DIR}/${local_ref}")
  fi
  candidates+=(
    "${MODEL_ROOT_VALUE}/${repo_id}"
    "${MODEL_ROOT_VALUE}/${hfd_ref}"
    "${MODEL_ROOT_VALUE}/${cache_ref}"
    "${CKPT_DIR}/${repo_basename}"
    "${CKPT_DIR}/${hfd_ref}"
    "${CKPT_DIR}/${cache_ref}"
    "${HFD_ROOT}/${hfd_ref}"
    "${HFD_ROOT}/${cache_ref}"
  )

  local candidate
  local seen=":"
  for candidate in "${candidates[@]}"; do
    if [[ "$seen" == *":${candidate}:"* ]]; then
      continue
    fi
    seen="${seen}${candidate}:"
    if [[ -d "$candidate" ]]; then
      realpath -m "$candidate"
      return 0
    fi
  done

  realpath -m "${candidates[0]}"
  return 1
}

replace_path_if_allowed() {
  local path="$1"
  if [[ -L "$path" || -f "$path" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      rm -f "$path"
    else
      return 1
    fi
  elif [[ -e "$path" ]]; then
    return 1
  fi
  return 0
}

link_one_repo() {
  local spec="$1"
  local repo_id="${spec%%=*}"
  local local_ref="${spec#*=}"
  if [[ "$repo_id" == "$spec" || -z "$repo_id" || -z "$local_ref" || "$repo_id" != */* ]]; then
    echo "Invalid --repo mapping: ${spec}" >&2
    return 2
  fi

  local local_dir
  if ! local_dir="$(resolve_local_dir "$repo_id" "$local_ref")"; then
    echo "skip ${repo_id}: local checkpoint not found at ${local_dir}"
    return 0
  fi

  local hfd_link="${HFD_ROOT}/$(hfd_name "$repo_id")"
  if [[ "$(realpath -m "$hfd_link")" == "$(realpath -m "$local_dir")" ]]; then
    echo "hfd source already present ${hfd_link}"
  elif replace_path_if_allowed "$hfd_link"; then
    ln -s "$local_dir" "$hfd_link"
    echo "linked hfd ${hfd_link} -> ${local_dir}"
  else
    echo "keep existing hfd ${hfd_link}"
  fi

  local repo_cache="${HF_HUB_CACHE_VALUE}/$(repo_cache_name "$repo_id")"
  local snapshot_dir="${repo_cache}/snapshots"
  local refs_dir="${repo_cache}/refs"
  local revision
  revision="$(hf_local_revision "$repo_id")"
  local snapshot_link="${snapshot_dir}/${revision}"
  local refs_main="${refs_dir}/main"
  mkdir -p "$snapshot_dir" "$refs_dir"

  if replace_path_if_allowed "$snapshot_link"; then
    ln -s "$local_dir" "$snapshot_link"
    echo "linked hf snapshot ${snapshot_link} -> ${local_dir}"
  else
    echo "keep existing hf snapshot ${snapshot_link}"
  fi

  local current_ref=""
  if [[ -f "$refs_main" || -L "$refs_main" ]]; then
    current_ref="$(tr -d '\r\n' <"$refs_main" || true)"
  fi
  if [[ "$FORCE" == "1" ]]; then
    rm -f "$refs_main"
    printf '%s' "$revision" >"$refs_main"
    echo "wrote hf ref ${refs_main}"
  elif [[ ! -e "$refs_main" ]]; then
    printf '%s' "$revision" >"$refs_main"
    echo "wrote hf ref ${refs_main}"
  elif [[ "$current_ref" == "local" || "$current_ref" == "worldfoundry-local" || -z "$current_ref" || ! -e "${snapshot_dir}/${current_ref}" ]]; then
    if [[ -f "$refs_main" || -L "$refs_main" ]]; then
      printf '%s' "$revision" >"$refs_main"
      echo "updated hf ref ${refs_main}"
    else
      echo "keep existing hf ref ${refs_main}"
    fi
  else
    echo "keep existing hf ref ${refs_main}"
  fi
}

echo "WorldFoundry checkpoint root: ${CKPT_DIR}"
echo "WorldFoundry HFD root: ${HFD_ROOT}"
echo "Hugging Face hub cache: ${HF_HUB_CACHE_VALUE}"

for spec in "${REPO_SPECS[@]}"; do
  link_one_repo "$spec"
done
