#!/usr/bin/env bash
# Vendored from https://github.com/JinPLu/WRBench (Apache-2.0),
# scripts/eval/score_runtime_v2_d3d6.sh at 629595dc60ec08a29711af0377280c4ac9dd40bc.
# The in-tree wrbench package sits directly under WRBENCH_ROOT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRBENCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCORING_DIR="${WRBENCH_ROOT}/wrbench/eval/scoring"
REPO_ROOT="${WRBENCH_ROOT}"
export PYTHONPATH="${WRBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

require_env() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "[error] required environment variable ${name} is not set" >&2
    exit 2
  fi
  printf '%s' "$value"
}

export FORCE_QWENVL_VIDEO_READER="$(require_env FORCE_QWENVL_VIDEO_READER)"
export WORLD_STATE_VIDEO_BACKEND="$(require_env WORLD_STATE_VIDEO_BACKEND)"

PY_SCORER="$(require_env PY_SCORER)"
PY_HELPER="$(require_env PY_HELPER)"

MANIFEST="$(require_env MANIFEST)"
OUT_DIR="$(require_env OUT_DIR)"
QWEN35_MODEL="$(require_env QWEN35_MODEL)"
QWEN3VL_MODEL="$(require_env QWEN3VL_MODEL)"

SCORER_PROFILE="$(require_env SCORER_PROFILE)"
RUN_TAG="$(require_env RUN_TAG)"
PROMPT_MODE="$(require_env PROMPT_MODE)"
TASK_CONTEXT_MODE="$(require_env TASK_CONTEXT_MODE)"
SOURCE_SCORES="${SOURCE_SCORES:-}"
NUM_SHARDS="$(require_env NUM_SHARDS)"
SHARD_IDS="$(require_env SHARD_IDS)"
CUDA_DEVICES_CSV="$(require_env CUDA_DEVICES_CSV)"
FPS="$(require_env FPS)"
PROGRESS_EVERY="$(require_env PROGRESS_EVERY)"
SKIP_EXISTING="$(require_env SKIP_EXISTING)"
QWEN35_VLM_NAME="$(require_env QWEN35_VLM_NAME)"
QWEN35_LOADER_FAMILY="$(require_env QWEN35_LOADER_FAMILY)"
QWEN35_DTYPE="$(require_env QWEN35_DTYPE)"
QWEN35_ATTN_IMPLEMENTATION="$(require_env QWEN35_ATTN_IMPLEMENTATION)"
QWEN35_NUM_SAMPLES="$(require_env QWEN35_NUM_SAMPLES)"
QWEN35_MAX_VIDEOS="$(require_env QWEN35_MAX_VIDEOS)"
QWEN35_EVIDENCE_CONTEXT_MODE="$(require_env QWEN35_EVIDENCE_CONTEXT_MODE)"
QWEN3VL_DTYPE="$(require_env QWEN3VL_DTYPE)"
QWEN3VL_ATTN_IMPLEMENTATION="$(require_env QWEN3VL_ATTN_IMPLEMENTATION)"
QWEN3VL_MAX_NEW_TOKENS="$(require_env QWEN3VL_MAX_NEW_TOKENS)"
QWEN3VL_MAX_VIDEOS="$(require_env QWEN3VL_MAX_VIDEOS)"
QWEN3VL_BINARY_PROMPT_SCHEMA="$(require_env QWEN3VL_BINARY_PROMPT_SCHEMA)"
QWEN3VL_RESCUE_PROMPT_SCHEMA="$(require_env QWEN3VL_RESCUE_PROMPT_SCHEMA)"
if [[ "$SKIP_EXISTING" != "0" && "$SKIP_EXISTING" != "1" ]]; then
  echo "[error] SKIP_EXISTING must be 0 or 1" >&2
  exit 2
fi

case "$SCORER_PROFILE" in
  wrbench_default)
    qwen35_runner_prefix=(
      "${SCORING_DIR}/run_qwen35_p25_d3d4_slot_parse_overlay.py"
      --repo-root "$REPO_ROOT"
      --
    )
    ;;
  ablation_manifest_metadata)
    qwen35_runner_prefix=("${SCORING_DIR}/run_local_qwen35_probe_logprob_scorer.py")
    ;;
  custom)
    qwen35_runner_prefix=("${SCORING_DIR}/run_local_qwen35_probe_logprob_scorer.py")
    ;;
  *)
    echo "[error] unsupported SCORER_PROFILE: $SCORER_PROFILE" >&2
    echo "[error] expected wrbench_default, ablation_manifest_metadata, or custom" >&2
    exit 2
    ;;
esac

QWEN35_OUT="${OUT_DIR}/qwen35_worldstate_scores"
BINARY_GATE_OUT="${OUT_DIR}/qwen3vl_oov_binary_gate"
RESCUE_OUT="${OUT_DIR}/qwen3vl_oov_guarded_rescue"
GATE_OVERLAY_OUT="${OUT_DIR}/qwen3vl_oov_gate_overlay"
FINAL_OUT="${OUT_DIR}/final_exports"
LOG_DIR="${OUT_DIR}/logs"

usage() {
  cat <<'EOF'
Usage:
  MANIFEST=/abs/path/manifest.json OUT_DIR=/abs/path/out \
  QWEN35_MODEL=/abs/path/Qwen3.5-9B QWEN3VL_MODEL=/abs/path/Qwen3-VL-8B-Instruct \
  bash scripts/eval/score_runtime_v2_d3d6.sh <stage>

Stages:
  preflight | qwen35 | merge_qwen35 | gate_binary | merge_binary |
  build_rescue | gate_rescue | merge_rescue | overlay_gate | export | all

Manifest contract:
  JSON list. Each row must contain video_id, path or video_path, and
  world_state_prompt or prompt_text. Optional benchmark metadata fields are
  passed through when present.

Notes:
  Required environment also includes PY_SCORER, PY_HELPER, SCORER_PROFILE,
  RUN_TAG, PROMPT_MODE, TASK_CONTEXT_MODE, NUM_SHARDS, SHARD_IDS,
  CUDA_DEVICES_CSV, FPS, PROGRESS_EVERY, SKIP_EXISTING,
  FORCE_QWENVL_VIDEO_READER, WORLD_STATE_VIDEO_BACKEND, QWEN35_* scorer
  hyperparameters, and QWEN3VL_* evidence hyperparameters.

  Example scorer profile value:
  SCORER_PROFILE=wrbench_default:
  visible D3/D4 + returned D5/D6 probe logprob scoring with shared re-observation
  judgeability gate for returned-state metrics.
EOF
}

if [[ "$#" -lt 1 ]]; then
  echo "[error] stage argument is required" >&2
  usage >&2
  exit 2
fi
stage="$1"
case "$stage" in
  preflight|qwen35|merge_qwen35|gate_binary|merge_binary|build_rescue|gate_rescue|merge_rescue|overlay_gate|export|all)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "[error] unknown stage: $stage" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "$MANIFEST" ]]; then
  echo "[error] MANIFEST is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "[error] MANIFEST is not readable: $MANIFEST" >&2
  exit 2
fi

mkdir -p "$QWEN35_OUT" "$BINARY_GATE_OUT" "$RESCUE_OUT" "$GATE_OVERLAY_OUT" "$FINAL_OUT" "$LOG_DIR"

json_list_len() {
  "$PY_HELPER" -c 'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$1"
}

run_sharded() {
  local name="$1"
  local py="$2"
  shift 2
  IFS=',' read -ra GPUS <<< "$CUDA_DEVICES_CSV"
  if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "[error] CUDA_DEVICES_CSV must contain at least one device" >&2
    return 2
  fi
  local i=0
  local pids=()
  for sid in $SHARD_IDS; do
    local gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    echo "[launch] ${name} shard=${sid}/${NUM_SHARDS} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" "$py" "$@" \
      --num-shards "$NUM_SHARDS" \
      --shard-id "$sid" \
      --local-rank 0 \
      > "${LOG_DIR}/${name}_shard_${sid}.log" 2>&1 &
    pids+=("$!")
    i=$((i + 1))
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

preflight() {
  "$PY_HELPER" - "$MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(rows, list):
    raise SystemExit("manifest must be a JSON list")
seen = set()
missing = []
unreadable = []
for idx, row in enumerate(rows):
    if not isinstance(row, dict):
        raise SystemExit(f"manifest row {idx} is not an object")
    video_id = str(row.get("video_id") or "").strip()
    video_path = str(row.get("path") or row.get("video_path") or "").strip()
    prompt = str(row.get("world_state_prompt") or row.get("prompt_text") or "").strip()
    if not video_id or not video_path or not prompt:
        missing.append(idx)
    if video_id in seen:
        raise SystemExit(f"duplicate video_id: {video_id}")
    seen.add(video_id)
    if video_path and not Path(video_path).exists():
        unreadable.append(video_id or f"row-{idx}")
if missing:
    raise SystemExit(f"rows missing video_id/path/world_state_prompt: {missing[:10]}")
if unreadable:
    raise SystemExit(f"unreadable video paths: {unreadable[:10]}")
print(f"[preflight] manifest_records={len(rows)}")
PY
}

qwen35_args=(
  "${qwen35_runner_prefix[@]}"
  --experiment-id "$RUN_TAG"
  --manifest-path "$MANIFEST"
  --output-dir "$QWEN35_OUT"
  --model-path "$QWEN35_MODEL"
  --fps "$FPS"
  --max-videos "$QWEN35_MAX_VIDEOS"
  --progress-every "$PROGRESS_EVERY"
  --prompt-mode "$PROMPT_MODE"
  --task-context-mode "$TASK_CONTEXT_MODE"
  --vlm-name "$QWEN35_VLM_NAME"
  --loader-family "$QWEN35_LOADER_FAMILY"
  --dtype "$QWEN35_DTYPE"
  --attn-implementation "$QWEN35_ATTN_IMPLEMENTATION"
  --num-samples "$QWEN35_NUM_SAMPLES"
  --evidence-context-mode "$QWEN35_EVIDENCE_CONTEXT_MODE"
  --num-shards "$NUM_SHARDS"
  --shard-id "0"
  --local-rank "0"
)
if [[ -n "$SOURCE_SCORES" ]]; then
  qwen35_args+=(--source-scores "$SOURCE_SCORES")
fi
if [[ "$SKIP_EXISTING" == "0" ]]; then
  qwen35_args+=(--strict-manifest-contract)
else
  qwen35_args+=(--skip-existing)
fi

maybe_skip_existing=()
if [[ "$SKIP_EXISTING" == "1" ]]; then
  maybe_skip_existing=(--skip-existing)
fi

run_stage() {
  local selected="$1"
  case "$selected" in
    preflight)
      preflight
      ;;
    qwen35)
      if [[ -z "$QWEN35_MODEL" || ! -d "$QWEN35_MODEL" ]]; then
        echo "QWEN35_MODEL must point to a local model directory (configure eval.scorers.qwen35_model)" >&2
        exit 1
      fi
      run_sharded qwen35 "$PY_SCORER" "${qwen35_args[@]}"
      ;;
    merge_qwen35)
      "$PY_SCORER" "${qwen35_args[@]}" --merge-only
      ;;
    gate_binary)
      run_sharded qwen3vl_binary "$PY_SCORER" \
        "${SCORING_DIR}/run_local_qwen3vl_video_evidence.py" \
        --manifest-path "$MANIFEST" \
        --output-dir "$BINARY_GATE_OUT" \
        --model-path "$QWEN3VL_MODEL" \
        --prompt-schema "$QWEN3VL_BINARY_PROMPT_SCHEMA" \
        --fps "$FPS" \
        --max-videos "$QWEN3VL_MAX_VIDEOS" \
        --dtype "$QWEN3VL_DTYPE" \
        --attn-implementation "$QWEN3VL_ATTN_IMPLEMENTATION" \
        --max-new-tokens "$QWEN3VL_MAX_NEW_TOKENS" \
        "${maybe_skip_existing[@]}"
      ;;
    merge_binary)
      "$PY_SCORER" "${SCORING_DIR}/run_local_qwen3vl_video_evidence.py" \
        --manifest-path "$MANIFEST" \
        --output-dir "$BINARY_GATE_OUT" \
        --model-path "$QWEN3VL_MODEL" \
        --prompt-schema "$QWEN3VL_BINARY_PROMPT_SCHEMA" \
        --fps "$FPS" \
        --max-videos "$QWEN3VL_MAX_VIDEOS" \
        --dtype "$QWEN3VL_DTYPE" \
        --attn-implementation "$QWEN3VL_ATTN_IMPLEMENTATION" \
        --max-new-tokens "$QWEN3VL_MAX_NEW_TOKENS" \
        --num-shards "$NUM_SHARDS" \
        --shard-id "0" \
        --local-rank "0" \
        --merge-only
      ;;
    build_rescue)
      "$PY_HELPER" "${SCORING_DIR}/build_qwen3vl_binary_na_rescue_manifest.py" \
        --manifest-path "$MANIFEST" \
        --binary-evidence-jsonl "$BINARY_GATE_OUT/evidence.jsonl" \
        --out-manifest "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue.json" \
        --out-summary "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue_summary.json"
      ;;
    gate_rescue)
      local rescue_records
      rescue_records="$(json_list_len "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue.json")"
      if [[ "$rescue_records" == "0" ]]; then
        echo "[skip] qwen3vl_rescue: rescue manifest is empty"
      else
        run_sharded qwen3vl_rescue "$PY_SCORER" \
          "${SCORING_DIR}/run_local_qwen3vl_video_evidence.py" \
          --manifest-path "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue.json" \
          --output-dir "$RESCUE_OUT" \
          --model-path "$QWEN3VL_MODEL" \
          --prompt-schema "$QWEN3VL_RESCUE_PROMPT_SCHEMA" \
          --fps "$FPS" \
          --max-videos "$QWEN3VL_MAX_VIDEOS" \
          --dtype "$QWEN3VL_DTYPE" \
          --attn-implementation "$QWEN3VL_ATTN_IMPLEMENTATION" \
          --max-new-tokens "$QWEN3VL_MAX_NEW_TOKENS" \
          "${maybe_skip_existing[@]}"
      fi
      ;;
    merge_rescue)
      local rescue_records
      rescue_records="$(json_list_len "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue.json")"
      if [[ "$rescue_records" == "0" ]]; then
        : > "$RESCUE_OUT/evidence.jsonl"
        echo "[skip] qwen3vl_rescue merge: rescue manifest is empty"
      else
        "$PY_SCORER" "${SCORING_DIR}/run_local_qwen3vl_video_evidence.py" \
          --manifest-path "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue.json" \
          --output-dir "$RESCUE_OUT" \
          --model-path "$QWEN3VL_MODEL" \
          --prompt-schema "$QWEN3VL_RESCUE_PROMPT_SCHEMA" \
          --fps "$FPS" \
          --max-videos "$QWEN3VL_MAX_VIDEOS" \
          --dtype "$QWEN3VL_DTYPE" \
          --attn-implementation "$QWEN3VL_ATTN_IMPLEMENTATION" \
          --max-new-tokens "$QWEN3VL_MAX_NEW_TOKENS" \
          --num-shards "$NUM_SHARDS" \
          --shard-id "0" \
          --local-rank "0" \
          --merge-only
      fi
      ;;
    overlay_gate)
      "$PY_HELPER" "${SCORING_DIR}/merge_qwen3vl_rescue_evidence.py" \
        --manifest-path "$MANIFEST" \
        --baseline-evidence-jsonl "$BINARY_GATE_OUT/evidence.jsonl" \
        --rescue-evidence-jsonl "$RESCUE_OUT/evidence.jsonl" \
        --out-evidence-jsonl "$GATE_OVERLAY_OUT/evidence_qwen3vl_guarded_overlay.jsonl" \
        --out-summary-json "$GATE_OVERLAY_OUT/evidence_qwen3vl_guarded_overlay_summary.json"
      ;;
    export)
      "$PY_HELPER" "${SCORING_DIR}/export_runtime_v2_evidence_first.py" \
        --scores-v7 "$QWEN35_OUT/scores_v7_candidate_runtime_v2_probe_score_available.json" \
        --evidence-jsonl "$GATE_OVERLAY_OUT/evidence_qwen3vl_guarded_overlay.jsonl" \
        --manifest-path "$MANIFEST" \
        --out-dir "$FINAL_OUT"
      ;;
  esac
}

if [[ "$stage" == "all" ]]; then
  for s in preflight qwen35 merge_qwen35 gate_binary merge_binary build_rescue gate_rescue merge_rescue overlay_gate export; do
    run_stage "$s"
  done
else
  run_stage "$stage"
fi

echo "[metric] out_dir=$OUT_DIR"
