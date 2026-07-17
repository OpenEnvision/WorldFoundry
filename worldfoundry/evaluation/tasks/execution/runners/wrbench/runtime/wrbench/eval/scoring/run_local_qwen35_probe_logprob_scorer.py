#!/usr/bin/env python3
"""Runtime V2 local Qwen3.5 yes/no probe-logprob scorer.

Unlike the direct numeric Runtime V2 scorer, this route asks compact binary
probes and derives continuous [0, 1] dimension scores from next-token Yes/No
probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence


try:
    from .prompts_v2_probe import (
        CAMERA_MOTION_CONTEXT_KEY,
        DEFAULT_PROMPT_MODE,
        PROMPT_MODE_P10_SHARED_OOV_JUDGEABILITY,
        PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED,
        PROBE_CATALOG_VERSION,
        RUNTIME_V2_PROBE_CATALOG,
        SUPPORTED_PROMPT_MODES,
        RuntimeV2Probe,
        active_probe_catalog,
        build_runtime_v2_probe_prompt,
        question_for_probe,
        validate_prompt_mode,
    )
    from .runtime_common import META_KEYS
except ImportError:
    from wrbench.eval.scoring.prompts_v2_probe import (
        CAMERA_MOTION_CONTEXT_KEY,
        DEFAULT_PROMPT_MODE,
        PROMPT_MODE_P10_SHARED_OOV_JUDGEABILITY,
        PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED,
        PROBE_CATALOG_VERSION,
        RUNTIME_V2_PROBE_CATALOG,
        SUPPORTED_PROMPT_MODES,
        RuntimeV2Probe,
        active_probe_catalog,
        build_runtime_v2_probe_prompt,
        question_for_probe,
        validate_prompt_mode,
    )
    from wrbench.eval.scoring.runtime_common import META_KEYS


SCHEMA_VERSION = "runtime_v2_probe_logprob_v2"
SCORE_EXPORT_POLICY = "d5d6_score_probes_exported_gate_probes_diagnostic_v1"
DEFAULT_MODEL_PATH = ""
DEFAULT_VLM_NAME = "local_qwen35_runtime_v2_probe_logprob"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"
DEFAULT_FPS = "2"
TASK_CONTEXT_MODE_NONE = "none"
TASK_CONTEXT_MODE_ALL_MANIFEST_METADATA = "all_manifest_metadata"
TASK_CONTEXT_MODE_VLM_HUMAN_P9 = "vlm_human_p9"
TASK_CONTEXT_MODE_CAMERA_MOTION = "camera_motion"
SUPPORTED_TASK_CONTEXT_MODES = {
    TASK_CONTEXT_MODE_NONE,
    TASK_CONTEXT_MODE_ALL_MANIFEST_METADATA,
    TASK_CONTEXT_MODE_VLM_HUMAN_P9,
    TASK_CONTEXT_MODE_CAMERA_MOTION,
}
VLM_HUMAN_P9_CONTEXT_KEYS = {
    "model",
    "variant_id",
    "family_id",
    "camera_type",
    "reasoning_tier",
    "event_tier",
    "event_tag",
    "event_delta",
    "oov_gap",
    "scenario_type",
    "protocol",
    "visibility_gap_level",
    "prompt_id",
}
MIN_NUMERIC_FPS = 2.0
GATE_THRESHOLD = 0.5
REBUILT_POOL_20260519_NAME = "runtime_v2_validation_pool_recount_20260519"
REBUILT_OUTPUT_SUFFIX_20260519 = "_rebuilt_pool_20260519"
REBUILT_POOL_20260519_MANIFEST_RECORDS = 261
REBUILT_POOL_20260519_PAIR_ALLOWLIST_RECORDS = 230
REBUILT_POOL_20260519_PAIR_ALLOWLIST_SHA256 = (
    "9701319ac53e7b4476dac053ed845f8e866c10396f4a379978fec616692e0d15"
)
YES_VARIANTS = ("Yes", " yes", " Yes", "yes", "YES")
NO_VARIANTS = ("No", " no", " No", "no", "NO")
ORDINAL_VARIANTS = ("1", "2", "3", "4", "5")

DIMENSION_SCORE_FIELDS = {
    "spatial_fidelity": "d3_spatial_in_view_score",
    "state_fidelity": "d4_state_in_view_score",
    "spatial_reasoning": "d5_spatial_oov_score",
    "state_reasoning": "d6_state_oov_score",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def filter_resume_rows(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Keep only rows compatible with the current score contract."""
    compatible: list[dict[str, Any]] = []
    for row in rows:
        if row.get("probe_status") != "ok":
            continue
        if row.get("runtime_v2_schema") != SCHEMA_VERSION:
            continue
        if row.get("runtime_v2_probe_catalog_version") != PROBE_CATALOG_VERSION:
            continue
        if row.get("runtime_v2_prompt_mode") != args.prompt_mode:
            continue
        if row.get("runtime_v2_task_context_mode", TASK_CONTEXT_MODE_ALL_MANIFEST_METADATA) != args.task_context_mode:
            continue
        if str(row.get("sampling_fps")) != str(args.fps):
            continue
        probe_results = row.get("vlm_probe_results")
        if not isinstance(probe_results, dict):
            continue
        expected_probe_ids = {probe.probe_id for probe in active_probe_catalog(args.prompt_mode)}
        if not expected_probe_ids.issubset(set(probe_results)):
            continue
        compatible.append(row)
    return compatible


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    write_jsonl(tmp_path, rows)
    tmp_path.replace(path)


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("sha256:"):
        return text.split(":", 1)[1]
    return text


def _jsonl_record_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def validate_rebuilt_pool_20260519_preflight(
    *,
    manifest: list[dict[str, Any]],
    manifest_path: Path,
    output_dir: Path,
    pair_allowlist_jsonl: Path,
    expected_allowlist_sha256: str | None = REBUILT_POOL_20260519_PAIR_ALLOWLIST_SHA256,
) -> dict[str, Any]:
    """Validate that a run is pinned to the 2026-05-19 rebuilt validation pool."""
    errors: list[str] = []
    manifest_records = len(manifest)
    allowlist_records = _jsonl_record_count(pair_allowlist_jsonl)
    allowlist_sha256 = sha256_file(pair_allowlist_jsonl)
    manifest_parent_ok = manifest_path.parent.name == REBUILT_POOL_20260519_NAME
    output_dir_suffix_ok = output_dir.name.endswith(REBUILT_OUTPUT_SUFFIX_20260519)

    if not manifest_parent_ok:
        errors.append(
            f"manifest_path must be under {REBUILT_POOL_20260519_NAME}: {manifest_path}"
        )
    if manifest_records != REBUILT_POOL_20260519_MANIFEST_RECORDS:
        errors.append(
            "manifest must contain "
            f"{REBUILT_POOL_20260519_MANIFEST_RECORDS} records, got {manifest_records}"
        )
    if allowlist_records != REBUILT_POOL_20260519_PAIR_ALLOWLIST_RECORDS:
        errors.append(
            "pair allowlist must contain "
            f"{REBUILT_POOL_20260519_PAIR_ALLOWLIST_RECORDS} records, got {allowlist_records}"
        )
    if _normalize_sha256(allowlist_sha256) != _normalize_sha256(expected_allowlist_sha256):
        errors.append(
            "pair allowlist sha256 mismatch: "
            f"expected {_normalize_sha256(expected_allowlist_sha256)}, got {allowlist_sha256}"
        )
    if not output_dir_suffix_ok:
        errors.append(
            f"output_dir must end with {REBUILT_OUTPUT_SUFFIX_20260519}: {output_dir}"
        )

    report = {
        "schema_version": "runtime_v2_rebuilt_pool_preflight_20260519_v1",
        "status": "PASS" if not errors else "FAIL",
        "pool_name": REBUILT_POOL_20260519_NAME,
        "manifest_path": str(manifest_path),
        "manifest_records": manifest_records,
        "expected_manifest_records": REBUILT_POOL_20260519_MANIFEST_RECORDS,
        "manifest_parent_ok": manifest_parent_ok,
        "pair_allowlist_jsonl": str(pair_allowlist_jsonl),
        "pair_allowlist_records": allowlist_records,
        "expected_pair_allowlist_records": REBUILT_POOL_20260519_PAIR_ALLOWLIST_RECORDS,
        "pair_allowlist_sha256": allowlist_sha256,
        "expected_pair_allowlist_sha256": _normalize_sha256(expected_allowlist_sha256),
        "output_dir": str(output_dir),
        "output_dir_suffix": REBUILT_OUTPUT_SUFFIX_20260519,
        "output_dir_suffix_ok": output_dir_suffix_ok,
        "errors": errors,
    }
    if errors:
        raise ValueError("; ".join(errors))
    return report


def manifest_video_path(item: dict[str, Any]) -> Path | None:
    raw_path = item.get("path") or item.get("video_path")
    if not raw_path:
        return None
    return Path(str(raw_path))


def validate_manifest_video_files(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    missing: list[dict[str, str]] = []
    for item in manifest:
        video_path = manifest_video_path(item)
        if video_path is None or not video_path.is_file():
            missing.append(
                {
                    "video_id": str(item.get("video_id") or ""),
                    "path": "" if video_path is None else str(video_path),
                }
            )
    report = {
        "schema_version": "runtime_v2_video_file_preflight_v1",
        "status": "PASS" if not missing else "FAIL",
        "manifest_records": len(manifest),
        "missing_video_count": len(missing),
        "missing_videos": missing[:20],
    }
    if missing:
        examples = ", ".join(
            f"{row['video_id']} -> {row['path'] or '<missing path field>'}"
            for row in missing[:5]
        )
        raise FileNotFoundError(
            "manifest video file preflight failed: "
            f"{len(missing)}/{len(manifest)} videos are missing; examples: {examples}"
        )
    return report


def stable_shard(video_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


def select_manifest_shard(manifest: list[dict[str, Any]], *, num_shards: int, shard_id: int) -> list[dict[str, Any]]:
    return [
        item
        for item in manifest
        if item.get("video_id") and stable_shard(str(item["video_id"]), num_shards) == shard_id
    ]


def id_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("video_id")): row for row in rows if row.get("video_id") is not None}


def load_evidence_by_video_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    evidence_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in rows:
        video_id = row.get("video_id")
        if video_id is None:
            continue
        key = str(video_id)
        if key in evidence_by_id:
            duplicate_ids.append(key)
            continue
        evidence_by_id[key] = row
    if duplicate_ids:
        raise ValueError(
            f"--evidence-jsonl contains duplicate video_id values: {sorted(set(duplicate_ids))[:5]}"
        )
    return evidence_by_id


def attach_evidence_context(
    item: dict[str, Any],
    *,
    evidence_by_id: dict[str, dict[str, Any]],
    evidence_context_mode: str | None,
) -> dict[str, Any]:
    if not evidence_by_id:
        return item
    video_id = str(item.get("video_id") or "")
    if video_id not in evidence_by_id:
        raise ValueError(f"--evidence-jsonl is missing evidence for video_id={video_id}")
    merged = dict(item)
    merged["_runtime_v2_evidence_context"] = evidence_by_id[video_id]
    merged["_runtime_v2_evidence_context_mode"] = evidence_context_mode
    return merged


def merge_item(source: dict[str, Any] | None, manifest_item: dict[str, Any]) -> dict[str, Any]:
    merged = dict(source or {})
    merged.update(manifest_item)
    return merged


def build_task_context(
    item: dict[str, Any],
    *,
    task_context_mode: str = TASK_CONTEXT_MODE_ALL_MANIFEST_METADATA,
) -> dict[str, Any]:
    if task_context_mode not in SUPPORTED_TASK_CONTEXT_MODES:
        raise ValueError(
            "task_context_mode must be one of: "
            + ", ".join(sorted(SUPPORTED_TASK_CONTEXT_MODES))
        )
    if task_context_mode == TASK_CONTEXT_MODE_NONE:
        return {}
    if task_context_mode == TASK_CONTEXT_MODE_CAMERA_MOTION:
        value = item.get("camera_motion") or item.get("camera_type")
        text = str(value or "").strip()
        if text and text.lower() not in {"none", "null"}:
            return {CAMERA_MOTION_CONTEXT_KEY: text}
        return {}
    keys = (
        META_KEYS
        if task_context_mode == TASK_CONTEXT_MODE_ALL_MANIFEST_METADATA
        else VLM_HUMAN_P9_CONTEXT_KEYS
    )
    context: dict[str, Any] = {}
    for key in keys:
        if key in {"video_id", "world_state_prompt"}:
            continue
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"none", "null"}:
            context[key] = text
    return context


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def binary_probability_from_logits(
    logits: Sequence[float] | Any,
    *,
    yes_token_ids: set[int],
    no_token_ids: set[int],
) -> float:
    """Return P(Yes | Yes or No) from selected next-token logits."""
    if not yes_token_ids:
        raise ValueError("yes_token_ids must not be empty")
    if not no_token_ids:
        raise ValueError("no_token_ids must not be empty")

    def at(index: int) -> float:
        value = logits[index]
        try:
            return float(value.item())
        except AttributeError:
            return float(value)

    yes_scores = [at(i) for i in yes_token_ids]
    no_scores = [at(i) for i in no_token_ids]
    yes_lse = _logsumexp(yes_scores)
    no_lse = _logsumexp(no_scores)
    denom = _logsumexp([yes_lse, no_lse])
    return round(math.exp(yes_lse - denom), 6)


def token_ids_for_variants(tokenizer: Any, variants: Iterable[str]) -> set[int]:
    token_ids: set[int] = set()
    for text in variants:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            token_ids.add(int(encoded[0]))
    if not token_ids:
        raise ValueError(f"no single-token variants found for {list(variants)}")
    return token_ids


def tokenizer_candidate_probe(tokenizer: Any) -> dict[str, Any]:
    """Record candidate-label tokenization for yes/no and ordinal adapters."""

    def probe_variants(variants: Iterable[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for text in variants:
            token_ids = [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
            rows.append(
                {
                    "text": text,
                    "token_ids": token_ids,
                    "token_count": len(token_ids),
                    "single_token": len(token_ids) == 1,
                }
            )
        return rows

    return {
        "schema_version": "runtime_v2_tokenizer_candidate_probe_v1",
        "yes_variants": probe_variants(YES_VARIANTS),
        "no_variants": probe_variants(NO_VARIANTS),
        "ordinal_1to5_variants": probe_variants(ORDINAL_VARIANTS),
    }


def _processor_chat_template(processor: Any) -> str:
    chat_template = getattr(processor, "chat_template", None)
    if isinstance(chat_template, dict) and "default" in chat_template:
        chat_template = chat_template["default"]
    return str(chat_template or "")


def chat_template_supports_variable(processor: Any, variable: str) -> bool:
    return variable in _processor_chat_template(processor)


def video_processor_valid_kwargs(processor: Any) -> set[str]:
    video_processor = getattr(processor, "video_processor", None)
    valid_kwargs = getattr(video_processor, "valid_kwargs", None)
    annotations = getattr(valid_kwargs, "__annotations__", None)
    return set(annotations or [])


def build_video_processor_kwargs(
    processor: Any,
    *,
    fps: str,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
) -> dict[str, Any]:
    videos_kwargs: dict[str, Any] = {}
    if fps != "full":
        videos_kwargs["fps"] = float(fps)
        videos_kwargs["do_sample_frames"] = True
    valid_kwargs = video_processor_valid_kwargs(processor)
    video_backend = os.environ["WORLD_STATE_VIDEO_BACKEND"].strip()
    if not video_backend:
        raise RuntimeError("WORLD_STATE_VIDEO_BACKEND is required")
    if video_backend and "video_backend" in valid_kwargs:
        videos_kwargs["video_backend"] = video_backend
    for key, value in {
        "min_pixels": video_min_pixels,
        "max_pixels": video_max_pixels,
    }.items():
        if value is None:
            continue
        if key not in valid_kwargs:
            video_processor = getattr(processor, "video_processor", None)
            raise ValueError(
                f"video processor {type(video_processor).__name__} does not support {key}"
            )
        videos_kwargs[key] = int(value)
    return videos_kwargs


def build_apply_chat_template_kwargs(
    processor: Any,
    *,
    fps: str,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
) -> dict[str, Any]:
    processor_kwargs: dict[str, Any] = {}
    videos_kwargs = build_video_processor_kwargs(
        processor,
        fps=fps,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
    )
    if videos_kwargs:
        processor_kwargs["videos_kwargs"] = videos_kwargs
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if processor_kwargs:
        kwargs["processor_kwargs"] = processor_kwargs
    if chat_template_supports_variable(processor, "enable_thinking"):
        kwargs["enable_thinking"] = False
    return kwargs


def derive_frames_used_from_processor_inputs(inputs: Any, processor: Any) -> int:
    try:
        grid = inputs.get("video_grid_thw") if hasattr(inputs, "get") else inputs["video_grid_thw"]
    except Exception:
        return 0
    if grid is None:
        return 0
    try:
        temporal_grid = grid[0, 0]
    except Exception:
        try:
            temporal_grid = grid[0][0]
        except Exception:
            return 0
    try:
        temporal_grid_value = int(temporal_grid.item())
    except Exception:
        try:
            temporal_grid_value = int(temporal_grid)
        except Exception:
            return 0
    temporal_patch_size = getattr(getattr(processor, "video_processor", None), "temporal_patch_size", 1)
    try:
        patch_size = int(temporal_patch_size)
    except Exception:
        patch_size = 1
    return max(0, temporal_grid_value * max(1, patch_size))


def _probe_meta(
    probe: RuntimeV2Probe,
    p_yes: float,
    status: str,
    *,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
) -> dict[str, Any]:
    return {
        "p_yes": round(float(p_yes), 6),
        "value": 1 if p_yes >= GATE_THRESHOLD else 0,
        "status": status,
        "dimension": probe.dimension,
        "role": probe.role,
        "polarity": probe.polarity,
        "gate_kind": probe.gate_kind,
        "question": question_for_probe(probe, prompt_mode),
    }


def dry_run_probe_results(
    video_id: str,
    *,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for probe in active_probe_catalog(prompt_mode):
        digest = hashlib.sha256(f"{video_id}|{probe.probe_id}".encode("utf-8")).digest()
        p_yes = round(0.05 + (digest[0] / 255.0) * 0.9, 6)
        results[probe.probe_id] = _probe_meta(probe, p_yes, "dry_run", prompt_mode=prompt_mode)
    return results


def _score_probe_value(probe: RuntimeV2Probe, p_yes: float) -> float:
    if probe.polarity == "negative":
        return 1.0 - p_yes
    return p_yes


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _probe_p_yes(probe_results: dict[str, dict[str, Any]], probe_id: str) -> float | None:
    value = (probe_results.get(probe_id) or {}).get("p_yes")
    if value is None:
        return None
    return float(value)


def _gate_for_dimension(
    probe_results: dict[str, dict[str, Any]],
    *,
    prefix: str,
) -> tuple[bool, str | None]:
    hidden = _probe_p_yes(probe_results, f"{prefix}_GATE_HIDDEN")
    returned = _probe_p_yes(probe_results, f"{prefix}_GATE_RETURN")
    unidentifiable = _probe_p_yes(probe_results, f"{prefix}_GATE_UNIDENTIFIABLE")
    if hidden is None or returned is None or unidentifiable is None:
        return False, "other"
    if hidden < GATE_THRESHOLD:
        return False, "no_oov"
    if returned < GATE_THRESHOLD:
        return False, "no_return"
    if unidentifiable >= GATE_THRESHOLD:
        return False, "unidentifiable"
    return True, None


def _shared_oov_gate(
    probe_results: dict[str, dict[str, Any]],
) -> tuple[bool, str | None]:
    ever_unjudgeable = _probe_p_yes(probe_results, "OOV_EVER_UNJUDGEABLE")
    return_judgeable = _probe_p_yes(probe_results, "OOV_RETURN_JUDGEABLE")
    if ever_unjudgeable is None or return_judgeable is None:
        return False, "other"
    if ever_unjudgeable < GATE_THRESHOLD:
        return False, "no_oov"
    if return_judgeable < GATE_THRESHOLD:
        return False, "no_return"
    return True, None


def _uses_shared_oov_gate(prompt_mode: str) -> bool:
    return any(probe.probe_id == "OOV_EVER_UNJUDGEABLE" for probe in active_probe_catalog(prompt_mode))


def aggregate_probe_results(
    probe_results: dict[str, dict[str, Any]],
    *,
    video_id: str,
    sampling_fps: str,
    frames_used: int,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
) -> dict[str, Any]:
    prompt_mode = validate_prompt_mode(prompt_mode)
    active_probes = active_probe_catalog(prompt_mode)
    dim_values: dict[str, list[float]] = {
        "spatial_fidelity": [],
        "state_fidelity": [],
        "spatial_reasoning": [],
        "state_reasoning": [],
    }
    raw_probe_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    missing_score_by_dimension: dict[str, list[str]] = {}

    for probe in active_probes:
        result = probe_results.get(probe.probe_id)
        if not result or result.get("p_yes") is None:
            missing.append(probe.probe_id)
            if probe.role == "score":
                missing_score_by_dimension.setdefault(probe.dimension, []).append(probe.probe_id)
            continue
        p_yes = float(result["p_yes"])
        raw_probe_rows.append(
            {
                "probe_id": probe.probe_id,
                "dimension": probe.dimension,
                "role": probe.role,
                "polarity": probe.polarity,
                "gate_kind": probe.gate_kind,
                "p_yes": round(p_yes, 6),
                "hard_answer": "Yes" if p_yes >= GATE_THRESHOLD else "No",
                "status": result.get("status", "unknown"),
                "question": question_for_probe(probe, prompt_mode),
            }
        )
        if probe.role == "score":
            dim_values[probe.dimension].append(_score_probe_value(probe, p_yes))

    complete_dimension_scores = {
        dimension: None if dimension in missing_score_by_dimension else _mean_or_none(values)
        for dimension, values in dim_values.items()
    }
    shared_gate_app: bool | None = None
    shared_gate_na: str | None = None
    if _uses_shared_oov_gate(prompt_mode):
        shared_gate_app, shared_gate_na = _shared_oov_gate(probe_results)
        d5_gate_app, d5_gate_na = shared_gate_app, shared_gate_na
        d6_gate_app, d6_gate_na = shared_gate_app, shared_gate_na
    else:
        d5_gate_app, d5_gate_na = _gate_for_dimension(probe_results, prefix="D5")
        d6_gate_app, d6_gate_na = _gate_for_dimension(probe_results, prefix="D6")
    d5_score_available = complete_dimension_scores["spatial_reasoning"] is not None
    d6_score_available = complete_dimension_scores["state_reasoning"] is not None
    d5_na = "missing_score_probe" if not d5_score_available else None
    d6_na = "missing_score_probe" if not d6_score_available else None
    return {
        "video_id": video_id,
        "probe_status": "ok" if not missing else "partial",
        "missing_probe_ids": missing,
        "score_missing_probe_ids_by_dimension": {
            dim: ids for dim, ids in missing_score_by_dimension.items() if ids
        },
        "video_status": "valid",
        "sampling_fps": sampling_fps,
        "frames_used": int(frames_used),
        "d3_spatial_in_view_score": complete_dimension_scores["spatial_fidelity"],
        "d4_state_in_view_score": complete_dimension_scores["state_fidelity"],
        "d5_spatial_oov_applicable": d5_score_available,
        "d5_spatial_oov_score": complete_dimension_scores["spatial_reasoning"] if d5_score_available else None,
        "d5_spatial_oov_na_reason": d5_na,
        "d5_spatial_oov_gate_applicable": d5_gate_app,
        "d5_spatial_oov_gate_na_reason": d5_gate_na,
        "d6_state_oov_applicable": d6_score_available,
        "d6_state_oov_score": complete_dimension_scores["state_reasoning"] if d6_score_available else None,
        "d6_state_oov_na_reason": d6_na,
        "d6_state_oov_gate_applicable": d6_gate_app,
        "d6_state_oov_gate_na_reason": d6_gate_na,
        "runtime_v2_shared_oov_gate_applicable": shared_gate_app,
        "runtime_v2_shared_oov_gate_na_reason": shared_gate_na,
        "vlm_probe_results": probe_results,
        "runtime_v2_probe_rows": raw_probe_rows,
        "runtime_v2_schema": SCHEMA_VERSION,
        "runtime_v2_probe_catalog_version": PROBE_CATALOG_VERSION,
        "runtime_v2_prompt_mode": prompt_mode,
        "runtime_v2_score_export_policy": SCORE_EXPORT_POLICY,
    }


def export_v7_style_probe_record(
    row: dict[str, Any],
    *,
    export_policy: str = "legacy",
) -> dict[str, Any]:
    prompt_mode = row.get("runtime_v2_prompt_mode") or DEFAULT_PROMPT_MODE
    if export_policy not in {"legacy", "score_available", "gate_masked"}:
        raise ValueError("export_policy must be one of: legacy, score_available, gate_masked")

    def export_oov_applicable(
        *,
        score_key: str,
        legacy_app_key: str,
        gate_app_key: str,
    ) -> bool:
        if row.get(score_key) is None:
            return False
        if export_policy == "score_available":
            return True
        if export_policy == "gate_masked":
            return bool(row.get(gate_app_key))
        if _uses_shared_oov_gate(prompt_mode):
            return bool(row.get(gate_app_key))
        return bool(row.get(legacy_app_key))

    d5_app = export_oov_applicable(
        score_key="d5_spatial_oov_score",
        legacy_app_key="d5_spatial_oov_applicable",
        gate_app_key="d5_spatial_oov_gate_applicable",
    )
    d6_app = export_oov_applicable(
        score_key="d6_state_oov_score",
        legacy_app_key="d6_state_oov_applicable",
        gate_app_key="d6_state_oov_gate_applicable",
    )
    return {
        "video_id": row.get("video_id"),
        "path": row.get("path") or row.get("video_path"),
        "model": row.get("model"),
        "variant_id": row.get("variant_id"),
        "world_state_prompt": row.get("world_state_prompt") or row.get("prompt_text"),
        "vlm_version": DEFAULT_PROMPT_MODE,
        "vlm_name": row.get("vlm_name") or DEFAULT_VLM_NAME,
        "vlm_spatial_fidelity": row.get("d3_spatial_in_view_score"),
        "vlm_state_fidelity": row.get("d4_state_in_view_score"),
        "vlm_spatial_reasoning": row.get("d5_spatial_oov_score") if d5_app else None,
        "vlm_state_reasoning": row.get("d6_state_oov_score") if d6_app else None,
        "vlm_dimension_applicable": {
            "spatial_fidelity": True,
            "state_fidelity": True,
            "spatial_reasoning": d5_app,
            "state_reasoning": d6_app,
        },
        "vlm_probe_results": row.get("vlm_probe_results") or {},
        "runtime_v2_probe_rows": row.get("runtime_v2_probe_rows") or [],
        "runtime_v2_probe_status": row.get("probe_status"),
        "runtime_v2_schema": SCHEMA_VERSION,
        "runtime_v2_probe_catalog_version": PROBE_CATALOG_VERSION,
        "runtime_v2_prompt_mode": prompt_mode,
        "runtime_v2_task_context_mode": row.get("runtime_v2_task_context_mode"),
        "runtime_v2_score_export_policy": SCORE_EXPORT_POLICY,
        "runtime_v2_score_export_view": export_policy,
        "runtime_v2_d5_na_reason": row.get("d5_spatial_oov_na_reason"),
        "runtime_v2_d6_na_reason": row.get("d6_state_oov_na_reason"),
        "runtime_v2_d5_gate_applicable": row.get("d5_spatial_oov_gate_applicable"),
        "runtime_v2_d5_gate_na_reason": row.get("d5_spatial_oov_gate_na_reason"),
        "runtime_v2_d6_gate_applicable": row.get("d6_state_oov_gate_applicable"),
        "runtime_v2_d6_gate_na_reason": row.get("d6_state_oov_gate_na_reason"),
        "runtime_v2_shared_oov_gate_applicable": row.get("runtime_v2_shared_oov_gate_applicable"),
        "runtime_v2_shared_oov_gate_na_reason": row.get("runtime_v2_shared_oov_gate_na_reason"),
        "runtime_v2_d5_raw_score": row.get("d5_spatial_oov_score"),
        "runtime_v2_d6_raw_score": row.get("d6_state_oov_score"),
        "runtime_v2_d5_score_available": row.get("d5_spatial_oov_score") is not None,
        "runtime_v2_d6_score_available": row.get("d6_state_oov_score") is not None,
    }


def merge_sharded_outputs(
    *,
    output_dir: Path,
    manifest: list[dict[str, Any]],
    require_complete: bool = True,
) -> dict[str, Any]:
    chunk_paths = sorted(output_dir.glob("raw_v2_probe_scores_shard_*.jsonl"))
    if not chunk_paths and (output_dir / "raw_v2_probe_scores.jsonl").exists():
        chunk_paths = [output_dir / "raw_v2_probe_scores.jsonl"]
    chunk_rows: list[dict[str, Any]] = []
    for path in chunk_paths:
        chunk_rows.extend(load_jsonl(path))
    raw_by_id = id_map(chunk_rows)
    manifest_ids = [str(item.get("video_id") or "") for item in manifest if item.get("video_id")]
    missing_video_ids = [video_id for video_id in manifest_ids if video_id not in raw_by_id]
    extra_video_ids = sorted(video_id for video_id in raw_by_id if video_id not in set(manifest_ids))
    if require_complete and missing_video_ids:
        raise RuntimeError(
            "cannot merge incomplete probe shard output: "
            f"{len(missing_video_ids)} manifest videos are missing"
        )
    ordered_rows = [raw_by_id[video_id] for video_id in manifest_ids if video_id in raw_by_id]
    candidate_rows = [export_v7_style_probe_record(row) for row in ordered_rows if row.get("probe_status") == "ok"]
    score_available_rows = [
        export_v7_style_probe_record(row, export_policy="score_available")
        for row in ordered_rows
        if row.get("probe_status") == "ok"
    ]
    gate_masked_rows = [
        export_v7_style_probe_record(row, export_policy="gate_masked")
        for row in ordered_rows
        if row.get("probe_status") == "ok"
    ]
    write_jsonl(output_dir / "raw_v2_probe_scores.jsonl", ordered_rows)
    write_json(output_dir / "scores_v7_candidate_runtime_v2_probe.json", candidate_rows)
    write_json(
        output_dir / "scores_v7_candidate_runtime_v2_probe_score_available.json",
        score_available_rows,
    )
    write_json(
        output_dir / "scores_v7_candidate_runtime_v2_probe_gate_masked.json",
        gate_masked_rows,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "score_export_policy": SCORE_EXPORT_POLICY,
        "merge_status": "complete" if not missing_video_ids else "incomplete",
        "chunk_files": [str(path) for path in chunk_paths],
        "chunk_files_count": len(chunk_paths),
        "manifest_records": len(manifest_ids),
        "records_written": len(ordered_rows),
        "candidate_records_written": len(candidate_rows),
        "probe_ok": sum(1 for row in ordered_rows if row.get("probe_status") == "ok"),
        "missing_video_ids": missing_video_ids,
        "extra_video_ids": extra_video_ids,
    }
    write_json(output_dir / "merge_summary.json", summary)
    return summary


class LocalQwen35ProbeLogprobScorer:
    def __init__(
        self,
        *,
        model_path: Path,
        fps: str,
        dtype: str,
        attn_implementation: str,
        local_rank: int,
        dry_run: bool,
        prompt_mode: str = DEFAULT_PROMPT_MODE,
        task_context_mode: str = TASK_CONTEXT_MODE_ALL_MANIFEST_METADATA,
        loader_family: str = "auto",
        video_min_pixels: int | None = None,
        video_max_pixels: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.fps = fps
        self.dry_run = dry_run
        self.prompt_mode = validate_prompt_mode(prompt_mode)
        if task_context_mode not in SUPPORTED_TASK_CONTEXT_MODES:
            raise ValueError(
                "task_context_mode must be one of: "
                + ", ".join(sorted(SUPPORTED_TASK_CONTEXT_MODES))
            )
        self.task_context_mode = task_context_mode
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.processor = None
        self.model = None
        self.yes_token_ids: set[int] = set()
        self.no_token_ids: set[int] = set()
        self.tokenizer_probe: dict[str, Any] | None = None
        self.apply_chat_template_kwargs: dict[str, Any] = {}
        if dry_run:
            return
        import torch  # type: ignore
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

        dtype_obj = {"bfloat16": torch.bfloat16}[dtype]
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        video_backend = os.environ["WORLD_STATE_VIDEO_BACKEND"].strip()
        if not video_backend:
            raise RuntimeError("WORLD_STATE_VIDEO_BACKEND is required")
        video_processor = getattr(self.processor, "video_processor", None)
        if video_backend and video_processor is not None:
            from transformers.video_utils import load_video  # type: ignore

            def fetch_videos_with_backend(video_url_or_urls, sample_indices_fn=None):
                if isinstance(video_url_or_urls, list):
                    return list(
                        zip(
                            *[
                                fetch_videos_with_backend(x, sample_indices_fn=sample_indices_fn)
                                for x in video_url_or_urls
                            ]
                        )
                    )
                return load_video(
                    video_url_or_urls,
                    backend=video_backend,
                    sample_indices_fn=sample_indices_fn,
                )

            video_processor.fetch_videos = fetch_videos_with_backend

        def load_with(model_cls: Any) -> Any:
            try:
                return model_cls.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                    local_files_only=True,
                    dtype=dtype_obj,
                    device_map={"": local_rank},
                    attn_implementation=attn_implementation,
                )
            except TypeError:
                return model_cls.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=dtype_obj,
                    device_map={"": local_rank},
                    attn_implementation=attn_implementation,
                )

        if loader_family == "qwen25vl":
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

            self.model = load_with(Qwen2_5_VLForConditionalGeneration)
        elif loader_family == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration  # type: ignore

            self.model = load_with(Qwen3VLForConditionalGeneration)
        else:
            try:
                self.model = load_with(AutoModelForImageTextToText)
            except Exception:
                if model_config_model_type(model_path) != "qwen2_5_vl":
                    raise
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
                except Exception:
                    raise
                self.model = load_with(Qwen2_5_VLForConditionalGeneration)
        self.model.eval()
        tokenizer = self.processor.tokenizer
        self.yes_token_ids = token_ids_for_variants(tokenizer, YES_VARIANTS)
        self.no_token_ids = token_ids_for_variants(tokenizer, NO_VARIANTS)
        self.tokenizer_probe = tokenizer_candidate_probe(tokenizer)
        self.apply_chat_template_kwargs = build_apply_chat_template_kwargs(
            self.processor,
            fps=str(self.fps),
            video_min_pixels=self.video_min_pixels,
            video_max_pixels=self.video_max_pixels,
        )

    def run_config_metadata(self) -> dict[str, Any]:
        processor = self.processor
        video_processor = getattr(processor, "video_processor", None) if processor is not None else None
        processor_kwargs = self.apply_chat_template_kwargs.get("processor_kwargs", {})
        videos_kwargs = processor_kwargs.get("videos_kwargs", {}) if isinstance(processor_kwargs, dict) else {}
        return {
            "processor_class": type(processor).__name__ if processor is not None else None,
            "video_processor_class": type(video_processor).__name__ if video_processor is not None else None,
            "processor_chat_template_supports_enable_thinking": (
                chat_template_supports_variable(processor, "enable_thinking")
                if processor is not None
                else None
            ),
            "enable_thinking_argument_sent": "enable_thinking" in self.apply_chat_template_kwargs,
            "processor_kwargs_policy": (
                "numeric_fps_sets_videos_kwargs_fps_and_do_sample_frames"
                if str(self.fps) != "full"
                else "processor_default_full_video_policy"
            ),
            "processor_videos_kwargs": dict(videos_kwargs),
            "video_min_pixels": self.video_min_pixels,
            "video_max_pixels": self.video_max_pixels,
            "yes_token_ids": sorted(self.yes_token_ids),
            "no_token_ids": sorted(self.no_token_ids),
            "tokenizer_candidate_probe": self.tokenizer_probe,
        }

    def _score_probe(self, item: dict[str, Any], probe: RuntimeV2Probe) -> tuple[dict[str, Any], int, dict[str, float]]:
        assert self.processor is not None and self.model is not None
        import torch  # type: ignore

        started = time.perf_counter()
        video_id = str(item["video_id"])
        video_path = str(item.get("path") or item.get("video_path") or "")
        prompt_text = str(item.get("world_state_prompt") or item.get("prompt_text") or "")
        prompt = build_runtime_v2_probe_prompt(
            world_state_prompt=prompt_text,
            video_id=video_id,
            probe=probe,
            task_context=build_task_context(item, task_context_mode=self.task_context_mode),
            fps=str(self.fps),
            prompt_mode=self.prompt_mode,
            evidence_context=item.get("_runtime_v2_evidence_context"),
            evidence_context_mode=item.get("_runtime_v2_evidence_context_mode"),
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_ready = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            messages,
            **deepcopy(self.apply_chat_template_kwargs),
        )
        processor_done = time.perf_counter()
        frames_used = derive_frames_used_from_processor_inputs(inputs, self.processor)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        model_started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        model_done = time.perf_counter()
        logits = outputs.logits[0, -1, :]
        p_yes = binary_probability_from_logits(
            logits,
            yes_token_ids=self.yes_token_ids,
            no_token_ids=self.no_token_ids,
        )
        return (
            _probe_meta(
                probe,
                p_yes,
                "logprob_forced_yes_no",
                prompt_mode=self.prompt_mode,
            ),
            frames_used,
            {
                "prompt_build_seconds": round(prompt_ready - started, 6),
                "processor_seconds": round(processor_done - prompt_ready, 6),
                "device_transfer_seconds": round(model_started - processor_done, 6),
                "model_forward_seconds": round(model_done - model_started, 6),
                "total_seconds": round(model_done - started, 6),
            },
        )

    def score(self, item: dict[str, Any]) -> dict[str, Any]:
        video_id = str(item["video_id"])
        if self.dry_run:
            row = aggregate_probe_results(
                dry_run_probe_results(video_id, prompt_mode=self.prompt_mode),
                video_id=video_id,
                sampling_fps=self.fps,
                frames_used=96 if self.fps == "full" else max(1, int(float(self.fps) * 6)),
                prompt_mode=self.prompt_mode,
            )
            row["runtime_v2_task_context_mode"] = self.task_context_mode
            return row
        probe_results: dict[str, dict[str, Any]] = {}
        frames_seen: list[int] = []
        timing_rows: list[dict[str, Any]] = []
        for probe in active_probe_catalog(self.prompt_mode):
            result, frames_used, timing = self._score_probe(item, probe)
            probe_results[probe.probe_id] = result
            frames_seen.append(frames_used)
            timing_rows.append({"probe_id": probe.probe_id, **timing})
        row = aggregate_probe_results(
            probe_results,
            video_id=video_id,
            sampling_fps=self.fps,
            frames_used=max(frames_seen or [0]),
            prompt_mode=self.prompt_mode,
        )
        row["runtime_v2_task_context_mode"] = self.task_context_mode
        row["runtime_v2_probe_timing_rows"] = timing_rows
        row["runtime_v2_timing_seconds"] = {
            "processor": round(sum(float(t["processor_seconds"]) for t in timing_rows), 6),
            "model_forward": round(sum(float(t["model_forward_seconds"]) for t in timing_rows), 6),
            "device_transfer": round(sum(float(t["device_transfer_seconds"]) for t in timing_rows), 6),
            "total": round(sum(float(t["total_seconds"]) for t in timing_rows), 6),
        }
        return row


def median_sample_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must be non-empty")
    probe_results: dict[str, dict[str, Any]] = {}
    prompt_mode = str(rows[0].get("runtime_v2_prompt_mode") or DEFAULT_PROMPT_MODE)
    for probe in active_probe_catalog(prompt_mode):
        metas = [
            (row.get("vlm_probe_results") or {}).get(probe.probe_id)
            for row in rows
        ]
        metas = [meta for meta in metas if isinstance(meta, dict) and meta.get("p_yes") is not None]
        if not metas:
            continue
        p_yes = round(float(median([float(meta["p_yes"]) for meta in metas])), 6)
        merged = dict(metas[0])
        merged.update(
            {
                "p_yes": p_yes,
                "value": 1 if p_yes >= GATE_THRESHOLD else 0,
                "status": "median_sample_logprob",
            }
        )
        probe_results[probe.probe_id] = merged
    row = aggregate_probe_results(
        probe_results,
        video_id=str(rows[0].get("video_id") or ""),
        sampling_fps=str(rows[0].get("sampling_fps") or ""),
        frames_used=max(int(sample.get("frames_used") or 0) for sample in rows),
        prompt_mode=prompt_mode,
    )
    row["runtime_v2_task_context_mode"] = rows[0].get(
        "runtime_v2_task_context_mode",
        TASK_CONTEXT_MODE_ALL_MANIFEST_METADATA,
    )
    row["sample_count"] = len(rows)
    row["sample_probe_status_counts"] = dict(Counter(str(sample.get("probe_status")) for sample in rows))
    return row


def _version_or_unavailable(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def _cuda_build_or_unavailable() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "unavailable"
    return str(getattr(torch.version, "cuda", None) or "unavailable")


def model_config_model_type(model_path: Path) -> str | None:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        config = load_json(config_path)
    except Exception:
        return None
    value = config.get("model_type") if isinstance(config, dict) else None
    return str(value) if value is not None else None


def build_run_config(
    *,
    args: argparse.Namespace,
    records_written: int,
    records_expected: int,
    candidate_records_written: int,
    max_frames_observed: int,
    frames_used_values: list[int],
    local_rank: int,
    scorer_metadata: dict[str, Any] | None = None,
    rebuilt_pool_preflight: dict[str, Any] | None = None,
    video_file_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scorer_metadata = scorer_metadata or {}
    frames = [int(value) for value in frames_used_values if int(value) >= 0]
    frames_used_summary = {
        "min": min(frames) if frames else 0,
        "max": max(frames) if frames else 0,
        "mean": round(sum(frames) / len(frames), 4) if frames else 0.0,
        "median": median(frames) if frames else 0,
        "n": len(frames),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "score_export_policy": SCORE_EXPORT_POLICY,
        "experiment_id": args.experiment_id,
        "prompt_mode": args.prompt_mode,
        "task_context_mode": args.task_context_mode,
        "strict_manifest_contract": bool(args.strict_manifest_contract),
        "probe_catalog_version": PROBE_CATALOG_VERSION,
        "calls_per_video": len(active_probe_catalog(args.prompt_mode)) * int(args.num_samples),
        "num_samples": int(args.num_samples),
        "fps": str(args.fps),
        "max_frames_observed": int(max_frames_observed),
        "frames_used_summary": frames_used_summary,
        "model_path": str(args.model_path),
        "vlm_name": str(args.vlm_name),
        "loader_family": str(args.loader_family),
        "model_config_model_type": model_config_model_type(args.model_path),
        "python_path": sys.executable,
        "transformers_version": _version_or_unavailable("transformers"),
        "torch_version": _version_or_unavailable("torch"),
        "cuda_build": _cuda_build_or_unavailable(),
        "torch_dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "video_sampling_policy": (
            "processor_default_not_all_source_frames"
            if str(args.fps) == "full"
            else "processor_kwargs_fps"
        ),
        **scorer_metadata,
        "device_map": {"": int(local_rank)},
        "local_rank": int(local_rank),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "input_manifest_path": str(args.manifest_path),
        "input_manifest_sha256": sha256_file(args.manifest_path),
        "source_scores_path": str(args.source_scores),
        "source_scores_sha256": sha256_file(args.source_scores),
        "pair_allowlist_jsonl": str(args.pair_allowlist_jsonl) if args.pair_allowlist_jsonl else None,
        "pair_allowlist_sha256": sha256_file(args.pair_allowlist_jsonl),
        "require_rebuilt_pool_20260519": bool(args.require_rebuilt_pool_20260519),
        "rebuilt_pool_20260519_preflight": rebuilt_pool_preflight,
        "video_file_preflight": video_file_preflight,
        "evidence_jsonl_path": str(args.evidence_jsonl) if args.evidence_jsonl else None,
        "evidence_jsonl_sha256": sha256_file(args.evidence_jsonl),
        "evidence_context_mode": args.evidence_context_mode if args.evidence_jsonl else None,
        "records_written": int(records_written),
        "records_expected": int(records_expected),
        "candidate_records_written": int(candidate_records_written),
        "dry_run": bool(args.dry_run),
        "skip_existing": bool(args.skip_existing),
        "progress_every": int(args.progress_every),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime V2 local Qwen3.5 probe-logprob scorer")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--prompt-mode", required=True)
    parser.add_argument(
        "--task-context-mode",
        choices=tuple(sorted(SUPPORTED_TASK_CONTEXT_MODES)),
        required=True,
    )
    parser.add_argument("--strict-manifest-contract", action="store_true")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument(
        "--source-scores",
        type=Path,
        help=(
            "Optional legacy/source score JSON list. If omitted, the manifest "
            "is used as the source metadata so this standalone scorer can score "
            "a plain video manifest directly."
        ),
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-allowlist-jsonl", type=Path)
    parser.add_argument("--require-rebuilt-pool-20260519", action="store_true")
    parser.add_argument("--expected-pair-allowlist-sha256")
    parser.add_argument("--model-path", type=Path, required=True, help="Local Qwen3.5 model directory (set via wrbench.runtime.json eval.scorers.qwen35_model).")
    parser.add_argument("--vlm-name", required=True)
    parser.add_argument("--loader-family", choices=("auto", "qwen3vl", "qwen25vl"), required=True)
    parser.add_argument("--fps", required=True)
    parser.add_argument("--max-videos", type=int, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--allow-incomplete-merge", action="store_true")
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--attn-implementation", required=True)
    parser.add_argument("--local-rank", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--progress-every", type=int, required=True)
    parser.add_argument("--video-min-pixels", type=int)
    parser.add_argument("--video-max-pixels", type=int)
    parser.add_argument("--evidence-jsonl", type=Path)
    parser.add_argument(
        "--evidence-context-mode",
        choices=("visibility_v1", "subquestion_v1"),
        required=True,
    )
    args = parser.parse_args(argv)
    if args.prompt_mode not in SUPPORTED_PROMPT_MODES:
        parser.error(
            "--prompt-mode must be one of: "
            + ", ".join(sorted(SUPPORTED_PROMPT_MODES))
        )
    if args.strict_manifest_contract:
        if args.prompt_mode != PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED:
            parser.error(
                "--strict-manifest-contract requires --prompt-mode "
                f"{PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED}"
            )
        if args.task_context_mode != TASK_CONTEXT_MODE_ALL_MANIFEST_METADATA:
            parser.error(
                "--strict-manifest-contract requires --task-context-mode "
                f"{TASK_CONTEXT_MODE_ALL_MANIFEST_METADATA}"
            )
        if args.skip_existing:
            parser.error("--strict-manifest-contract does not allow --skip-existing")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        parser.error("--shard-id must satisfy 0 <= shard_id < num_shards")
    if args.max_videos < 0:
        parser.error("--max-videos must be >= 0")
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.dtype != DEFAULT_DTYPE:
        parser.error("--dtype must be bfloat16 for this route")
    if args.attn_implementation != DEFAULT_ATTN_IMPLEMENTATION:
        parser.error("--attn-implementation must be flash_attention_2 for this route")
    if args.video_min_pixels is not None and args.video_min_pixels <= 0:
        parser.error("--video-min-pixels must be positive")
    if args.video_max_pixels is not None and args.video_max_pixels <= 0:
        parser.error("--video-max-pixels must be positive")
    if (
        args.video_min_pixels is not None
        and args.video_max_pixels is not None
        and args.video_min_pixels > args.video_max_pixels
    ):
        parser.error("--video-min-pixels must be <= --video-max-pixels")
    if args.require_rebuilt_pool_20260519 and args.pair_allowlist_jsonl is None:
        parser.error("--require-rebuilt-pool-20260519 requires --pair-allowlist-jsonl")
    if args.require_rebuilt_pool_20260519 and not args.expected_pair_allowlist_sha256:
        parser.error("--require-rebuilt-pool-20260519 requires --expected-pair-allowlist-sha256")
    if str(args.fps) != "full":
        try:
            fps_value = float(args.fps)
        except ValueError:
            parser.error("--fps must be a positive number or full")
        if fps_value < MIN_NUMERIC_FPS:
            parser.error("--fps must be >= 2 or full")
        args.fps = str(int(fps_value)) if fps_value.is_integer() else str(fps_value)
    return args


def _stage_result(
    *,
    args: argparse.Namespace,
    records_written: int,
    records_expected: int,
    candidate_records_written: int,
) -> str:
    decision = "pass" if records_written >= records_expected else "retry_once"
    return (
        f"# Stage {args.experiment_id}\n\n"
        f"- experiment_id: {args.experiment_id}\n"
        "- scorer: runtime_v2_probe_logprob\n"
        f"- decision: {decision}\n\n"
        "## metrics\n"
        f"- records_expected: {records_expected}\n"
        f"- records_written: {records_written}\n"
        f"- candidate_records_written: {candidate_records_written}\n\n"
        "## notes\n"
        "- Candidate Runtime V2 probe-logprob artifacts generated. This does not overwrite canonical V7.\n"
        "- D5/D6 score probes are exported as continuous scores; gate probes are diagnostic provenance.\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    local_rank = int(args.local_rank)
    manifest = load_json(args.manifest_path)
    source_scores = load_json(args.source_scores) if args.source_scores else manifest
    if not isinstance(manifest, list) or not isinstance(source_scores, list):
        raise ValueError("--manifest-path and --source-scores must contain JSON lists")
    rebuilt_pool_preflight = None
    if args.require_rebuilt_pool_20260519:
        rebuilt_pool_preflight = validate_rebuilt_pool_20260519_preflight(
            manifest=manifest,
            manifest_path=args.manifest_path,
            output_dir=args.output_dir,
            pair_allowlist_jsonl=args.pair_allowlist_jsonl,
            expected_allowlist_sha256=args.expected_pair_allowlist_sha256,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_manifest = manifest[: args.max_videos] if args.max_videos else manifest
        merge_sharded_outputs(
            output_dir=args.output_dir,
            manifest=merge_manifest,
            require_complete=not args.allow_incomplete_merge,
        )
        return 0
    source_by_id = id_map(source_scores)
    shard_items = select_manifest_shard(manifest, num_shards=args.num_shards, shard_id=args.shard_id)
    if args.max_videos:
        shard_items = shard_items[: args.max_videos]
    video_file_preflight = validate_manifest_video_files(shard_items)
    evidence_by_id = load_evidence_by_video_id(args.evidence_jsonl)
    if evidence_by_id:
        missing_evidence_ids = [
            str(item.get("video_id"))
            for item in shard_items
            if item.get("video_id") is not None and str(item.get("video_id")) not in evidence_by_id
        ]
        if missing_evidence_ids:
            raise ValueError(
                "--evidence-jsonl is missing evidence for shard videos: "
                f"{missing_evidence_ids[:5]}"
            )
    raw_path = (
        args.output_dir / "raw_v2_probe_scores.jsonl"
        if args.num_shards == 1
        else args.output_dir / f"raw_v2_probe_scores_shard_{args.shard_id}.jsonl"
    )
    loaded_existing_rows = load_jsonl(raw_path) if args.skip_existing else []
    existing_rows = filter_resume_rows(loaded_existing_rows, args=args) if args.skip_existing else []
    existing_by_id = id_map(existing_rows)
    scorer = LocalQwen35ProbeLogprobScorer(
        model_path=args.model_path,
        fps=str(args.fps),
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        local_rank=local_rank,
        dry_run=args.dry_run,
        prompt_mode=args.prompt_mode,
        task_context_mode=args.task_context_mode,
        loader_family=args.loader_family,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
    )
    rows_by_id = dict(existing_by_id)
    shard_total = len(shard_items)
    completed_at_start = sum(
        1 for item in shard_items if str(item.get("video_id")) in existing_by_id
    )
    if args.skip_existing and completed_at_start:
        print(
            f"[resume] shard={args.shard_id}/{args.num_shards} "
            f"existing={completed_at_start}/{shard_total} "
            f"loaded={len(loaded_existing_rows)} compatible={len(existing_rows)} path={raw_path}",
            flush=True,
        )
    for item_index, item in enumerate(shard_items, start=1):
        video_id = str(item["video_id"])
        if args.skip_existing and video_id in existing_by_id:
            continue
        item_started = time.perf_counter()
        merged_item = merge_item(source_by_id.get(video_id), item)
        merged_item = attach_evidence_context(
            merged_item,
            evidence_by_id=evidence_by_id,
            evidence_context_mode=args.evidence_context_mode if args.evidence_jsonl else None,
        )
        sample_rows = [scorer.score(merged_item) for _ in range(args.num_samples)]
        row = sample_rows[0] if len(sample_rows) == 1 else median_sample_rows(sample_rows)
        row.update(
            {
                "video_id": video_id,
                "path": merged_item.get("path") or merged_item.get("video_path"),
                "model": merged_item.get("model"),
                "variant_id": merged_item.get("variant_id"),
                "world_state_prompt": merged_item.get("world_state_prompt") or merged_item.get("prompt_text"),
                "vlm_name": args.vlm_name,
            }
        )
        rows_by_id[video_id] = row
        ordered_checkpoint_rows = [
            rows_by_id[str(shard_item["video_id"])]
            for shard_item in shard_items
            if str(shard_item.get("video_id")) in rows_by_id
        ]
        write_jsonl_atomic(raw_path, ordered_checkpoint_rows)
        if args.progress_every and (
            item_index == shard_total or len(ordered_checkpoint_rows) % args.progress_every == 0
        ):
            timing = row.get("runtime_v2_timing_seconds") or {}
            elapsed = time.perf_counter() - item_started
            print(
                f"[progress] shard={args.shard_id}/{args.num_shards} "
                f"records={len(ordered_checkpoint_rows)}/{shard_total} "
                f"video_id={video_id} elapsed={elapsed:.2f}s "
                f"processor={float(timing.get('processor') or 0):.2f}s "
                f"forward={float(timing.get('model_forward') or 0):.2f}s",
                flush=True,
            )
    ordered_rows = [rows_by_id[str(item["video_id"])] for item in shard_items if str(item.get("video_id")) in rows_by_id]
    candidate_rows = [export_v7_style_probe_record(row) for row in ordered_rows if row.get("probe_status") == "ok"]
    score_available_rows = [
        export_v7_style_probe_record(row, export_policy="score_available")
        for row in ordered_rows
        if row.get("probe_status") == "ok"
    ]
    gate_masked_rows = [
        export_v7_style_probe_record(row, export_policy="gate_masked")
        for row in ordered_rows
        if row.get("probe_status") == "ok"
    ]
    max_frames_observed = max([int(row.get("frames_used") or 0) for row in ordered_rows] or [0])
    frames_used_values = [int(row.get("frames_used") or 0) for row in ordered_rows]
    write_jsonl_atomic(raw_path, ordered_rows)
    if args.num_shards == 1:
        write_json(args.output_dir / "scores_v7_candidate_runtime_v2_probe.json", candidate_rows)
        write_json(
            args.output_dir / "scores_v7_candidate_runtime_v2_probe_score_available.json",
            score_available_rows,
        )
        write_json(
            args.output_dir / "scores_v7_candidate_runtime_v2_probe_gate_masked.json",
            gate_masked_rows,
        )
    write_json(
        args.output_dir / ("run_config.json" if args.num_shards == 1 else f"run_config_shard_{args.shard_id}.json"),
        build_run_config(
            args=args,
            records_written=len(ordered_rows),
            records_expected=len(shard_items),
            candidate_records_written=len(candidate_rows),
            max_frames_observed=max_frames_observed,
            frames_used_values=frames_used_values,
            local_rank=local_rank,
            scorer_metadata=scorer.run_config_metadata(),
            rebuilt_pool_preflight=rebuilt_pool_preflight,
            video_file_preflight=video_file_preflight,
        ),
    )
    (args.output_dir / ("stage_result.md" if args.num_shards == 1 else f"stage_result_shard_{args.shard_id}.md")).write_text(
        _stage_result(
            args=args,
            records_written=len(ordered_rows),
            records_expected=len(shard_items),
            candidate_records_written=len(candidate_rows),
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
