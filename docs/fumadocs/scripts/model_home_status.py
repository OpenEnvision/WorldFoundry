"""Map catalog/manifest status dialect to reader-facing labels.

Generated model-home prose must never emit raw tokens such as
``profile_resolves_*``, ``official_demo_parity_pending``, or
``in_tree_checkpoint_runtime_static_verified``.
"""

from __future__ import annotations

import re
from typing import Any

# Four reader labels. Do not invent a fifth.
READER_LABELS = {
    "en": {
        "verified": "Verified",
        "pending": "Runner parity pending",
        "static": "Static load only",
        "planned": "Planned",
    },
    "zh": {
        "verified": "已验证",
        "pending": "Runner 一致性待确认",
        "static": "仅静态加载",
        "planned": "计划中",
    },
}

# Longest-first so ``in_tree_checkpoint_runtime_static_verified`` wins over
# a shorter ``in_tree_checkpoint_runtime`` prefix.
_TOKEN_KEYS: tuple[tuple[str, str], ...] = (
    ("native_checkpoint_structure_validated", "static"),
    ("native_checkpoint_structure_verified", "static"),
    ("in_tree_checkpoint_runtime_static_verified", "static"),
    ("in_tree_checkpoint_runtime_static_validated", "static"),
    ("static_runtime_verified_checkpoint_assets_staged", "static"),
    ("static_runtime_verified", "static"),
    ("official_in_process_runtime_static_checkpoint_validation_complete", "static"),
    ("official_demo_parity_pending", "pending"),
    ("profile_resolves_catalog_variant_runner_parity_pending", "pending"),
    ("native_pipeline_cuda_artifact_pending", "pending"),
    ("native_structure_and_scheduler_parity_validated_gpu_parity_pending", "pending"),
    ("catalog_demo_and_runner_parity_recorded_checkpoint_runtime_required", "verified"),
    ("all_released_checkpoints_gpu_validated", "verified"),
    ("all_declared_variants_checkpoint_gpu_validated", "verified"),
    ("in_tree_runtime_selected_checkpoint_gpu_validated", "verified"),
    ("selected_checkpoint_gpu_validated", "verified"),
    ("checkpoint_gpu_validated", "verified"),
    ("checkpoint_backed_verified", "verified"),
    ("checkpoint_environment_required", "pending"),
    ("partial_studio_visual_validation_verified", "verified"),
    ("full_default_gpu_parity_verified", "verified"),
    ("full_demo_verified", "verified"),
    ("converted_checkpoint_strict_restore_and_gpu_action_probe_validated", "verified"),
    ("robotwin_and_umi_checkpoints_gpu_validated", "verified"),
    ("in_tree_checkpoint_runtime_gpu_server_init_verified_rollout_pending", "pending"),
    ("in_tree_checkpoint_runtime_gpu_forward_server_init_verified", "pending"),
    ("in_tree_checkpoint_runtime_checkpoint_gpu_probe_verified", "pending"),
    ("in_tree_vendored_import_verified", "static"),
    ("in_tree_checkpoint_runtime", "pending"),
    ("in_tree_in_process_checkpoint_runtime", "pending"),
    ("in_tree_vendor_ready_checkpoint_pending", "pending"),
    ("in_tree_inference_ready_checkpoint_pending", "pending"),
    ("in_tree_import_verification_in_progress", "planned"),
    ("checkpoint_assets_staged_cpu_schema_validated_gpu_pending", "pending"),
    ("implemented_checkpoint_pending", "pending"),
    ("implemented_checkpoint_required", "pending"),
    ("implemented_pending_gpu_parity", "pending"),
    ("implemented_pending_native_gpu_parity", "pending"),
    ("implemented_public_checkpoint_not_staged", "pending"),
    ("pending_official_sample_parity", "pending"),
    ("pending_gpu_validation", "pending"),
    ("pending_visual_qa", "pending"),
    ("runtime_ported", "pending"),
    ("checkpoint_backed_runtime_ready", "pending"),
    ("native_checkpoint_layout_validated", "static"),
    ("native_checkpoint_validated", "static"),
    ("metadata_only", "planned"),
    ("component_verified", "pending"),
)

_VERIFIED_EXACT = frozenset(
    {
        "verified",
        "validated",
        "passed",
        "checkpoint_verified",
    }
)
_STATIC_EXACT = frozenset(
    {
        "static",
        "static_load",
        "static_only",
        "static_runtime",
    }
)
_PLANNED_EXACT = frozenset(
    {
        "planned",
        "todo",
        "proposed",
        "profile",
        "profile_only",
        "metadata_only",
        "not_started",
        "blocked",
    }
)
_PENDING_EXACT = frozenset(
    {
        "pending",
        "integrated",
        "runtime_ported",
        "ported",
        "implemented",
        "configured",
        "partial",
        "not_recorded",
        "not_applicable",
    }
)

# Any leftover catalog snake_case that must not reach the reader.
MANIFEST_DIALECT_RE = re.compile(
    r"(?i)\b("
    r"profile_resolves_[a-z0-9_]+|"
    r"official_demo_parity_pending|"
    r"official_demo_[a-z0-9_]+|"
    r"in_tree_checkpoint_runtime[a-z0-9_]*|"
    r"in_tree_checkpoint_gated_runtime|"
    r"in_tree_in_process_checkpoint_runtime|"
    r"in_tree_vendored_[a-z0-9_]+|"
    r"in_tree_import_verification_in_progress|"
    r"in_tree_inference_ready_checkpoint_pending|"
    r"in_tree_runtime_selected_checkpoint_gpu_validated|"
    r"static_runtime_verified|"
    r"static_runtime_verified_[a-z0-9_]+|"
    r"runtime_ported|"
    r"checkpoint_assets_staged_[a-z0-9_]+|"
    r"checkpoint_gpu_validated|"
    r"checkpoint_backed_verified|"
    r"checkpoint_backed_runtime_ready|"
    r"checkpoint_environment_required|"
    r"partial_studio_visual_validation_verified|"
    r"all_released_checkpoints_gpu_validated|"
    r"all_declared_variants_checkpoint_gpu_validated|"
    r"implemented_checkpoint_[a-z0-9_]+|"
    r"implemented_pending_[a-z0-9_]+|"
    r"native_pipeline_cuda_artifact_pending|"
    r"native_checkpoint_[a-z0-9_]+|"
    r"native_structure_and_scheduler_[a-z0-9_]+|"
    r"catalog_demo_and_runner_[a-z0-9_]+|"
    r"metadata_only(?:_post_training_base)?"
    r")\b"
)

_STATUS_PHRASE_RE = re.compile(
    r"(?i)\b((?:Runner|Integration|runner)\s+status\s+is|Integration is|集成为|runner 状态为)\s+"
    r"(?:\*{1,2})?['\"]?("
    r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+|"
    r"configured|integrated|ported"
    r")['\"]?(?:\*{1,2})?"
)

_HUMANIZED_DIALECT_RE = re.compile(
    r"(?i)\b("
    r"In Tree Checkpoint Runtime Static Verified|"
    r"In Tree Checkpoint Runtime Static Validated|"
    r"In Tree Checkpoint Runtime|"
    r"Official Demo Parity Pending|"
    r"Runtime Ported|"
    r"Profile Resolves Catalog Variant Runner Parity Pending|"
    r"All Released Checkpoints Gpu Validated|"
    r"Static Runtime Verified|"
    r"Metadata Only|"
    r"Implemented Checkpoint Pending|"
    r"Checkpoint Backed Verified|"
    r"Checkpoint Environment Required"
    r")\b"
)


def reader_label(key: str, locale: str) -> str:
    table = READER_LABELS["zh" if locale == "zh" else "en"]
    return table.get(key, table["pending"])


def normalize_token(value: str | None) -> str:
    return re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())


def classify_token(value: str | None) -> str:
    """Return one of verified | pending | static | planned."""

    raw = str(value or "").strip()
    if not raw:
        return "pending"
    token = normalize_token(raw)
    if token.startswith("profile_resolves_"):
        return "pending"
    for needle, key in _TOKEN_KEYS:
        if needle in token:
            return key
    if token in _VERIFIED_EXACT:
        return "verified"
    if token in _STATIC_EXACT or token.startswith("static_"):
        return "static"
    if token in _PLANNED_EXACT:
        return "planned"
    if "static" in token and ("verified" in token or "validated" in token or "load" in token):
        return "static"
    if any(part in token for part in ("gpu_validated", "parity_recorded", "runner_verified")):
        return "verified"
    if token in _PENDING_EXACT or "pending" in token or "ported" in token:
        return "pending"
    if "verified" in token or "validated" in token or token == "passed":
        return "verified"
    return "pending"


def recipe_status_key(recipe: dict[str, Any]) -> str:
    status = recipe.get("status") if isinstance(recipe.get("status"), dict) else {}
    runner = str(status.get("runner") or "")
    integration = str(status.get("integration") or "")
    group = str(status.get("group") or "")
    demo = str(status.get("demo") or "")

    runner_key = classify_token(runner) if runner and runner != "not_recorded" else ""
    if runner_key == "verified":
        return "verified"
    if runner_key == "static":
        return "static"

    integration_key = classify_token(integration) if integration else ""
    if integration_key == "static":
        return "static"
    if group == "verified" or integration_key == "verified":
        return "verified"
    if group in {"planned", "profile", "blocked"} or integration_key == "planned":
        return "planned"
    if demo and demo not in {"not_recorded", "not_applicable"}:
        demo_key = classify_token(demo)
        if demo_key == "verified":
            return "verified"
        if demo_key == "planned":
            return "planned"
    if runner_key:
        return runner_key
    if integration_key:
        return integration_key
    return classify_token(group or "pending")


def status_label(recipe: dict[str, Any], locale: str) -> str:
    return reader_label(recipe_status_key(recipe), locale)


def status_sentence(recipe: dict[str, Any], locale: str) -> str:
    label = status_label(recipe, locale)
    if locale == "zh":
        return f"状态：{label}。"
    return f"Status: {label}."


def _humanized_to_key(text: str) -> str:
    return classify_token(normalize_token(text))


def sanitize_manifest_dialect(text: str, locale: str) -> str:
    """Replace raw catalog tokens with reader labels. Leave other prose intact."""

    if not text:
        return text

    def replace_known(match: re.Match[str]) -> str:
        return reader_label(classify_token(match.group(0)), locale)

    def replace_status_phrase(match: re.Match[str]) -> str:
        prefix, token = match.group(1), match.group(2)
        return f"{prefix} {reader_label(classify_token(token), locale)}"

    # Phrase first so "status is token" does not also match the reader label.
    updated = _STATUS_PHRASE_RE.sub(replace_status_phrase, text)
    updated = MANIFEST_DIALECT_RE.sub(replace_known, updated)
    updated = _HUMANIZED_DIALECT_RE.sub(lambda m: reader_label(_humanized_to_key(m.group(0)), locale), updated)
    return updated


def dialect_hits(text: str) -> list[str]:
    found: list[str] = []
    for match in MANIFEST_DIALECT_RE.finditer(text):
        found.append(match.group(0))
    if "profile_resolves_" in text and "profile_resolves_" not in found:
        found.append("profile_resolves_")
    return found
