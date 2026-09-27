#!/usr/bin/env python3
"""Build the docs model-recipe index from WorldFoundry's source-of-truth manifests.

The generated JSON deliberately keeps catalog support, runtime integration,
environment compatibility, checkpoint provenance, and runner evidence separate.
Missing data stays missing instead of being inferred into a stronger claim.

Optional per-model ``docs:`` block
----------------------------------

A catalog manifest MAY carry a curated documentation block. When present it
seeds the generated per-model MDX page under
``docs/fumadocs/content/docs/guides/supported-models/<id>.mdx``; when absent the
generator composes an equivalent narrative from the recorded catalog, runtime,
binding, and evidence fields. Hand-written pages set ``pageSource: authored``
and are not overwritten. The article source of truth is the MDX file.
``homepage:`` is accepted as an alias of ``docs:`` (``docs:`` wins on
conflicts). All keys are optional. Schema::

    docs:
      # 2-5 plain-language sentences: what the model is at *inference* time,
      # what it takes in, what it produces. Do not lead with pretraining
      # hours, datasets, optimizer, or the authors' training recipe.
      # English required if the block is used; the *_zh mirror keeps the
      # Chinese page in sync.
      overview: >-
        ...
      overview_zh: >-
        ...
      # Runtime internals (backbone, conditioning, decoding, controls).
      # Only recorded facts — write "Not recorded in this repository" when
      # the manifest and official sources do not describe a detail.
      # Training recipes stay out unless one clause is needed to name a checkpoint.
      architecture: >-
        ...
      architecture_zh: >-
        ...
      # Practical notes for running this entry (assets, keys, environment
      # quirks, resolution/frames/steps, what the WorldFoundry route does).
      usage_notes: >-
        ...
      usage_notes_zh: >-
        ...
      # Formal name of the publishing institution. Companies first, then
      # universities/labs. Never a GitHub username or avatar identity.
      publisher: Alibaba (Tongyi Lab)
      publisher_kind: company   # company | university | lab
      publisher_zh: 阿里巴巴（通义实验室）
      # Short capability bullets rendered under "What this model is".
      highlights: [ ... ]
      highlights_zh: [ ... ]
      # Free-form modality tokens, e.g. text / image / video / action.
      # Derived from task ids (``text-to-video`` => text -> video) if absent.
      modalities:
        inputs: [text, image]
        outputs: [video]
      # Typical scenarios; rendered as a bullet list when present.
      use_cases: [ ... ]
      use_cases_zh: [ ... ]
      # Hardware guidance. min_vram_gb is also auto-discovered from any
      # recorded ``min_vram_gb`` field elsewhere in the manifest.
      hardware:
        min_vram_gb: 24
        recommended: "1x A100 80GB"
        notes: [ ... ]
      # Benchmarks that make sense for this model. ``id`` must be a
      # benchmark id under worldfoundry/data/benchmarks/catalog (same ids
      # as /docs/evaluation/benchmark-hub/<id>). Benchmarks whose names
      # appear in this manifest's evidence notes are additionally surfaced
      # automatically with source="manifest".
      recommended_benchmarks:
        - id: vbench
          reason: ...
          reason_zh: ...
      # Honest constraints. Synthesized from parity records when absent —
      # never write marketing claims here; unverified GPU evidence must
      # stay "Not recorded".
      limitations: [ ... ]
      limitations_zh: [ ... ]
      # Optional paper teaser. Usually filled by harvest-model-media.py
      # from arXiv / in-repo demos; a catalog may override it.
      paper:
        title: ...
        year: 2025
        venue: CVPR 2025
        arxiv_id: 2411.18673
      figures:
        - src: /models/ac3d/paper.png
          caption: ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = Path(__file__).resolve().parents[1]
OUT = DOCS_ROOT / "lib" / "model-recipes-data.json"
INDEX_OUT = DOCS_ROOT / "lib" / "model-recipes-index.json"
PAPER_MEDIA_PATH = DOCS_ROOT / "lib" / "model-paper-media.json"

# Inference specs expose resolved default paths. Pin their path environment so
# checked-in docs do not depend on the developer's home, cache, or adjacent
# checkpoint directories.
CANONICAL_PROJECT_ROOT = "/workspace"
CANONICAL_PACKAGE_ROOT = f"{CANONICAL_PROJECT_ROOT}/worldfoundry"
CANONICAL_HOME = "/home/ubuntu/.cache/worldfoundry"
os.environ.update(
    {
        "WORLDFOUNDRY_HOME": CANONICAL_HOME,
        "WORLDFOUNDRY_CACHE_DIR": CANONICAL_HOME,
        "WORLDFOUNDRY_DATA_DIR": f"{CANONICAL_HOME}/data",
        "WORLDFOUNDRY_ARTIFACT_DIR": f"{CANONICAL_HOME}/artifacts",
        "WORLDFOUNDRY_MODEL_DIR": f"{CANONICAL_HOME}/models",
        "WORLDFOUNDRY_MODEL_SOURCE_DIR": f"{CANONICAL_HOME}/official_runtime_repos",
        "WORLDFOUNDRY_GITHUB_REPOS_ROOT": f"{CANONICAL_HOME}/official_runtime_repos",
        "WORLDFOUNDRY_CKPT_DIR": f"{CANONICAL_HOME}/checkpoints",
        "WORLDFOUNDRY_HFD_ROOT": f"{CANONICAL_HOME}/checkpoints/hfd",
    }
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from worldfoundry.runtime.inference_catalog import get_model_inference_spec  # noqa: E402
from model_page_mdx import sync_model_pages  # noqa: E402

CATALOG_ROOT = ROOT / "worldfoundry/data/models/catalog"
PROFILE_ROOT = ROOT / "worldfoundry/data/models/runtime/profiles"
ENVIRONMENT_ROOT = ROOT / "worldfoundry/data/models/runtime/environments"
BINDING_ROOT = ROOT / "worldfoundry/data/models/bindings/pipelines"
BENCH_STATUS_PATH = DOCS_ROOT / "lib" / "benchmark-catalog-status.json"

CATEGORY_META = {
    "video": {
        "label": "Video",
        "label_zh": "视频",
        "description": "Video, image, and audio-visual generation or editing runtimes.",
    },
    "multimodal": {
        "label": "Multimodal",
        "label_zh": "多模态",
        "description": "Interleaved text and image generation or understanding runtimes.",
    },
    "world_models": {
        "label": "World models",
        "label_zh": "世界模型",
        "description": "Interactive worlds, prediction, navigation, and simulator-shaped systems.",
    },
    "three_d_four_d": {
        "label": "3D & 4D",
        "label_zh": "3D 与 4D",
        "description": "Reconstruction, geometry, point clouds, scenes, and dynamic representations.",
    },
    "vla_va_wam": {
        "label": "Embodied",
        "label_zh": "具身智能",
        "description": "VLA, vision-action, world-action, and robot-policy runtimes.",
    },
    "hosted_api": {
        "label": "Hosted API",
        "label_zh": "托管 API",
        "description": "Provider-backed models that require credentials or a hosted endpoint.",
    },
}

STATUS_ORDER = {
    "verified": 0,
    "integrated": 1,
    "runtime_ported": 2,
    "profile": 3,
    "planned": 4,
    "blocked": 5,
}

# Docs catalog hides blocked 3D/4D entries.
# Extra IDs are runtime/profile rows of a model that already has a public recipe.
DOCS_EXCLUDED_RECIPE_IDS: set[str] = {
    "dust3r-base-model",
    "dreamx-world-5b-cam",
    "matrix-game-3.5-third-person",
    "minwm-wan-action2v",
    "pi0-worldfoundry",
}


def docs_catalog_visible(category_id: str, model_id: str, status_group: str) -> bool:
    if model_id in DOCS_EXCLUDED_RECIPE_IDS:
        return False
    if category_id == "three_d_four_d" and status_group == "blocked":
        return False
    return True


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text()) or {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        result = str(value).strip()
        return result or None
    return None


def compact_text(value: Any, limit: int = 420) -> str | None:
    result = text(value)
    if not result:
        return None
    result = re.sub(r"\s+", " ", result).strip()
    if len(result) <= limit:
        return result
    return result[: limit - 1].rstrip() + "…"


def portable_doc_value(value: Any) -> Any:
    """Keep checked-in docs independent of the checkout's absolute path."""

    if isinstance(value, dict):
        return {key: portable_doc_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_doc_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(portable_doc_value(item) for item in value)
    if isinstance(value, str):
        replacements = (
            (str(ROOT / "worldfoundry"), CANONICAL_PACKAGE_ROOT),
            (str(ROOT), CANONICAL_PROJECT_ROOT),
            (str(ROOT.parent), ""),
        )
        for source, replacement in replacements:
            value = value.replace(source, replacement)
        return value
    return value


def unique_strings(values: Iterable[Any], limit: int | None = None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = compact_text(value)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        output.append(candidate)
        if limit is not None and len(output) >= limit:
            break
    return output


def humanize(value: str | None) -> str:
    if not value:
        return "Not recorded"
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ")).strip().title()


def model_items(path: Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return [item for item in data["models"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def manifest_index(root: Path, id_keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.yaml")):
        if path.name.startswith("_"):
            continue
        data = load_yaml(path)
        if not isinstance(data, dict):
            continue
        identifiers = [path.stem]
        identifiers.extend(text(data.get(key)) for key in id_keys)
        for identifier in identifiers:
            if identifier:
                output.setdefault(identifier.removeprefix("runtime-profile:"), data)
    return output


def get_status(value: Any) -> str | None:
    if isinstance(value, dict):
        return text(value.get("status"))
    return text(value)


def status_data(item: dict[str, Any], profile: dict[str, Any] | None) -> dict[str, str]:
    integration = (
        get_status(item.get("integration"))
        or text(item.get("integration_status"))
        or text(item.get("status"))
    )
    if not integration and profile:
        execution = profile.get("execution") if isinstance(profile.get("execution"), dict) else {}
        integration = (
            get_status(profile.get("integration"))
            or text(profile.get("integration_status"))
            or text(execution.get("integration_status"))
        )

    runner = get_status(item.get("runner_parity"))
    demo = get_status(item.get("demo_parity"))
    normalized = (integration or "profile_only").lower().replace("-", "_")
    runner_normalized = (runner or "").lower().replace("-", "_")

    if runner_normalized in {"verified", "validated", "passed"}:
        group = "verified"
        label = "Runner verified"
    elif "blocked" in normalized or normalized in {"unavailable", "missing"}:
        group = "blocked"
        label = "Blocked"
    elif normalized in {"planned", "todo", "proposed"}:
        group = "planned"
        label = "Planned"
    elif normalized == "integrated_checkpoint_gated":
        group = "runtime_ported"
        label = "Runtime ported"
    elif normalized in {"integrated", "verified", "ready", "supported"}:
        group = "integrated"
        label = "Integrated"
    elif "ported" in normalized or "runtime" in normalized:
        group = "runtime_ported"
        label = "Runtime ported"
    else:
        group = "profile"
        label = "Profile only"

    return {
        "group": group,
        "label": label,
        "integration": integration or "not_recorded",
        "runner": runner or "not_recorded",
        "demo": demo or "not_recorded",
    }


def profile_candidates(item: dict[str, Any], model_id: str) -> list[str]:
    variants = [variant for variant in as_list(item.get("variants")) if isinstance(variant, dict)]
    candidates: list[Any] = [item.get("runtime_profile")]
    candidates.extend(variant.get("runtime_profile") for variant in variants)
    candidates.extend([model_id, *as_list(item.get("aliases"))])
    return unique_strings(
        candidate.removeprefix("runtime-profile:") if isinstance(candidate, str) else candidate
        for candidate in candidates
    )


def select_manifest(index: dict[str, dict[str, Any]], candidates: list[str]) -> tuple[str | None, dict[str, Any] | None]:
    for candidate in candidates:
        if candidate in index:
            return candidate, index[candidate]
    return None, None


def github_owner(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"github\.com/([^/]+)", url)
    return match.group(1) if match else None


def github_readme_url(url: str | None) -> str | None:
    """Return the stable README anchor for a GitHub repository URL.

    Catalog source links occasionally point at a subdirectory or a revision.
    Model pages should still give readers one predictable starting point for
    the upstream usage instructions, so reduce those URLs to their repository
    root before adding the README anchor.
    """

    if not url:
        return None
    match = re.match(r"https?://(?:www\.)?github\.com/([^/]+)/([^/?#]+)", url)
    if not match:
        return None
    owner, repository = match.groups()
    repository = repository.removesuffix(".git")
    return f"https://github.com/{owner}/{repository}#readme"


def source_links(item: dict[str, Any], profile: dict[str, Any] | None) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(kind: str, label: str, url: Any, revision: Any = None) -> None:
        normalized = text(url)
        if not normalized or not normalized.startswith(("https://", "http://")) or normalized in seen:
            return
        seen.add(normalized)
        entry = {"kind": kind, "label": label, "url": normalized}
        normalized_revision = text(revision)
        if normalized_revision:
            entry["revision"] = normalized_revision
        links.append(entry)

    official = item.get("official_sources") if isinstance(item.get("official_sources"), dict) else {}
    for key, label, kind in [
        ("project_page", "Project", "project"),
        ("paper", "Paper", "paper"),
        ("docs", "Documentation", "docs"),
    ]:
        value = official.get(key)
        if isinstance(value, dict):
            add(kind, label, value.get("url"), value.get("revision"))
        else:
            add(kind, label, value)

    github = official.get("github")
    for record in as_list(github):
        if isinstance(record, dict):
            add("source", "GitHub", record.get("url"), record.get("revision") or record.get("sha"))
        else:
            add("source", "GitHub", record)

    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    for key in ("official_repo_url", "repo_url", "github", "url"):
        value = source.get(key)
        if isinstance(value, dict):
            add("source", "GitHub", value.get("url"), value.get("revision") or value.get("sha"))
        else:
            add("source", "GitHub", value, source.get("revision"))
    add("weights", "Hugging Face", f"https://huggingface.co/{source['hf_repo_id']}" if source.get("hf_repo_id") else None)

    source_status = item.get("source_status") if isinstance(item.get("source_status"), dict) else {}
    github_status = source_status.get("github") if isinstance(source_status.get("github"), dict) else {}
    add("source", "GitHub", github_status.get("url"), github_status.get("head_sha"))

    for record in as_list(official.get("huggingface")):
        if isinstance(record, dict):
            repo_id = record.get("repo_id") or record.get("id")
            add("weights", "Hugging Face", f"https://huggingface.co/{repo_id}" if repo_id else record.get("url"), record.get("revision") or record.get("sha"))
        elif isinstance(record, str):
            add("weights", "Hugging Face", record if record.startswith("http") else f"https://huggingface.co/{record}")

    if profile:
        for record in as_list(profile.get("source_repos")):
            if isinstance(record, dict):
                add("source", "Source", record.get("url"), record.get("revision") or record.get("sha"))

    # Not every catalog manifest has a separately curated documentation URL,
    # but an upstream GitHub repository is still a first-class official usage
    # source. Surface its README explicitly instead of making readers infer
    # that the generic source link contains installation and inference details.
    for source_link in tuple(links):
        if source_link["kind"] != "source":
            continue
        readme_url = github_readme_url(source_link["url"])
        if readme_url:
            add("docs", "Upstream README", readme_url, source_link.get("revision"))

    return links


def checkpoint_data(item: dict[str, Any], profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(record: Any) -> None:
        if not isinstance(record, dict):
            return
        repo_id = text(record.get("repo_id") or record.get("id") or record.get("repo"))
        if not repo_id:
            return
        revision = text(record.get("revision") or record.get("sha")) or ""
        key = (repo_id, revision)
        if key in seen:
            return
        seen.add(key)
        entry: dict[str, Any] = {"id": repo_id}
        for source_key, target_key in [
            ("revision", "revision"),
            ("sha", "revision"),
            ("license", "license"),
            ("role", "role"),
            ("status", "status"),
        ]:
            value = text(record.get(source_key))
            if value and target_key not in entry:
                entry[target_key] = value
        for key_name in ("gated", "private"):
            if isinstance(record.get(key_name), bool):
                entry[key_name] = record[key_name]
        notes = unique_strings(as_list(record.get("notes")), limit=3)
        if notes:
            entry["notes"] = notes
        output.append(entry)

    for record in as_list(item.get("checkpoints")) + as_list(item.get("checkpoint_refs")):
        add(record)
    checkpoint = item.get("checkpoint") if isinstance(item.get("checkpoint"), dict) else {}
    for record in as_list(checkpoint.get("repos")):
        add(record)
    for variant in as_list(item.get("variants")):
        if isinstance(variant, dict):
            for record in as_list(variant.get("checkpoint_refs")):
                add(record)
    if profile:
        for record in as_list(profile.get("checkpoints")):
            add(record)
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    if source.get("hf_repo_id"):
        add({"repo_id": source.get("hf_repo_id"), "license": source.get("license")})
    return output


def provider_name(item: dict[str, Any], links: list[dict[str, str]], checkpoints: list[dict[str, Any]]) -> str:
    # The curated publisher (docs.publisher) is the formal institution name and
    # therefore the best identity to show; catalog-level organization fields
    # come next, before any repo-derived guesses.
    curated = docs_override(item)
    for candidate in (curated.get("publisher"), item.get("publisher"), item.get("organization")):
        normalized = text(candidate)
        if normalized:
            return normalized
    for key in ("developer", "organization", "family"):
        candidate = text(item.get(key))
        if candidate:
            return candidate
    raw_provider = text(item.get("provider"))
    if raw_provider and raw_provider not in {"official_repo", "hosted_api", "pipeline", "local"}:
        return humanize(raw_provider)
    for checkpoint in checkpoints:
        if "/" in checkpoint["id"]:
            return checkpoint["id"].split("/", 1)[0]
    for link in links:
        owner = github_owner(link["url"])
        if owner:
            return owner
    if raw_provider == "hosted_api":
        return "Hosted provider"
    return "Upstream project"


def task_data(item: dict[str, Any], profile: dict[str, Any] | None) -> list[str]:
    values = as_list(item.get("tasks"))
    values.extend(as_list(item.get("task")))
    values.extend(as_list(item.get("capabilities")))
    if not values and profile:
        values.extend(as_list(profile.get("groups")))
        values.extend(as_list(profile.get("task_family")))
    return unique_strings(values)


def variant_data(
    item: dict[str, Any],
    default_status: dict[str, str],
    profiles: dict[str, dict[str, Any]],
    environments: dict[str, dict[str, Any]],
    bindings: dict[str, dict[str, Any]],
    unified_environment: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build variant records with the runtime route that actually serves each variant."""
    output: list[dict[str, Any]] = []
    for variant in as_list(item.get("variants")):
        if not isinstance(variant, dict):
            continue
        variant_id = text(variant.get("id") or variant.get("model_id"))
        if not variant_id:
            continue
        profile_candidates = unique_strings(
            (
                candidate.removeprefix("runtime-profile:") if isinstance(candidate, str) else candidate
                for candidate in (
                    variant.get("runtime_profile"),
                    variant_id,
                    item.get("runtime_profile"),
                    text(item.get("id") or item.get("model_id")),
                    *as_list(item.get("aliases")),
                )
            )
        )
        profile_id, profile = select_manifest(profiles, profile_candidates)
        profile_execution = profile.get("execution") if profile and isinstance(profile.get("execution"), dict) else {}
        profile_integration = get_status(profile.get("integration")) if profile else None
        integration = get_status(variant.get("integration")) or profile_integration
        integration = integration or text(profile_execution.get("integration_status")) or default_status["integration"]
        binding_candidates = unique_strings(
            (
                variant.get("pipeline_binding"),
                profile_execution.get("pipeline_binding"),
                variant_id,
            )
        )
        binding_id, binding = select_manifest(bindings, binding_candidates)
        environment_candidates = unique_strings(
            (
                variant.get("runtime_profile"),
                variant_id,
                profile_id,
                variant.get("pipeline_binding"),
                binding_id,
                item.get("runtime_profile"),
                text(item.get("id") or item.get("model_id")),
            )
        )
        environment_candidates = [candidate.removeprefix("runtime-profile:") for candidate in environment_candidates]
        environment_id, environment = select_manifest(environments, environment_candidates)
        if environment is None and profile is not None:
            environment_id, environment = "_unified", unified_environment
        variant_runtime = runtime_data(
            item,
            profile_id,
            profile,
            environment_id,
            environment,
            binding_id,
            binding,
        )
        pipeline = binding.get("pipeline") if binding and isinstance(binding.get("pipeline"), dict) else {}
        loading = pipeline.get("loading") if isinstance(pipeline.get("loading"), dict) else {}
        invocation = pipeline.get("invocation") if isinstance(pipeline.get("invocation"), dict) else {}
        variant_item = {**item, **variant}
        entry = {
            "id": variant_id,
            "label": text(variant.get("name") or variant.get("display_name")) or variant_id,
            "task": text(variant.get("task")) or "",
            "runtimeProfile": variant_runtime["profileId"] or "",
            "pipelineBinding": variant_runtime["bindingId"] or "",
            "status": integration,
            "pipelineTarget": variant_runtime["pipelineTarget"],
            "runner": variant_runtime["runner"] or variant_runtime["runnerTarget"],
            "loadingMethod": text(loading.get("method")),
            "invocationMode": text(invocation.get("mode")),
            "environmentName": variant_runtime["environmentName"],
            "environmentKind": variant_runtime["environmentKind"],
            "python": variant_runtime["python"],
            "cudaLabel": variant_runtime["cudaLabel"],
            "backendStage": variant_runtime["backendStage"],
            "runtimeStatus": variant_runtime["runtimeStatus"],
            "inputContract": input_contract(profile),
            "artifacts": artifact_data(variant_item, profile),
        }
        output.append(entry)
    return output


def cuda_label(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.lower().replace("-", "_")
    if normalized == "cpu":
        return "CPU"
    if normalized == "prepare_only":
        return "Prepare-only"
    match = re.fullmatch(r"cu(\d{2,3})(_or_newer|_recommended)?", normalized)
    if match:
        digits = match.group(1)
        version = f"{digits[:-1]}.{digits[-1]}"
        if match.group(2) == "_or_newer":
            return f"CUDA {version}+"
        return f"CUDA {version}"
    if normalized.startswith("cuda"):
        return value.upper().replace("CUDA", "CUDA ").replace("  ", " ").strip()
    return humanize(value)


def package_versions(packages: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for package in packages:
        normalized = package.strip()
        name_match = re.match(r"([A-Za-z0-9_.-]+)", normalized)
        if not name_match:
            continue
        name = name_match.group(1).lower().replace("_", "-")
        if name in {"torch", "torchvision", "torchaudio", "diffusers", "transformers", "xfuser", "accelerate", "flash-attn"}:
            output.setdefault(name, normalized)
    return output


def runtime_data(
    item: dict[str, Any],
    profile_id: str | None,
    profile: dict[str, Any] | None,
    env_id: str | None,
    environment: dict[str, Any] | None,
    binding_id: str | None,
    binding: dict[str, Any] | None,
) -> dict[str, Any]:
    execution = profile.get("execution") if profile and isinstance(profile.get("execution"), dict) else {}
    pipeline = binding.get("pipeline") if binding and isinstance(binding.get("pipeline"), dict) else {}
    loading = pipeline.get("loading") if isinstance(pipeline.get("loading"), dict) else {}
    invocation = pipeline.get("invocation") if isinstance(pipeline.get("invocation"), dict) else {}
    pip_packages = unique_strings(as_list(environment.get("pip_packages")) if environment else [])
    conda_packages = unique_strings(as_list(environment.get("conda_packages")) if environment else [])
    env_name = text(environment.get("env_name")) if environment else None
    env_kind = "unrecorded"
    if env_name:
        env_kind = "unified" if "unified" in env_name.lower() else "dedicated"

    return {
        "profileId": profile_id,
        "bindingId": binding_id or text(item.get("pipeline_binding")),
        "runnerTarget": text(item.get("runner_target")),
        "runner": text(binding.get("runner")) if binding else None,
        "pipelineTarget": text(pipeline.get("target") or execution.get("pipeline_target")),
        "loadingMethod": text(loading.get("method")),
        "invocationMode": text(invocation.get("mode")),
        "backendStage": text(execution.get("backend_stage") or (profile.get("backend_stage") if profile else None)),
        "runtimeStatus": text(execution.get("runtime_status") or (profile.get("runtime_status") if profile else None)),
        "environmentId": env_id,
        "environmentName": env_name,
        "environmentKind": env_kind,
        "python": text(environment.get("python")) if environment else None,
        "cudaProfile": text(environment.get("cuda_profile")) if environment else None,
        "cudaLabel": cuda_label(text(environment.get("cuda_profile"))) if environment else None,
        "driverStatus": text(environment.get("driver_status")) if environment else None,
        "condaPackages": conda_packages,
        "pipPackages": pip_packages,
        "packageVersions": package_versions(pip_packages),
        "validationImports": unique_strings(as_list(environment.get("validation_imports")) if environment else []),
        "notes": unique_strings(
            [
                *as_list(execution.get("notes")),
                *as_list(profile.get("notes") if profile else None),
                *as_list(environment.get("notes") if environment else None),
            ],
            limit=12,
        ),
    }


def input_contract(profile: dict[str, Any] | None) -> list[dict[str, str]]:
    schema = profile.get("input_schema") if profile else None
    if not isinstance(schema, dict):
        return []
    output: list[dict[str, str]] = []
    for key, value in schema.items():
        if isinstance(value, bool):
            detail = "Required" if value else "Optional"
        elif isinstance(value, list):
            detail = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            detail = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            detail = text(value) or "Recorded"
        output.append({"field": str(key), "detail": detail})
    return output


def artifact_data(item: dict[str, Any], profile: dict[str, Any] | None) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: Any, filename: Any = None) -> None:
        normalized_kind = text(kind)
        normalized_filename = text(filename) or ""
        if not normalized_kind or (normalized_kind, normalized_filename) in seen:
            return
        seen.add((normalized_kind, normalized_filename))
        output.append({"kind": normalized_kind, "filename": normalized_filename})

    for artifact in as_list(item.get("output_artifacts")):
        if isinstance(artifact, dict):
            add(artifact.get("kind") or artifact.get("type"), artifact.get("path") or artifact.get("filename"))
        else:
            add(artifact)
    if profile:
        add(profile.get("artifact_kind"), profile.get("artifact_filename"))
    for area in (item.get("demo_parity"), item.get("runner_parity")):
        if isinstance(area, dict):
            for artifact in as_list(area.get("expected_artifacts")):
                if isinstance(artifact, dict):
                    add("expected artifact", artifact.get("path"))
    return output


def task_field_detail(field: Any) -> str:
    required = "Required" if getattr(field, "required", False) else "Optional"
    pieces = [required]
    default = getattr(field, "default", None)
    if default is not None:
        if isinstance(default, (dict, list, tuple)):
            rendered = json.dumps(default, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(default)
        pieces.append(f"default={rendered}")
    choices = list(getattr(field, "choices", ()) or ())
    if choices:
        pieces.append(f"choices={', '.join(str(choice) for choice in choices)}")
    return "; ".join(pieces)


def core_inference_task_data(spec: Any, variant_ids: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in getattr(spec, "tasks", ()):
        inputs = []
        for field in task.inputs:
            inputs.append(
                {
                    "field": field.field_id,
                    "detail": task_field_detail(field),
                    "kind": field.kind,
                    "target": field.target,
                    "required": bool(field.required),
                    "default": field.default,
                    "choices": list(field.choices),
                    "description": field.description,
                }
            )
        artifacts = [
            {
                "kind": artifact.kind,
                "filename": artifact.artifact_id,
                "description": artifact.description,
            }
            for artifact in task.outputs
        ]
        output.append(
            {
                "id": task.task_id,
                "label": task.label,
                "description": task.description,
                "source": "inference_spec",
                "variantIds": list(variant_ids),
                "inputs": inputs,
                "artifacts": artifacts,
            }
        )
    return output


def catalog_inference_task_data(
    item: dict[str, Any],
    profile: dict[str, Any] | None,
    variant_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose task profiles for models without a curated Python inference spec.

    Catalog ``tasks`` are capabilities, while a variant's ``task`` is the
    closest executable profile when variants exist. Keep the raw catalog
    values for fallback model-run schemas, but bind them to the variants that
    declare them so the docs cannot pair an arbitrary task with a variant.
    """
    declared_tasks = (
        unique_strings(record.get("task") for record in variant_records if record.get("task"))
        if variant_records
        else task_data(item, profile)
    )
    if not declared_tasks:
        declared_tasks = ["default"]

    output: list[dict[str, Any]] = []
    for task_id in declared_tasks:
        matching = [record for record in variant_records if record.get("task") == task_id]
        source_record = matching[0] if matching else (variant_records[0] if variant_records else None)
        fields = source_record.get("inputContract", []) if source_record else input_contract(profile)
        inputs = [
            {
                "field": field.get("field", ""),
                "detail": field.get("detail", "Recorded"),
                "kind": "string",
                "target": "input",
                "required": field.get("detail") == "Required",
                "description": "Recorded input field from the runtime profile.",
            }
            for field in fields
            if field.get("field")
        ]
        artifacts = source_record.get("artifacts", []) if source_record else artifact_data(item, profile)
        output.append(
            {
                "id": task_id,
                "label": humanize(task_id),
                "description": "Catalog task profile used by the model runtime.",
                "source": "catalog",
                "variantIds": [record.get("id") for record in matching if record.get("id")],
                "inputs": inputs,
                "artifacts": [
                    {
                        "kind": artifact.get("kind", ""),
                        "filename": artifact.get("filename", ""),
                    }
                    for artifact in artifacts
                    if artifact.get("kind") or artifact.get("filename")
                ],
            }
        )
    return output


def inference_task_data(
    item: dict[str, Any],
    profile: dict[str, Any] | None,
    variant_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model_id = text(item.get("id") or item.get("model_id")) or ""
    spec = get_model_inference_spec(model_id)
    if spec is None:
        alias_specs = {
            candidate.model_family_id: candidate
            for alias in as_list(item.get("aliases"))
            if alias
            for candidate in [get_model_inference_spec(str(alias))]
            if candidate is not None
        }
        if len(alias_specs) == 1:
            spec = next(iter(alias_specs.values()))
    if spec is not None:
        return core_inference_task_data(spec, [record["id"] for record in variant_records])
    return catalog_inference_task_data(item, profile, variant_records)


def recipe_notes(item: dict[str, Any], profile: dict[str, Any] | None) -> list[str]:
    integration = item.get("integration") if isinstance(item.get("integration"), dict) else {}
    runner = item.get("runner_parity") if isinstance(item.get("runner_parity"), dict) else {}
    demo = item.get("demo_parity") if isinstance(item.get("demo_parity"), dict) else {}
    checkpoint = item.get("checkpoint") if isinstance(item.get("checkpoint"), dict) else {}
    return unique_strings(
        [
            *as_list(item.get("notes")),
            *as_list(integration.get("notes")),
            *as_list(runner.get("notes")),
            *as_list(demo.get("notes")),
            *as_list(checkpoint.get("notes")),
            *as_list(profile.get("notes") if profile else None),
        ],
        limit=18,
    )


# ---------------------------------------------------------------------------
# Model homepage docs block: curated ``docs:`` override + synthesized fallback
# ---------------------------------------------------------------------------

DOCS_CATEGORY_NOUNS = {
    "video": ("video generation model", "视频生成模型"),
    "multimodal": ("multimodal text and image model", "图文多模态模型"),
    "world_models": ("world model", "世界模型"),
    "three_d_four_d": ("3D/4D reconstruction and generation model", "3D/4D 重建与生成模型"),
    "vla_va_wam": ("embodied vision-language-action model", "具身智能（视觉-语言-动作）模型"),
    "hosted_api": ("hosted, provider-backed model", "托管 API 模型"),
}

TASK_PHRASES_ZH = {
    "text-to-video": "文生视频",
    "image-to-video": "图生视频",
    "video-to-video": "视频到视频转换",
    "text-to-image": "文生图",
    "image-to-image": "图像编辑",
    "text-image-to-video": "图文联合生成视频",
    "reference-to-video": "参考图生成视频",
    "reference-video-to-video": "参考视频转换",
    "audio-video-generation": "音视频联合生成",
    "video-generation": "视频生成",
    "long-video-generation": "长视频生成",
    "autoregressive-video-generation": "自回归视频生成",
    "interactive-video-generation": "交互式视频生成",
    "camera-controlled-video": "相机可控视频生成",
    "camera-control": "相机控制",
    "camera_control": "相机控制",
    "depth-controlled-video": "深度可控视频生成",
    "trajectory-controlled-video": "轨迹可控视频生成",
    "action-conditioned-video": "动作条件视频生成",
    "interactive-world-model": "交互式世界模型",
    "world-model": "世界模型",
    "world_model": "世界模型",
    "world": "世界模型",
    "world-generation": "世界生成",
    "3d-world-generation": "3D 世界生成",
    "image-to-3d-world": "图像生成 3D 世界",
    "robot-world-model": "机器人世界模型",
    "robotics-world-model": "机器人世界模型",
    "game-world-model": "游戏世界模型",
    "minecraft-world-model": "Minecraft 世界模型",
    "multi-agent-world-model": "多智能体世界模型",
    "embodied-world-model": "具身世界模型",
    "diffusion-world-model": "扩散世界模型",
    "vla": "视觉-语言-动作（VLA）",
    "vla.policy_rollout": "VLA 策略执行",
    "vla.action_prediction": "VLA 动作预测",
    "robot_policy": "机器人策略",
    "policy_rollout": "策略执行",
    "embodied_policy": "具身策略",
    "embodied_benchmark": "具身评测",
    "wam": "世界-动作模型（WAM）",
    "wam.world_action_modeling": "世界-动作建模",
    "3d-reconstruction": "3D 重建",
    "geometry-prior": "几何先验",
    "geometry": "几何估计",
    "novel-view-synthesis": "新视角合成",
    "gaussian-splatting": "高斯泼溅（Gaussian Splatting）",
    "point-cloud": "点云",
    "metric-depth-estimation": "米制深度估计",
    "monocular-depth-estimation": "单目深度估计",
    "panoramic-depth-estimation": "全景深度估计",
    "depth": "深度估计",
    "trajectory": "轨迹预测",
    "navigation": "导航",
    "memory-research": "记忆机制研究",
    "multimodal-reasoning": "多模态推理",
    "image-question-answering": "图像问答",
    "video-question-answering": "视频问答",
    "hosted-api": "托管 API 推理",
    "video": "视频生成",
}

MODALITY_TOKEN_MAP = {
    "text": "text",
    "prompt": "text",
    "image": "image",
    "images": "image",
    "reference": "image",
    "video": "video",
    "audio": "audio",
    "action": "action",
    "actions": "action",
    "depth": "depth",
    "trajectory": "trajectory",
    "camera": "camera pose",
    "3d": "3D scene",
    "4d": "4D scene",
    "world": "world state",
    "mesh": "mesh",
    "pointcloud": "point cloud",
    "point-cloud": "point cloud",
}

MODALITY_LABELS_ZH = {
    "text": "文本",
    "image": "图像",
    "video": "视频",
    "audio": "音频",
    "action": "动作",
    "depth": "深度",
    "trajectory": "轨迹",
    "camera pose": "相机位姿",
    "3D scene": "3D 场景",
    "4D scene": "4D 场景",
    "world state": "世界状态",
    "mesh": "网格",
    "point cloud": "点云",
}


def load_benchmark_status() -> dict[str, dict[str, Any]]:
    if not BENCH_STATUS_PATH.exists():
        return {}
    data = json.loads(BENCH_STATUS_PATH.read_text())
    return data if isinstance(data, dict) else {}


BENCH_STATUS = load_benchmark_status()


def load_paper_media() -> dict[str, dict[str, Any]]:
    if not PAPER_MEDIA_PATH.exists():
        return {}
    data = json.loads(PAPER_MEDIA_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


PAPER_MEDIA = load_paper_media()


def paper_media_for(model_id: str, docs: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    extra = PAPER_MEDIA.get(model_id) if isinstance(PAPER_MEDIA.get(model_id), dict) else {}
    paper = extra.get("paper") if isinstance(extra.get("paper"), dict) else None
    curated_paper = docs.get("paper") if isinstance(docs.get("paper"), dict) else None
    if curated_paper:
        paper = {**(paper or {}), **curated_paper}
    figures = extra.get("figures") if isinstance(extra.get("figures"), list) else []
    curated_figures = docs.get("figures") if isinstance(docs.get("figures"), list) else []
    if curated_figures:
        figures = curated_figures
    cleaned_figures = [
        item
        for item in figures
        if isinstance(item, dict)
        and item.get("src")
        and item.get("kind") != "video"
        and not str(item.get("src") or "").lower().endswith((".mp4", ".webm", ".mov", ".m4v"))
    ]
    return paper, cleaned_figures


def bench_scan_patterns() -> list[tuple[str, "re.Pattern[str]"]]:
    """One conservative word-boundary pattern per benchmark id.

    Short names without digits stay case-sensitive so generic words
    (e.g. a benchmark literally named "MinD") cannot match prose.
    """
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for bench_id, entry in BENCH_STATUS.items():
        names = unique_strings([entry.get("name"), *as_list(entry.get("aliases"))])
        for candidate in names:
            if len(candidate) < 4:
                continue
            flags = 0 if len(candidate) < 6 and not any(ch.isdigit() for ch in candidate) else re.IGNORECASE
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(candidate)}(?![A-Za-z0-9])", flags)
            patterns.append((bench_id, pattern))
    return patterns


BENCH_SCAN_PATTERNS = bench_scan_patterns()


def docs_override(item: dict[str, Any]) -> dict[str, Any]:
    """Merge the optional curated ``docs:`` / ``homepage:`` blocks."""
    merged: dict[str, Any] = {}
    for key in ("homepage", "docs"):
        block = item.get(key)
        if isinstance(block, dict):
            merged.update(block)
    return merged


def evidence_texts(item: dict[str, Any]) -> list[str]:
    """Evidence-adjacent strings a benchmark name may legitimately appear in."""
    texts: list[str] = []

    def collect(record: Any) -> None:
        if isinstance(record, dict):
            for value in as_list(record.get("notes")):
                candidate = text(value)
                if candidate:
                    texts.append(candidate)

    for value in as_list(item.get("notes")):
        candidate = text(value)
        if candidate:
            texts.append(candidate)
    for key in ("integration", "runner_parity", "demo_parity"):
        collect(item.get(key))
    for variant in as_list(item.get("variants")):
        if isinstance(variant, dict):
            for value in as_list(variant.get("notes")):
                candidate = text(value)
                if candidate:
                    texts.append(candidate)
            for key in ("integration", "runner_parity", "demo_parity"):
                collect(variant.get(key))
    return texts


def parity_note_quotes(item: dict[str, Any], limit: int = 4) -> list[str]:
    quotes: list[str] = []
    for key in ("runner_parity", "demo_parity"):
        record = item.get(key)
        if isinstance(record, dict):
            quotes.extend(as_list(record.get("notes")))
    for variant in as_list(item.get("variants")):
        if isinstance(variant, dict):
            for key in ("runner_parity", "demo_parity"):
                record = variant.get(key)
                if isinstance(record, dict):
                    quotes.extend(as_list(record.get("notes")))
    return unique_strings(quotes, limit=limit)


def benchmark_refs(item: dict[str, Any], docs: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(bench_id: str, source: str, reason: str | None, reason_zh: str | None) -> None:
        entry = BENCH_STATUS.get(bench_id)
        if entry is None or bench_id in seen:
            if entry is None and source == "docs":
                print(f"warning: docs.recommended_benchmarks references unknown benchmark id '{bench_id}'", file=sys.stderr)
            return
        seen.add(bench_id)
        refs.append(
            {
                "id": bench_id,
                "name": entry.get("name") or bench_id,
                "category": entry.get("category") or "",
                "categoryZh": entry.get("categoryZh") or entry.get("category") or "",
                "summary": entry.get("summary") or "",
                "summaryZh": entry.get("summaryZh") or entry.get("summary") or "",
                "href": f"/docs/evaluation/benchmark-hub/{bench_id}",
                "source": source,
                "reason": reason or "Recommended in this model's catalog manifest.",
                "reasonZh": reason_zh or reason or "模型 catalog manifest 中推荐的评测。",
            }
        )

    for record in as_list(docs.get("recommended_benchmarks")):
        if isinstance(record, dict):
            bench_id = text(record.get("id"))
            if bench_id:
                add(bench_id, "docs", compact_text(record.get("reason")), compact_text(record.get("reason_zh")))
        elif text(record):
            add(str(text(record)), "docs", None, None)

    corpus = "\n".join(evidence_texts(item))
    if corpus:
        for bench_id, pattern in BENCH_SCAN_PATTERNS:
            if bench_id in seen:
                continue
            if pattern.search(corpus):
                add(
                    bench_id,
                    "manifest",
                    "Referenced in this model's manifest evidence notes.",
                    "该模型 manifest 的证据记录中提到了这一评测。",
                )
    return refs[:6]


def find_min_vram(value: Any) -> int | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "min_vram_gb":
                try:
                    return int(nested)
                except (TypeError, ValueError):
                    continue
            found = find_min_vram(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = find_min_vram(nested)
            if found is not None:
                return found
    return None


def hardware_data(item: dict[str, Any], docs: dict[str, Any]) -> dict[str, Any]:
    block = docs.get("hardware") if isinstance(docs.get("hardware"), dict) else {}
    min_vram = None
    if block.get("min_vram_gb") is not None:
        try:
            min_vram = int(block["min_vram_gb"])
        except (TypeError, ValueError):
            min_vram = None
    if min_vram is None:
        min_vram = find_min_vram(item)
    return {
        "minVramGb": min_vram,
        "recommended": compact_text(block.get("recommended")),
        "notes": unique_strings(as_list(block.get("notes")), limit=4),
    }


def modality_data(tasks: list[str], docs: dict[str, Any]) -> dict[str, list[str]]:
    block = docs.get("modalities") if isinstance(docs.get("modalities"), dict) else {}
    curated_inputs = unique_strings(as_list(block.get("inputs")))
    curated_outputs = unique_strings(as_list(block.get("outputs")))
    if curated_inputs or curated_outputs:
        return {"inputs": curated_inputs, "outputs": curated_outputs}

    inputs: list[str] = []
    outputs: list[str] = []

    def push(collection: list[str], token: str) -> None:
        label = MODALITY_TOKEN_MAP.get(token)
        if label and label not in collection:
            collection.append(label)

    for task in tasks:
        normalized = task.lower().replace("_", "-")
        if "-to-" in normalized:
            left, _, right = normalized.partition("-to-")
            for token in left.split("-"):
                push(inputs, token)
            for token in right.split("-"):
                push(outputs, token)
        elif normalized.endswith("video-generation") or normalized in {"video", "video-generation"}:
            push(outputs, "video")
        elif "depth" in normalized:
            push(inputs, "image")
            push(outputs, "depth")
        elif "point-cloud" in normalized or "pointcloud" in normalized:
            push(outputs, "point-cloud")
    return {"inputs": inputs, "outputs": outputs}


def task_phrase_en(task: str) -> str:
    # Keep hyphenated task ids readable ("text-to-video"), only soften
    # underscores and namespace dots ("vla.policy_rollout" -> "vla policy rollout").
    return task.replace("_", " ").replace(".", " ").strip().lower()


def task_phrase_zh(task: str) -> str:
    # Unmapped task ids stay as the recorded identifier: an accurate English
    # term reads better in a Chinese sentence than a fabricated translation.
    return TASK_PHRASES_ZH.get(task) or TASK_PHRASES_ZH.get(task.lower()) or task


def join_en(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def join_zh(items: list[str]) -> str:
    return "、".join(items)


def runnable_variant_ids(variant_records: list[dict[str, Any]]) -> list[str]:
    unavailable = ("not_recorded", "planned", "profile", "blocked", "unavailable", "missing")
    output = []
    for record in variant_records:
        status = str(record.get("status") or "").lower().replace("-", "_")
        if record.get("pipelineTarget") and not any(marker in status for marker in unavailable):
            output.append(record["id"])
    return output


def synthesize_overview(
    item: dict[str, Any],
    category_id: str,
    name: str,
    provider: str,
    tasks: list[str],
    aliases: list[str],
    status: dict[str, str],
    runtime: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Compose the EN and ZH homepage narrative from recorded facts only."""
    noun_en, noun_zh = DOCS_CATEGORY_NOUNS.get(category_id, ("model", "模型"))
    en: list[str] = []
    zh: list[str] = []

    # Identity and recorded capabilities.
    identity_en = f"{name} is a {noun_en} from {provider}."
    identity_zh = f"{name} 是来自 {provider} 的{noun_zh}。"
    if tasks:
        task_list_en = join_en([task_phrase_en(task) for task in tasks[:4]])
        task_list_zh = join_zh([task_phrase_zh(task) for task in tasks[:4]])
        identity_en += f" The catalog records it for {task_list_en}."
        identity_zh += f"Catalog 将它记录为{task_list_zh}任务。"
    if aliases:
        identity_en += f" It is also cataloged under the alias{'es' if len(aliases) > 1 else ''} {join_en(aliases[:3])}."
        identity_zh += f"它还有别名 {join_zh(aliases[:3])}。"
    en.append(identity_en)
    zh.append(identity_zh)

    # WorldFoundry integration route.
    runnable_ids = runnable_variant_ids(variant_records)
    pipeline_target = runtime.get("pipelineTarget")
    runner = runtime.get("runner") or runtime.get("runnerTarget")
    if runnable_ids:
        route_en = (
            f"WorldFoundry serves it through {len(runnable_ids)} recorded runnable "
            f"variant{'s' if len(runnable_ids) > 1 else ''} ({join_en(runnable_ids[:4])})."
        )
        route_zh = f"WorldFoundry 通过 {len(runnable_ids)} 个已记录的可运行 variant（{join_zh(runnable_ids[:4])}）来运行它。"
        if pipeline_target:
            route_en += f" The default route binds to the {pipeline_target} pipeline."
            route_zh += f"默认路径绑定到 {pipeline_target} pipeline。"
        en.append(route_en)
        zh.append(route_zh)
    elif pipeline_target:
        route_en = f"WorldFoundry records a binding to the {pipeline_target} pipeline"
        route_zh = f"WorldFoundry 记录了到 {pipeline_target} pipeline 的绑定"
        if runner:
            route_en += f" via {runner}"
            route_zh += f"（runner：{runner}）"
        if status["group"] in {"planned", "profile", "runtime_ported", "blocked"}:
            route_en += "; this binding alone does not establish a successful run"
            route_zh += "；仅有绑定记录并不代表已成功运行"
        en.append(route_en + ".")
        zh.append(route_zh + "。")
    else:
        en.append(
            "No runnable WorldFoundry pipeline is bound to this entry yet; "
            "this page records upstream provenance and readiness state only."
        )
        zh.append("该条目目前尚未绑定可运行的 WorldFoundry pipeline；本页只记录上游来源与就绪状态。")

    # Environment.
    env_name = runtime.get("environmentName")
    if env_name:
        env_kind = runtime.get("environmentKind")
        kind_en = "dedicated" if env_kind == "dedicated" else "shared unified"
        kind_zh = "独立" if env_kind == "dedicated" else "统一"
        env_en = f"It runs in the {kind_en} environment {env_name}"
        env_zh = f"它运行在{kind_zh}环境 {env_name} 中"
        details_en = []
        details_zh = []
        if runtime.get("python"):
            details_en.append(f"Python {runtime['python']}")
            details_zh.append(f"Python {runtime['python']}")
        if runtime.get("cudaLabel"):
            details_en.append(str(runtime["cudaLabel"]))
            details_zh.append(str(runtime["cudaLabel"]))
        torch_pin = runtime.get("packageVersions", {}).get("torch")
        if torch_pin and torch_pin != "torch":
            details_en.append(torch_pin)
            details_zh.append(torch_pin)
        if details_en:
            env_en += f" ({', '.join(details_en)})"
            env_zh += f"（{'，'.join(details_zh)}）"
        en.append(env_en + ".")
        zh.append(env_zh + "。")

    # Weights.
    if checkpoints:
        first = checkpoints[0]
        weight_en = f"Weights are pulled from the Hugging Face repository {first['id']}"
        weight_zh = f"权重来自 Hugging Face 仓库 {first['id']}"
        if first.get("revision"):
            weight_en += f", pinned to revision {first['revision'][:9]}"
            weight_zh += f"，固定在 revision {first['revision'][:9]}"
        if first.get("license"):
            weight_en += f" (license: {first['license']})"
            weight_zh += f"（license：{first['license']}）"
        if len(checkpoints) > 1:
            weight_en += f", plus {len(checkpoints) - 1} more recorded repositor{'ies' if len(checkpoints) > 2 else 'y'}"
            weight_zh += f"，另有 {len(checkpoints) - 1} 个已记录仓库"
        gated = [checkpoint["id"] for checkpoint in checkpoints if checkpoint.get("gated")]
        weight_en += "."
        weight_zh += "。"
        if gated:
            weight_en += f" Access to {join_en(gated[:2])} is gated and requires accepting the upstream terms."
            weight_zh += f"其中 {join_zh(gated[:2])} 为 gated 仓库，需要先在上游接受使用条款。"
        en.append(weight_en)
        zh.append(weight_zh)

    # Evidence honesty.
    runner_status = status.get("runner", "not_recorded")
    demo_status = status.get("demo", "not_recorded")
    if runner_status.lower().replace("-", "_") in {"verified", "validated", "passed"}:
        en.append("A WorldFoundry runner-parity artifact is recorded for this route.")
        zh.append("该路径已记录 WorldFoundry runner parity 产物。")
    elif runner_status != "not_recorded" or demo_status != "not_recorded":
        en.append(
            f"Runner parity is currently \u201c{humanize(runner_status)}\u201d and native-demo parity is "
            f"\u201c{humanize(demo_status)}\u201d; treat end-to-end GPU evidence as not yet recorded."
        )
        zh.append(
            f"当前 runner parity 为“{humanize(runner_status)}”，原生 demo parity 为“{humanize(demo_status)}”；"
            "端到端 GPU 证据应视为尚未记录。"
        )
    else:
        en.append(
            "No runner or native-demo evidence is recorded; a catalog entry alone is not proof of a successful GPU run."
        )
        zh.append("尚未记录任何 runner 或原生 demo 证据；仅有 catalog 条目并不代表 GPU 已成功跑通。")

    return en, zh


def synthesize_highlights(
    category_id: str,
    tasks: list[str],
    status: dict[str, str],
    runtime: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    en: list[str] = []
    zh: list[str] = []
    for task in tasks[:3]:
        en.append(humanize(task))
        zh.append(task_phrase_zh(task))
    runnable_ids = runnable_variant_ids(variant_records)
    if runnable_ids:
        en.append(f"{len(runnable_ids)} runnable WorldFoundry variant{'s' if len(runnable_ids) > 1 else ''}")
        zh.append(f"{len(runnable_ids)} 个可运行的 WorldFoundry variant")
    if runtime.get("environmentName"):
        kind = "Dedicated env" if runtime.get("environmentKind") == "dedicated" else "Unified env"
        kind_zh = "独立环境" if runtime.get("environmentKind") == "dedicated" else "统一环境"
        detail = " · ".join(
            part for part in (f"Python {runtime['python']}" if runtime.get("python") else None, runtime.get("cudaLabel")) if part
        )
        en.append(f"{kind}{f' · {detail}' if detail else ''}")
        zh.append(f"{kind_zh}{f' · {detail}' if detail else ''}")
    if checkpoints:
        public = all(not checkpoint.get("gated") and not checkpoint.get("private") for checkpoint in checkpoints)
        en.append("Public Hugging Face weights" if public else "Gated or private weights — check access first")
        zh.append("公开的 Hugging Face 权重" if public else "权重为 gated/私有——请先确认访问权限")
    return en[:6], zh[:6]


def synthesize_limitations(
    item: dict[str, Any],
    status: dict[str, str],
    runtime: dict[str, Any],
    checkpoints: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    en: list[str] = []
    zh: list[str] = []
    group = status["group"]
    if group == "blocked":
        en.append("This entry is currently blocked; see the manifest notes below for the recorded blocker.")
        zh.append("该条目当前处于 blocked 状态；具体 blocker 见下方 manifest 记录。")
    elif group in {"planned", "profile"}:
        if runtime.get("pipelineTarget"):
            en.append("A pipeline binding is recorded, but a successful run has not been established for this entry.")
            zh.append("该条目已记录 pipeline 绑定，但尚未建立成功运行的证据。")
        else:
            en.append(
                "No runnable WorldFoundry route exists yet. The entry records provenance and readiness only; "
                "run commands are intentionally omitted."
            )
            zh.append("尚无可运行的 WorldFoundry 路径。该条目只记录来源与就绪状态，因此有意省略了运行命令。")
    runner_status = status.get("runner", "not_recorded").lower().replace("-", "_")
    if runner_status not in {"verified", "validated", "passed"}:
        if runner_status == "not_recorded":
            en.append("Runner evidence: Not recorded. No verified end-to-end WorldFoundry GPU artifact exists for this model.")
            zh.append("Runner 证据：未记录。该模型还没有经过验证的 WorldFoundry 端到端 GPU 产物。")
        else:
            en.append(
                f"Runner evidence is \u201c{humanize(status['runner'])}\u201d — not a verified end-to-end GPU run. "
                "Do not treat this page as proof the route is fully validated."
            )
            zh.append(
                f"Runner 证据为“{humanize(status['runner'])}”——这不是经过验证的端到端 GPU 运行结果，"
                "请勿将本页视为该路径已被完整验证的证明。"
            )
    demo_status = status.get("demo", "not_recorded").lower().replace("-", "_")
    if demo_status == "not_recorded":
        en.append("Native demo evidence: Not recorded.")
        zh.append("原生 Demo 证据：未记录。")
    gated = [checkpoint["id"] for checkpoint in checkpoints if checkpoint.get("gated")]
    if gated:
        en.append(f"Checkpoint access is gated for {join_en(gated[:3])}; accept the upstream terms before downloading.")
        zh.append(f"Checkpoint {join_zh(gated[:3])} 为 gated 仓库，下载前需要先在上游接受条款。")
    private = [checkpoint["id"] for checkpoint in checkpoints if checkpoint.get("private")]
    if private:
        en.append(f"Recorded checkpoint repositories are private: {join_en(private[:3])}.")
        zh.append(f"以下 checkpoint 仓库为私有：{join_zh(private[:3])}。")
    quotes = parity_note_quotes(item)
    en.extend(quotes)
    zh.extend(quotes)
    return en[:8], zh[:8]


def docs_data(
    item: dict[str, Any],
    category_id: str,
    name: str,
    provider: str,
    tasks: list[str],
    aliases: list[str],
    status: dict[str, str],
    runtime: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
    model_id: str = "",
) -> dict[str, Any]:
    docs = docs_override(item)
    curated = bool(docs)

    def paragraphs(value: Any) -> list[str]:
        if isinstance(value, list):
            return unique_strings(value, limit=6)
        normalized = compact_text(value, 1200)
        if not normalized:
            return []
        return [part.strip() for part in re.split(r"\n\s*\n", str(value).strip()) if part.strip()] or [normalized]

    overview = paragraphs(docs.get("overview"))
    overview_zh = paragraphs(docs.get("overview_zh"))
    synthesized_en, synthesized_zh = synthesize_overview(
        item, category_id, name, provider, tasks, aliases, status, runtime, checkpoints, variant_records
    )
    if not overview:
        overview = synthesized_en
    if not overview_zh:
        overview_zh = overview if curated and paragraphs(docs.get("overview")) else synthesized_zh

    highlights = unique_strings(as_list(docs.get("highlights")), limit=6)
    highlights_zh = unique_strings(as_list(docs.get("highlights_zh")), limit=6)
    if not highlights or not highlights_zh:
        synthesized_h_en, synthesized_h_zh = synthesize_highlights(
            category_id, tasks, status, runtime, checkpoints, variant_records
        )
        highlights = highlights or synthesized_h_en
        highlights_zh = highlights_zh or synthesized_h_zh

    limitations = unique_strings(as_list(docs.get("limitations")), limit=8)
    limitations_zh = unique_strings(as_list(docs.get("limitations_zh")), limit=8)
    if not limitations or not limitations_zh:
        synthesized_l_en, synthesized_l_zh = synthesize_limitations(item, status, runtime, checkpoints)
        limitations = limitations or synthesized_l_en
        limitations_zh = limitations_zh or synthesized_l_zh

    publisher_name = compact_text(docs.get("publisher") or item.get("publisher") or item.get("organization"))
    publisher_kind = text(docs.get("publisher_kind"))
    if publisher_kind and publisher_kind not in {"company", "university", "lab"}:
        print(f"warning: docs.publisher_kind '{publisher_kind}' is not company/university/lab", file=sys.stderr)
        publisher_kind = None
    publisher = (
        {
            "name": publisher_name,
            "nameZh": compact_text(docs.get("publisher_zh")) or publisher_name,
            "kind": publisher_kind,
        }
        if publisher_name
        else None
    )

    paper, figures = paper_media_for(model_id or text(item.get("id")), docs)

    return {
        "curated": curated,
        "publisher": publisher,
        "overview": overview,
        "overviewZh": overview_zh,
        "architecture": paragraphs(docs.get("architecture")),
        "architectureZh": paragraphs(docs.get("architecture_zh")),
        "usageNotes": paragraphs(docs.get("usage_notes")),
        "usageNotesZh": paragraphs(docs.get("usage_notes_zh")),
        "highlights": highlights,
        "highlightsZh": highlights_zh,
        "modalities": modality_data(tasks, docs),
        "useCases": unique_strings(as_list(docs.get("use_cases")), limit=6),
        "useCasesZh": unique_strings(as_list(docs.get("use_cases_zh")), limit=6),
        "hardware": hardware_data(item, docs),
        "benchmarks": benchmark_refs(item, docs),
        "limitations": limitations,
        "limitationsZh": limitations_zh,
        "paper": paper,
        "figures": figures,
    }


def summary_for(item: dict[str, Any], tasks: list[str], notes: list[str]) -> str:
    for candidate in [item.get("description"), item.get("summary")]:
        normalized = compact_text(candidate, 240)
        if normalized:
            return normalized
    if notes:
        return compact_text(notes[0], 240) or notes[0]
    if tasks:
        labels = ", ".join(humanize(task).lower() for task in tasks[:3])
        return f"Manifested for {labels}. Open the recipe to inspect runtime and provenance records."
    return "Cataloged in WorldFoundry. Open the recipe to inspect the available runtime and provenance records."


def command_placeholder(field: dict[str, Any]) -> str | None:
    if not field.get("required") or field.get("default") not in (None, ""):
        return None
    field_id = str(field.get("field") or "")
    if field_id in {"prompt", "instruction", "text", "caption"}:
        return '"Describe the desired output."'
    if field_id in {"input_path", "image", "video", "audio", "images", "input"}:
        return "/path/to/input"
    if field.get("kind") in {"json", "interaction_tokens"}:
        return "'{}'"
    return "VALUE"


def direct_run_command(runtime_model_id: str, task: dict[str, Any]) -> str:
    lines = [
        "worldfoundry-eval run \\",
        f"  {shlex.quote(runtime_model_id)} \\",
        f"  --pipeline.task-profile {shlex.quote(str(task['id']))} \\",
    ]
    for field in task.get("inputs", []):
        placeholder = command_placeholder(field)
        if placeholder is None:
            continue
        option = str(field.get("field") or "").replace("_", "-")
        lines.append(f"  --pipeline.{option} {placeholder} \\")
    lines.append("  --json")
    return "\n".join(lines)


def command_data(model_id: str, runtime_model_id: str, default_task: dict[str, Any]) -> dict[str, str]:
    return {
        "prepare": f"bash scripts/inference/prepare_model_infer.sh {runtime_model_id}",
        "install": f"bash scripts/setup/model_env_install.sh --model {runtime_model_id}",
        "inspect": f"worldfoundry-eval zoo model-show --model-id {model_id} --include-manifest --json",
        "check": f"worldfoundry-eval zoo model-download --model-id {model_id} --check-local --json",
        "run": direct_run_command(runtime_model_id, default_task),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when generated recipe data or generated model MDX pages are stale.",
    )
    args = parser.parse_args()

    profiles = manifest_index(PROFILE_ROOT, ("model_id", "id"))
    environments = manifest_index(ENVIRONMENT_ROOT, ("model_id", "id"))
    bindings = manifest_index(BINDING_ROOT, ("binding_id", "model_id", "id"))
    unified_environment = load_yaml(ENVIRONMENT_ROOT / "_unified.yaml")

    recipes: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()

    for category_id, category in CATEGORY_META.items():
        category_root = CATALOG_ROOT / category_id
        for path in sorted(category_root.glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            for item in model_items(path):
                model_id = text(item.get("id") or item.get("model_id"))
                if not model_id:
                    continue
                name = text(item.get("name") or item.get("display_name")) or model_id
                candidates = profile_candidates(item, model_id)
                profile_id, profile = select_manifest(profiles, candidates)

                variants = [variant for variant in as_list(item.get("variants")) if isinstance(variant, dict)]
                env_candidates = unique_strings(
                    [
                        *(variant.get("runtime_profile") for variant in variants),
                        *(variant.get("id") for variant in variants),
                        *candidates,
                    ]
                )
                env_candidates = [candidate.removeprefix("runtime-profile:") for candidate in env_candidates]
                env_id, environment = select_manifest(environments, env_candidates)
                if environment is None and profile is not None:
                    env_id, environment = "_unified", unified_environment

                binding_candidates = unique_strings(
                    [
                        item.get("pipeline_binding"),
                        *(variant.get("pipeline_binding") for variant in variants),
                        model_id,
                    ]
                )
                binding_id, binding = select_manifest(bindings, binding_candidates)

                status = status_data(item, profile)
                tasks = task_data(item, profile)
                links = source_links(item, profile)
                checkpoints = checkpoint_data(item, profile)
                notes = recipe_notes(item, profile)
                variant_records = variant_data(item, status, profiles, environments, bindings, unified_environment)
                inference_tasks = inference_task_data(item, profile, variant_records)
                runtime_model_id = variant_records[0]["id"] if variant_records else model_id
                aliases = unique_strings(as_list(item.get("aliases")))
                provider = provider_name(item, links, checkpoints)
                runtime = runtime_data(item, profile_id, profile, env_id, environment, binding_id, binding)
                docs = docs_data(
                    item,
                    category_id,
                    name,
                    provider,
                    tasks,
                    aliases,
                    status,
                    runtime,
                    checkpoints,
                    variant_records,
                    model_id,
                )
                summary = summary_for(item, tasks, notes)
                if docs["curated"] and docs["overview"]:
                    lead = re.split(
                        r"(?:In this catalog|Both integration and runner|WorldFoundry serves it|WorldFoundry binds it|在本 catalog|本 catalog|WorldFoundry 通过)",
                        docs["overview"][0],
                        maxsplit=1,
                    )[0].strip()
                    summary = compact_text(lead, 240) or compact_text(docs["overview"][0], 240) or summary

                recipe = {
                    "id": model_id,
                    "name": name,
                    "category": category_id,
                    "categoryLabel": category["label"],
                    "categoryLabelZh": category["label_zh"],
                    "provider": provider,
                    "summary": summary,
                    "aliases": aliases,
                    "tasks": tasks,
                    "status": status,
                    "runtime": runtime,
                    "sources": links,
                    "checkpoints": checkpoints,
                    "variants": variant_records,
                    "inferenceTasks": inference_tasks,
                    "inputContract": input_contract(profile),
                    "artifacts": artifact_data(item, profile),
                    "notes": notes,
                    "docs": docs,
                    "commands": command_data(model_id, runtime_model_id, inference_tasks[0]),
                    "catalogPath": str(path.relative_to(ROOT)),
                }
                if not docs_catalog_visible(category_id, model_id, status["group"]):
                    continue
                recipes.append(portable_doc_value(recipe))
                category_counts[category_id] += 1

    recipes.sort(
        key=lambda recipe: (
            STATUS_ORDER.get(recipe["status"]["group"], 99),
            recipe["name"].lower(),
            recipe["id"],
        )
    )
    payload = {
        "total": len(recipes),
        "categories": [
            {
                "id": category_id,
                **meta,
                "count": category_counts[category_id],
            }
            for category_id, meta in CATEGORY_META.items()
        ],
        "recipes": recipes,
    }
    serialized_payload = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    index_payload = {
        "total": payload["total"],
        "categories": payload["categories"],
        "recipes": [
            {
                "id": recipe["id"],
                "name": recipe["name"],
                "category": recipe["category"],
                "categoryLabel": recipe["categoryLabel"],
                "categoryLabelZh": recipe["categoryLabelZh"],
                "provider": recipe["provider"],
                "summary": recipe["summary"],
                "aliases": recipe["aliases"],
                "tasks": recipe["tasks"],
                "status": recipe["status"],
                "runtime": {
                    key: recipe["runtime"].get(key)
                    for key in (
                        "profileId",
                        "environmentName",
                        "environmentKind",
                        "python",
                        "cudaLabel",
                    )
                },
                "checkpoint": recipe["checkpoints"][0] if recipe["checkpoints"] else None,
                "teaser": next(
                    (
                        item.get("poster") or item.get("src")
                        for item in recipe.get("docs", {}).get("figures") or []
                        if isinstance(item, dict) and (item.get("poster") or item.get("kind") != "video")
                    ),
                    None,
                ),
            }
            for recipe in recipes
        ],
    }
    serialized_index = json.dumps(index_payload, ensure_ascii=False, separators=(",", ":")) + "\n"

    outputs = ((OUT, serialized_payload), (INDEX_OUT, serialized_index))
    if args.check:
        stale_paths = [
            path
            for path, expected in outputs
            if not path.is_file() or path.read_text(encoding="utf-8") != expected
        ]
        page_status = sync_model_pages(recipes, check=True)
        if stale_paths:
            for path in stale_paths:
                print(f"stale generated model recipe data: {path}", file=sys.stderr)
        if stale_paths or page_status:
            print("run `npm --prefix docs/fumadocs run models:generate` to refresh it", file=sys.stderr)
            return 1
        print(f"model recipe data is current: {OUT} and {INDEX_OUT}")
        return 0

    for path, content in outputs:
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
    print(f"wrote {OUT} and {INDEX_OUT} recipes={len(recipes)}")
    return sync_model_pages(recipes, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
