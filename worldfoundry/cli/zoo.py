"""CLI commands for inspecting model-zoo and benchmark-zoo manifests."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import (
    hfd_root_path,
    resolve_worldfoundry_path,
)
from worldfoundry.evaluation.tasks.execution.orchestration.run_mode import BENCHMARK_RUN_PUBLIC_MODES
from worldfoundry.evaluation.utils import (
    BENCHMARK_ZOO_DIR,
    MODEL_ZOO_DIR,
    REPO_ROOT,
    TMP_ROOT,
    manifest_paths,
    worldfoundry_hfd_dataset_root,
)

from .utils import canonical_benchmark_zoo_id, canonical_model_zoo_id, json_dump, parse_key_value_mapping

DEFAULT_MODEL_ZOO_DIR = MODEL_ZOO_DIR
DEFAULT_BENCHMARK_ZOO_DIR = BENCHMARK_ZOO_DIR
_DEFAULT_HFD_ROOT = hfd_root_path()
_BENCHMARK_RUN_MODE_CHOICES = tuple(sorted(BENCHMARK_RUN_PUBLIC_MODES))

# ── Script path constants ───────────────────────────────────────

_SCRIPT_MODEL_ZOO_DOWNLOAD_CHECKPOINTS = "scripts/model_zoo/download_checkpoints.py"
_SCRIPT_SETUP_DOWNLOAD_EMBODIED_ACTION_ASSETS = "scripts/setup/download_embodied_action_official_assets.py"

# ── Default path helpers ────────────────────────────────────────


def _default_hfd_dataset_root() -> Path:
    """Return the default HFD dataset root path."""
    return worldfoundry_hfd_dataset_root()


# ── Formatting and display helpers ───────────────────────────────


def _iter_manifest_files(root: Path) -> list[Path]:
    """Enumerate manifest YAML/JSON files under a directory."""
    from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
        is_catalog_metadata_manifest,
        is_default_benchmark_catalog_root,
        iter_benchmark_catalog_manifest_paths,
        resolve_benchmark_catalog_root,
    )

    resolved = resolve_benchmark_catalog_root(root)
    if resolved.is_file():
        return [resolved]
    if is_default_benchmark_catalog_root(resolved):
        return list(iter_benchmark_catalog_manifest_paths(resolved))
    return [
        path
        for path in manifest_paths(resolved)
        if not is_catalog_metadata_manifest(path)
    ]


def _dash(value: object | None) -> str:
    """Render a value as a string, falling back to ``-`` when empty or ``None``."""
    text = "" if value is None else str(value)
    return text if text else "-"


def _compact_list(values: object, *, limit: int = 2) -> str:
    """Render a list as a compact string, truncating beyond *limit* items."""
    if values is None:
        items: list[str] = []
    elif isinstance(values, str):
        items = [values]
    else:
        try:
            items = [str(item) for item in values if str(item)]
        except TypeError:
            items = [str(values)]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" +{len(items) - limit}"


def _print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Print a left-aligned text table with header underline."""
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


# ── Manifest and needs helpers ───────────────────────────────────


def _model_manifest_paths(manifest_dir: Path) -> dict[str, str]:
    """Map each model id to its manifest file path under *manifest_dir*."""
    from worldfoundry.evaluation.models.catalog.schema import load_entries

    paths: dict[str, str] = {}
    for path in _iter_manifest_files(manifest_dir):
        for entry in load_entries(path):
            paths.setdefault(entry.model_id, str(path))
    return paths


def _benchmark_manifest_paths(manifest_dir: Path) -> dict[str, str]:
    """Map each benchmark id to its manifest file path under *manifest_dir*."""
    from worldfoundry.evaluation.tasks.catalog.schema import load_entries

    paths: dict[str, str] = {}
    for path in _iter_manifest_files(manifest_dir):
        for entry in load_entries(path):
            paths.setdefault(entry.benchmark_id, str(path))
    return paths


def _model_needs(entry) -> tuple[str, ...]:
    """Derive the requirement tags (api-key, checkpoint, gpu, runner) for a model entry."""
    needs: list[str] = []
    if entry.source_status == "api":
        needs.append("api-key")
    if entry.requires_auth:
        needs.append("auth")
    if entry.hf_repo_ids:
        needs.append("checkpoint")
    min_vram = entry.min_vram_gb
    variant_vram = [variant.min_vram_gb for variant in entry.variants if variant.min_vram_gb is not None]
    if min_vram is None and variant_vram:
        min_vram = min(variant_vram)
    if min_vram is not None:
        needs.append(f"gpu>={min_vram:g}GB")
    if entry.runner_entry_kind != "runnable_runner":
        needs.append("runner")
    return tuple(dict.fromkeys(needs))


def _declared_dataset_path_exists(raw_path: str) -> bool:
    """Check whether a declared dataset path exists on the local filesystem."""
    path = resolve_worldfoundry_path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.exists()


def _benchmark_declared_dataset_paths_ready(entry) -> bool:
    """Check whether all declared HuggingFace datasets are present locally."""
    dataset_ids = {dataset_id for dataset_id in entry.hf_dataset_ids if dataset_id}
    if not dataset_ids:
        return True
    ready_ids: set[str] = set()
    for ref in entry.dataset_refs or (entry.dataset,):
        if not ref.hf_dataset_id or not ref.path:
            continue
        if _declared_dataset_path_exists(ref.path):
            ready_ids.add(ref.hf_dataset_id)
    return dataset_ids <= ready_ids


def _benchmark_needs(entry) -> tuple[str, ...]:
    """Derive the requirement tags for a benchmark entry."""
    needs = list(entry.requires)
    if entry.requires_auth:
        needs.append("auth")
    runner_availability = entry.runner_availability if isinstance(entry.runner_availability, Mapping) else {}
    dataset_optional = runner_availability.get("dataset_optional_for_generated_artifacts") is True
    if entry.hf_dataset_ids and not dataset_optional and not _benchmark_declared_dataset_paths_ready(entry):
        needs.append("dataset")
    if entry.runner_target is None:
        needs.append("runner")
    return tuple(dict.fromkeys(needs))


def _split_filter_values(values: list[str] | None) -> tuple[str, ...]:
    """Split comma-separated filter values into a flat tuple of non-empty tokens."""
    tokens: list[str] = []
    for value in values or ():
        tokens.extend(item.strip() for item in str(value).split(","))
    return tuple(item for item in tokens if item)


def _need_matches(actual: str, expected: str) -> bool:
    """Check whether an actual need tag matches an expected filter, supporting ``gpu>=`` prefixes."""
    if actual == expected:
        return True
    if expected == "gpu" and actual.startswith("gpu"):
        return True
    return actual.startswith(f"{expected}>=")


def _has_requested_needs(entry, requested: tuple[str, ...]) -> bool:
    """Check whether a benchmark entry satisfies all requested need filters."""
    if not requested:
        return True
    actual_needs = _benchmark_needs(entry)
    return all(any(_need_matches(actual, expected) for actual in actual_needs) for expected in requested)


def _benchmark_need_tags(entry) -> tuple[str, ...]:
    """Classify benchmark need strings into standardized display tags."""
    tags: list[str] = []
    for need in _benchmark_needs(entry):
        text = str(need).lower()
        if need in {"auth", "dataset", "runner"}:
            tags.append(need)
        elif "gpu" in text or "cuda" in text:
            tags.append("gpu")
        elif (
            text == "api"
            or text.startswith("api ")
            or "api-key" in text
            or "api_key" in text
            or "api key" in text
            or "apikey" in text
        ):
            tags.append("api-key")
        elif "checkpoint" in text or "ckpt" in text or "weight" in text:
            tags.append("checkpoint")
        elif "result" in text:
            tags.append("official-results")
        elif "generated" in text or "artifact" in text or "video dir" in text:
            tags.append("generated-artifacts")
        elif "simulator" in text or "sapien" in text:
            tags.append("simulator")
        elif "root" in text or "repo" in text or "checkout" in text or "dataset" in text:
            tags.append("official-assets")
        else:
            tags.append("runtime")
    return tuple(dict.fromkeys(tags))


# ── Model discovery and readiness helpers ───────────────────────


def _model_next_action(entry) -> str:
    """Derive the suggested next action for a model-zoo entry."""
    if entry.integration_status == "blocked":
        return "review blocker notes before integration"
    if entry.runner_entry_kind == "runnable_runner":
        return f"run with worldfoundry-eval run --model {entry.model_id}"
    if entry.runner_entry_kind == "runner_candidate":
        return "add runtime profile and runner evidence"
    return "add runner manifest, runtime profile, and tests"


def _model_has_runner_target(entry) -> bool:
    """Check whether a model entry declares a runner target on itself or any variant."""
    return bool(entry.runner_target) or any(variant.runner_target for variant in entry.variants)


def _model_user_commands(entry) -> dict[str, str]:
    """Build the suggested CLI commands dictionary for a model entry."""
    model_id = entry.model_id
    commands = {
        "show": f"worldfoundry-eval zoo model-show --model-id {model_id} --json",
        "manifest": f"worldfoundry-eval zoo model-specs --model-id {model_id} --json",
    }
    if entry.hf_repo_ids:
        commands["checkpoint_check"] = f"worldfoundry-eval zoo model-download --model-id {model_id} --check-local --json"
    if entry.is_runnable_runner_entry:
        commands["run"] = (
            f"worldfoundry-eval run --model {model_id} --benchmark <benchmark-id> "
            f"--mode official-run --output-dir tmp/model_benchmark/{model_id}/<benchmark-id> --json"
        )
    elif _model_has_runner_target(entry):
        commands["plan"] = (
            f"worldfoundry-eval run --model {model_id} --benchmark <benchmark-id> "
            f"--mode official-run --plan-only --output-dir tmp/model_benchmark/{model_id}/<benchmark-id> --json"
        )
    return commands


def _model_command_readiness(entry) -> dict[str, bool]:
    """Compute the readiness flags for model CLI commands."""
    runner_ready = entry.is_runnable_runner_entry
    runner_target_declared = _model_has_runner_target(entry)
    checkpoint_command_ready = bool(entry.hf_repo_ids)
    return {
        "checkpoint_command_ready": checkpoint_command_ready,
        "validation_command_ready": True,
        "runner_target_declared": runner_target_declared,
        "runner_ready": runner_ready,
        "plan_ready": runner_ready or runner_target_declared,
        "one_command_ready": runner_ready,
    }


def _model_list_payload(entry) -> dict[str, Any]:
    """Build the full model list payload with discovery and readiness metadata."""
    payload = entry.to_dict()
    payload["verification_status"] = entry.verification_status
    payload["runner_entry_kind"] = entry.runner_entry_kind
    payload["runnable"] = entry.is_runnable_runner_entry
    payload.update(_model_command_readiness(entry))
    payload["commands"] = _model_user_commands(entry)
    payload["needs"] = list(_model_needs(entry))
    payload["next_action"] = _model_next_action(entry)
    return payload


# ── Benchmark discovery and readiness helpers ───────────────────


def _benchmark_next_action(entry) -> str:
    """Derive the suggested next action for a benchmark-zoo entry."""
    commands = _benchmark_user_commands(entry)
    if entry.integration_status == "blocked":
        if entry.blockers:
            return f"resolve blocker: {entry.blockers[0]}"
        return "resolve blocker before integration"
    if _benchmark_official_runner_ready(entry):
        return f"run with worldfoundry-eval run --benchmark {entry.benchmark_id}"
    if _benchmark_bounded_official_component_ready(entry):
        return f"run bounded official component with {commands['official_run']}"
    if _benchmark_official_runtime_declared(entry) or _benchmark_bounded_official_validation_ready(entry):
        if entry.verification_status == "normalizer_only" and commands.get("normalizer_run"):
            return f"run official validation with {commands['normalizer_run']}"
        validation_command = commands.get("official_validation") or commands.get("validate")
        if validation_command:
            return f"run official validation with {validation_command}"
        return "run official validation with worldfoundry-eval zoo benchmark-run --mode official-validation"
    if entry.runner_target:
        return "connect the runner to official-validation or official-run evidence"
    return "add catalog entry, task YAML, runnable evaluator, and validation evidence"


def _benchmark_maturity(entry) -> str:
    """Return the manifest maturity label for a benchmark entry."""
    maturity = str(entry.maturity or "").strip()
    return maturity or "planned"


def _benchmark_normalizer_command(entry) -> str | None:
    """Return a manifest validation command when the entry exposes a normalizer surface."""
    if not entry.validation_command:
        return None
    runner_availability = entry.runner_availability if isinstance(entry.runner_availability, Mapping) else {}
    runner_runtime = entry.runner_runtime if isinstance(entry.runner_runtime, Mapping) else {}
    surface = str(runner_availability.get("surface") or "")
    runtime_kind = str(runner_runtime.get("kind") or "")
    if (
        entry.verification_status == "normalizer_only"
        or surface == "official_result_normalizer"
        or runtime_kind in {"external_official_results_runner", "official_snapshot_normalizer"}
        or bool(runner_runtime.get("results_path_env"))
    ):
        return _shell_command_text(entry.validation_command)
    return None


def _benchmark_official_runner_ready(entry) -> bool:
    """Check whether a benchmark entry has a verified official runner surface."""
    return (
        entry.integration_status == "integrated"
        and entry.verification_status == "verified"
        and entry.runner_target is not None
        and bool(entry.official_benchmark_verified)
        and bool(entry.integration_evidence)
    )


def _benchmark_bounded_official_component_ready(entry) -> bool:
    """Check for a verified, runnable official component without claiming the full suite."""
    runner_availability = entry.runner_availability if isinstance(entry.runner_availability, Mapping) else {}
    return bool(
        entry.integration_status == "integrated"
        and entry.verification_status == "verified"
        and entry.runner_target is not None
        and entry.run_command is not None
        and entry.integration_evidence
        and runner_availability.get("bounded_official_component") is True
        and runner_availability.get("runtime_verified") is True
    )


def _benchmark_leaderboard_capable(entry) -> bool:
    """Check whether a benchmark entry is declared leaderboard-capable."""
    runner_availability = getattr(entry, "runner_availability", {})
    return bool(
        entry.integration_status == "integrated"
        and entry.runner_target is not None
        and isinstance(runner_availability, Mapping)
        and runner_availability.get("leaderboard_capable") is True
    )


def _benchmark_public_integration_status(entry) -> str:
    """Derive the public-facing integration status, upgrading credential-gated entries."""
    if entry.integration_status == "integrated" and entry.runner_availability.get("credential_gated"):
        return "integrated"
    if _benchmark_leaderboard_capable(entry):
        return "integrated"
    if (
        entry.integration_status == "integrated"
        and entry.verification_status in {"verified", "normalizer_only"}
        and entry.runner_target is not None
        and bool(entry.integration_evidence)
    ):
        return "integrated"
    if entry.integration_status == "integrated" and not _benchmark_official_runner_ready(entry):
        return "planned"
    return entry.integration_status


def _benchmark_official_runtime_declared(entry) -> bool:
    """Check whether a benchmark entry declares a run or validation command."""
    return entry.run_command is not None or entry.validation_command is not None


def _benchmark_bounded_official_validation_ready(entry) -> bool:
    """Check whether a benchmark has a verified official validation asset in its runner."""
    return _benchmark_verified_official_validation_asset(entry) is not None


def _benchmark_verified_official_validation_asset(entry) -> Mapping[str, Any] | None:
    """Search runner assets for a verified official validation scope."""
    assets = getattr(entry.runner, "assets", {})
    if not isinstance(assets, Mapping):
        return None
    for name, value in assets.items():
        if not isinstance(value, Mapping) or value.get("status") != "verified":
            continue
        scope = f"{name} {value.get('scope', '')}".lower()
        if "official" in scope and "validation" in scope:
            return value
    return None


def _benchmark_official_evidence_ready(entry) -> bool:
    """Check whether official benchmark verification and integration evidence are both present."""
    if entry.official_benchmark_verified and entry.integration_evidence:
        return True
    asset = _benchmark_verified_official_validation_asset(entry)
    return bool(
        asset
        and asset.get("official_benchmark_verified") is True
        and asset.get("integration_evidence") is True
    )


def _benchmark_user_commands(entry) -> dict[str, str]:
    """Build suggested CLI commands from manifest-declared command fields."""
    commands: dict[str, str] = {}
    if entry.validation_command:
        commands["official_validation"] = _shell_command_text(entry.validation_command)
        normalizer_command = _benchmark_normalizer_command(entry)
        if normalizer_command is not None:
            commands["normalizer_run"] = normalizer_command
    if entry.run_command:
        commands["official_run"] = _shell_command_text(entry.run_command)
    if entry.benchmark_id == "videoscore" and "normalizer_run" not in commands:
        commands["normalizer_run"] = (
            "worldfoundry-eval zoo benchmark-run --benchmark-id videoscore "
            "--mode official-validation --official-results-path '<eval_*_videoscore.json>' "
            "--output-dir '<out>' --json"
        )
    if _benchmark_official_runner_ready(entry):
        commands["ready_now"] = str(entry.ready_now_command)
        commands["eval"] = str(entry.ready_now_command)
    return commands


def _shell_command_text(command: str | tuple[str, ...]) -> str:
    """Join a shell command tuple or return a plain string command."""
    if isinstance(command, str):
        return command
    return shlex.join(str(part) for part in command)


def _benchmark_command_readiness(entry) -> dict[str, bool]:
    """Compute the readiness flags for benchmark CLI commands."""
    official_ready = _benchmark_official_runner_ready(entry)
    bounded_official_component_ready = _benchmark_bounded_official_component_ready(entry)
    leaderboard_capable = _benchmark_leaderboard_capable(entry)
    commands = _benchmark_user_commands(entry)
    normalizer_command_ready = "normalizer_run" in commands
    official_runtime_declared = _benchmark_official_runtime_declared(entry)
    bounded_official_validation_ready = _benchmark_bounded_official_validation_ready(entry)
    official_evidence_ready = _benchmark_official_evidence_ready(entry)
    return {
        "eval_ready": official_ready,
        "official_ready": official_ready,
        "official_runner_ready": official_ready,
        "official_runtime_declared": official_runtime_declared,
        "official_runtime_command_ready": official_runtime_declared,
        "bounded_official_component_ready": bounded_official_component_ready,
        "bounded_official_validation_ready": bounded_official_validation_ready,
        "official_evidence_ready": official_evidence_ready,
        "leaderboard_capable": leaderboard_capable,
        "leaderboard_integration_ready": leaderboard_capable,
        "leaderboard_ready": official_ready and bool(entry.leaderboard_valid),
        "full_leaderboard_ready": official_ready and bool(entry.leaderboard_valid),
        "ready_now_command_ready": official_ready and bool(entry.ready_now_command),
        "normalizer_command_ready": normalizer_command_ready,
        "normalizer_ready": normalizer_command_ready,
        "validation_or_normalizer_ready": (
            official_ready or bounded_official_component_ready or normalizer_command_ready
        ),
        "one_command_ready": official_ready or bounded_official_component_ready,
    }


def _benchmark_ready_surface(entry) -> str:
    """Return the one-command surface label for a benchmark entry."""
    readiness = _benchmark_command_readiness(entry)
    if readiness["eval_ready"]:
        return "eval"
    if readiness["bounded_official_component_ready"]:
        return "bounded"
    return "-"


# ── Script-bridge helpers ───────────────────────────────────────


def _load_repo_script(relative_path: str) -> ModuleType:
    """Delegate repo script loading to the shared ``cli._load_repo_script``."""
    import worldfoundry.cli as cli_module

    return cli_module._load_repo_script(relative_path)


def _add_path_arg(argv: list[str], flag: str, value: Path | None) -> None:
    """Append a flag+value pair to *argv* when *value* is not ``None``."""
    if value is not None:
        argv.extend([flag, str(value)])


def _add_value_arg(argv: list[str], flag: str, value) -> None:
    """Append a flag+value pair to *argv* when *value* is not ``None``."""
    if value is not None:
        argv.extend([flag, str(value)])


def _add_repeated_args(argv: list[str], flag: str, values: list[str] | None) -> None:
    """Append repeated flag+value pairs for each item in *values*."""
    for value in values or ():
        argv.extend([flag, str(value)])


# ── Zoo models command handlers ─────────────────────────────────


def _handle_zoo_models_list(args: argparse.Namespace) -> int:
    from worldfoundry.evaluation.models.catalog import load_model_zoo_registry

    registry = load_model_zoo_registry(args.manifest_dir)
    entries = registry.list()
    if args.integration_status:
        entries = [entry for entry in entries if entry.integration_status == args.integration_status]
    if args.source_status:
        entries = [entry for entry in entries if entry.source.status == args.source_status]

    payload = [_model_list_payload(entry) for entry in entries]
    if args.json:
        json_dump(payload)
        return 0

    manifest_paths = _model_manifest_paths(args.manifest_dir)
    rows = []
    for entry in entries:
        rows.append(
            (
                entry.model_id,
                entry.integration_status,
                entry.source.status,
                "yes" if entry.is_runnable_runner_entry else "no",
                _compact_list(_model_needs(entry), limit=3),
                entry.runner_entry_kind,
                _compact_list(registry.aliases_for(entry.model_id), limit=2),
                _dash(manifest_paths.get(entry.model_id)),
            )
        )
    _print_table(
        ("id", "status", "source", "runnable", "needs", "runner", "aliases", "manifest"),
        rows,
    )
    return 0


def _handle_zoo_model_specs(args: argparse.Namespace) -> int:
    """Export model-zoo entries as public WorldFoundry ``WorldModelManifest`` objects."""
    from worldfoundry.evaluation.models.catalog.manifest import model_zoo_entries_to_world_model_manifests
    from worldfoundry.evaluation.models.catalog.schema import load_entries

    entries = []
    for path in _iter_manifest_files(args.manifest_dir):
        entries.extend(load_entries(path))
    if args.model_id:
        selected = {
            canonical_model_zoo_id(model_id, args.manifest_dir)
            for model_id in args.model_id
        }
        entries = [entry for entry in entries if entry.model_id in selected]
    if args.integration_status:
        entries = [entry for entry in entries if entry.integration_status == args.integration_status]
    if args.source_status:
        entries = [entry for entry in entries if entry.source.status == args.source_status]

    manifests = model_zoo_entries_to_world_model_manifests(entries)
    payload = [manifest.to_dict() for manifest in manifests]
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        json_dump(payload)
        return 0

    for manifest in manifests:
        checkpoint_count = len(manifest.metadata.get("hf_repo_ids", ()))
        print(
            f"{manifest.model_id}: capabilities={len(manifest.capabilities)} "
            f"checkpoints={checkpoint_count}"
        )
    if args.output_json:
        print(f"wrote: {args.output_json}")
    return 0


def _handle_zoo_model_show(args: argparse.Namespace) -> int:
    """Show one model-zoo entry with discovery, readiness, and command metadata."""
    from worldfoundry.evaluation.models.catalog import load_model_zoo_registry

    registry = load_model_zoo_registry(args.manifest_dir)
    entry = registry.get(args.model_id)
    payload = entry.to_dict()
    payload["registry_aliases"] = list(registry.aliases_for(args.model_id))
    payload["discovery"] = {
        "runnable": entry.is_runnable_runner_entry,
        "needs": list(_model_needs(entry)),
        "runner": entry.runner_entry_kind,
        "manifest_path": _model_manifest_paths(args.manifest_dir).get(entry.model_id),
        "next_action": _model_next_action(entry),
        "commands": _model_user_commands(entry),
        **_model_command_readiness(entry),
    }
    if args.include_manifest:
        manifest = registry.to_world_model_manifests()
        payload["world_model_manifest"] = next(
            item.to_dict() for item in manifest if item.model_id == entry.model_id
        )
    if args.json:
        json_dump(payload)
        return 0

    print(f"model_id: {entry.model_id}")
    print(f"name: {entry.name or entry.model_id}")
    print(f"source_status: {entry.source_status}")
    print(f"integration_status: {entry.integration_status}")
    print(f"runnable: {payload['discovery']['runnable']}")
    print(f"needs: {', '.join(payload['discovery']['needs']) if payload['discovery']['needs'] else '-'}")
    print(f"runner: {payload['discovery']['runner']}")
    print(f"manifest_path: {payload['discovery']['manifest_path'] or '-'}")
    print(f"next_action: {payload['discovery']['next_action']}")
    print(f"tasks: {', '.join(entry.tasks) if entry.tasks else '-'}")
    print(f"aliases: {', '.join(payload['registry_aliases']) if payload['registry_aliases'] else '-'}")
    print(f"hf_repo_ids: {', '.join(entry.hf_repo_ids) if entry.hf_repo_ids else '-'}")
    integrated_variants = [variant.variant_id for variant in entry.variants if variant.integration_status == "integrated"]
    if entry.variants:
        print(f"variants: {len(entry.variants)}")
        print(f"integrated_variants: {', '.join(integrated_variants) if integrated_variants else '-'}")
    return 0


def _handle_zoo_model_download(args: argparse.Namespace) -> int:
    """Download or check Hugging Face checkpoints declared by a model-zoo manifest."""
    module = _load_repo_script(_SCRIPT_MODEL_ZOO_DOWNLOAD_CHECKPOINTS)
    model_id = canonical_model_zoo_id(args.model_id, args.manifest_dir)
    argv: list[str] = []
    _add_path_arg(argv, "--manifest-dir", args.manifest_dir)
    _add_value_arg(argv, "--model-id", model_id)
    _add_repeated_args(argv, "--repo-id", args.repo_id)
    _add_path_arg(argv, "--cache-dir", args.cache_dir)
    if args.execute:
        argv.append("--execute")
    if args.disable_xet:
        argv.append("--disable-xet")
    if args.disable_hf_transfer:
        argv.append("--disable-hf-transfer")
    _add_value_arg(argv, "--timeout", args.timeout)
    _add_value_arg(argv, "--retries", args.retries)
    _add_value_arg(argv, "--max-workers", args.max_workers)
    if args.check_local:
        argv.append("--check-local")
    if args.allow_all_execute:
        argv.append("--allow-all-execute")
    _add_path_arg(argv, "--report-path", args.report_path)
    if args.json:
        argv.append("--json")
    return module.main(argv)


def _handle_zoo_embodied_assets(args: argparse.Namespace) -> int:
    """Download or check official embodied action model assets."""
    module = _load_repo_script(_SCRIPT_SETUP_DOWNLOAD_EMBODIED_ACTION_ASSETS)
    argv: list[str] = list(args.models or [])
    _add_path_arg(argv, "--hf-root", args.hf_root)
    _add_path_arg(argv, "--asset-root", args.asset_root)
    _add_path_arg(argv, "--openpi-root", args.openpi_root)
    _add_path_arg(argv, "--repos-root", args.repos_root)
    _add_path_arg(argv, "--report-jsonl", args.report_jsonl)
    _add_path_arg(argv, "--summary-json", args.summary_json)
    _add_path_arg(argv, "--log-dir", args.log_dir)
    _add_value_arg(argv, "--max-workers", args.max_workers)
    _add_value_arg(argv, "--url-timeout-seconds", args.url_timeout_seconds)
    if not args.skip_existing:
        argv.append("--no-skip-existing")
    if args.plan_only:
        argv.append("--plan-only")
    if args.list:
        argv.append("--list")
    return module.main(argv)



# ── Zoo benchmarks command handlers ─────────────────────────────


def _handle_zoo_benchmarks_list(args: argparse.Namespace) -> int:
    from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry

    registry = load_benchmark_zoo_registry(args.manifest_dir)
    entries = registry.list()
    if args.integration_status:
        entries = [entry for entry in entries if _benchmark_public_integration_status(entry) == args.integration_status]
    if args.source_status:
        entries = [entry for entry in entries if entry.source.status == args.source_status]
    requested_needs = _split_filter_values(getattr(args, "needs", None))
    if requested_needs:
        entries = [entry for entry in entries if _has_requested_needs(entry, requested_needs)]
    if getattr(args, "ready_now", False):
        entries = [entry for entry in entries if _benchmark_command_readiness(entry)["one_command_ready"]]
    if getattr(args, "official_ready", False):
        entries = [entry for entry in entries if _benchmark_command_readiness(entry)["official_runner_ready"]]

    payload = [_benchmark_list_payload(entry) for entry in entries]
    if args.json:
        json_dump(payload)
        return 0

    manifest_paths = _benchmark_manifest_paths(args.manifest_dir)
    rows = []
    for entry in entries:
        row = (
            entry.benchmark_id,
            _benchmark_public_integration_status(entry),
            _benchmark_maturity(entry),
            _benchmark_ready_surface(entry),
            "yes" if entry.leaderboard_valid else "no",
            "yes" if entry.runner_target else "no",
            _compact_list(_benchmark_need_tags(entry), limit=4),
            _compact_list(registry.aliases_for(entry.benchmark_id), limit=2),
        )
        if getattr(args, "show_manifest", False):
            row = (*row, _dash(manifest_paths.get(entry.benchmark_id)))
        rows.append(row)
    headers = ("id", "status", "surface", "ready", "leaderboard", "runner", "needs", "aliases")
    if getattr(args, "show_manifest", False):
        headers = (*headers, "manifest")
    _print_table(
        headers,
        rows,
    )
    return 0


def _benchmark_list_payload(entry) -> dict[str, Any]:
    """Build the full benchmark list payload with discovery and readiness metadata."""
    payload = entry.to_dict()
    payload.pop("contract_validation_command", None)
    payload["integration_status"] = _benchmark_public_integration_status(entry)
    payload["declared_maturity"] = payload.get("maturity")
    payload["maturity"] = _benchmark_maturity(entry)
    payload["verification_status"] = entry.verification_status
    payload.update(_benchmark_command_readiness(entry))
    payload["commands"] = _benchmark_user_commands(entry)
    payload["needs"] = list(_benchmark_needs(entry))
    payload["next_action"] = _benchmark_next_action(entry)
    return payload


def _handle_zoo_benchmark_specs(args: argparse.Namespace) -> int:
    """Export benchmark-zoo entries as public WorldFoundry ``BenchmarkSpec`` objects."""
    from worldfoundry.evaluation.tasks.catalog.schema import load_entries
    from worldfoundry.evaluation.tasks.catalog.specs import (
        benchmark_zoo_entries_to_benchmark_specs,
    )

    entries = []
    for path in _iter_manifest_files(args.manifest_dir):
        entries.extend(load_entries(path))
    if args.benchmark_id:
        selected = {
            canonical_benchmark_zoo_id(benchmark_id, args.manifest_dir)
            for benchmark_id in args.benchmark_id
        }
        entries = [entry for entry in entries if entry.benchmark_id in selected]
    if args.integration_status:
        entries = [entry for entry in entries if _benchmark_public_integration_status(entry) == args.integration_status]
    if args.source_status:
        entries = [entry for entry in entries if entry.source.status == args.source_status]

    specs = benchmark_zoo_entries_to_benchmark_specs(entries)
    payload = [spec.to_dict() for spec in specs]
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        json_dump(payload)
        return 0

    for spec in specs:
        task_count = len(spec.tasks)
        metric_count = len(spec.metrics)
        print(f"{spec.benchmark_id}: tasks={task_count} metrics={metric_count}")
    if args.output_json:
        print(f"wrote: {args.output_json}")
    return 0


def _handle_zoo_benchmark_show(args: argparse.Namespace) -> int:
    """Show one benchmark-zoo entry with discovery, readiness, and command metadata."""
    from worldfoundry.evaluation.tasks.catalog.specs import benchmark_zoo_entries_to_benchmark_specs
    from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry

    registry = load_benchmark_zoo_registry(args.manifest_dir)
    entry = registry.get(args.benchmark_id)
    payload = entry.to_dict()
    payload.pop("contract_validation_command", None)
    payload["integration_status"] = _benchmark_public_integration_status(entry)
    payload["registry_aliases"] = list(registry.aliases_for(args.benchmark_id))
    discovery = {
        "runnable": entry.runner_target is not None and entry.integration_status != "blocked",
        "needs": list(_benchmark_needs(entry)),
        "surface": _benchmark_maturity(entry),
        "declared_surface": entry.maturity,
        "manifest_path": _benchmark_manifest_paths(args.manifest_dir).get(entry.benchmark_id),
        "next_action": _benchmark_next_action(entry),
        "ready_now_command": entry.ready_now_command,
        "one_click_command": entry.one_click_command,
        "commands": _benchmark_user_commands(entry),
        **_benchmark_command_readiness(entry),
    }
    payload["discovery"] = discovery
    payload["declared_maturity"] = payload.get("maturity")
    payload["maturity"] = discovery["surface"]
    payload["verification_status"] = entry.verification_status
    payload["commands"] = discovery["commands"]
    payload["needs"] = discovery["needs"]
    payload["next_action"] = discovery["next_action"]
    payload.update(_benchmark_command_readiness(entry))
    if args.include_spec:
        spec = benchmark_zoo_entries_to_benchmark_specs([entry])[0]
        payload["benchmark_spec"] = spec.to_dict()
    if args.json:
        json_dump(payload)
        return 0

    print(f"benchmark_id: {entry.benchmark_id}")
    print(f"name: {entry.name or entry.benchmark_id}")
    print(f"source_status: {entry.source_status}")
    print(f"open_source_status: {entry.open_source_status}")
    print(f"integration_status: {entry.integration_status}")
    print(f"surface: {payload['discovery']['surface']}")
    print(f"runnable: {payload['discovery']['runnable']}")
    print(f"needs: {', '.join(payload['discovery']['needs']) if payload['discovery']['needs'] else '-'}")
    print(f"manifest_path: {payload['discovery']['manifest_path'] or '-'}")
    print(f"next_action: {payload['discovery']['next_action']}")
    print(
        "bounded_official_component_ready: "
        f"{payload['discovery']['bounded_official_component_ready']}"
    )
    print(f"official_runner_ready: {payload['discovery']['official_runner_ready']}")
    print(f"one_command_ready: {payload['discovery']['one_command_ready']}")
    print(f"official_benchmark_verified: {entry.official_benchmark_verified}")
    print(f"integration_evidence: {entry.integration_evidence}")
    print(f"leaderboard_valid: {entry.leaderboard_valid}")
    print(f"aliases: {', '.join(payload['registry_aliases']) if payload['registry_aliases'] else '-'}")
    print(f"domains: {', '.join(entry.domains) if entry.domains else '-'}")
    print(f"modalities: {', '.join(entry.modalities) if entry.modalities else '-'}")
    print(f"hf_dataset_ids: {', '.join(entry.hf_dataset_ids) if entry.hf_dataset_ids else '-'}")
    print(f"runner: {entry.runner.verification_status}")
    print(f"requires: {', '.join(entry.requires) if entry.requires else '-'}")
    print(f"blockers: {'; '.join(entry.blockers) if entry.blockers else '-'}")
    print(f"metrics: {len(entry.metrics)}")
    return 0


def _benchmark_run_cli_success(result: Any) -> bool:
    """Determine CLI success from a benchmark run result, accepting contract-only and normalizer-only outcomes."""
    if result.ok:
        return True

    scorecard_path = Path(result.scorecard_path)
    if not scorecard_path.is_file():
        return False

    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    if metadata.get("contract_only") is True:
        return True
    if metadata.get("normalizer_only") is True:
        try:
            scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(scorecard, Mapping):
            return False
        evaluation = scorecard.get("evaluation")
        evaluation = evaluation if isinstance(evaluation, Mapping) else {}
        return scorecard.get("normalization_ok") is True or evaluation.get("available") is True
    return False


def _handle_zoo_benchmark_run(args: argparse.Namespace) -> int:
    """Run a benchmark-zoo manifest benchmark evaluation."""
    from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import run_benchmark_execution

    benchmark_id = canonical_benchmark_zoo_id(args.benchmark_id, args.manifest_dir)
    env_overrides = parse_key_value_mapping(args.env)
    result = run_benchmark_execution(
        benchmark_id,
        output_dir=args.output_dir,
        manifest_path=args.manifest_dir,
        mode=args.mode,
        generated_artifact_dir=args.generated_artifact_dir,
        official_results_path=args.official_results_path,
        score_dir=args.score_dir,
        benchmark_data_root=args.benchmark_data_root,
        prompt_manifest=args.prompt_manifest,
        metrics=args.metrics,
        limit=args.limit,
        result_model_id=args.result_model_id,
        timeout_seconds=args.timeout,
        workdir=args.workdir,
        env_overrides=env_overrides,
    )
    payload = result.to_dict()
    if args.json:
        json_dump(payload)
    else:
        print(
            f"{result.benchmark_id}: ok={result.ok} "
            f"official_benchmark_verified={result.official_benchmark_verified} "
            f"integration_evidence={result.integration_evidence} "
            f"scorecard={result.scorecard_path}"
        )
    return 0 if _benchmark_run_cli_success(result) else 1


# ── Parser registration ─────────────────────────────────────────


def _dispatch_zoo_handler(name: str):
    """Build a lazy bridge to the zoo handler kept on the public CLI module.

    Args:
        name: Handler function name in this module.
    """
    def _dispatch(args: argparse.Namespace) -> int:
        return globals()[name](args)

    return _dispatch


def register_zoo_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register model-zoo and benchmark-zoo CLI commands.

    Args:
        subparsers: Root argparse subparser collection.
    """
    zoo_parser = subparsers.add_parser("zoo", help="Inspect model-zoo and benchmark-zoo manifests")
    zoo_subparsers = zoo_parser.add_subparsers(dest="zoo_command", required=True)

    zoo_models_parser = zoo_subparsers.add_parser("models", help="List model-zoo entries")
    zoo_models_parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MODEL_ZOO_DIR)
    zoo_models_parser.add_argument("--integration-status", choices=["integrated", "planned", "blocked"])
    zoo_models_parser.add_argument("--source-status", choices=["open_source", "api", "closed", "unknown"])
    zoo_models_parser.add_argument("--json", action="store_true")
    zoo_models_parser.set_defaults(func=_dispatch_zoo_handler("_handle_zoo_models_list"))

    zoo_model_specs_parser = zoo_subparsers.add_parser(
        "model-specs",
        help="Export model-zoo entries as public WorldFoundry WorldModelManifest objects",
    )
    zoo_model_specs_parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MODEL_ZOO_DIR)
    zoo_model_specs_parser.add_argument("--model-id", action="append")
    zoo_model_specs_parser.add_argument("--integration-status", choices=["integrated", "planned", "blocked"])
    zoo_model_specs_parser.add_argument("--source-status", choices=["open_source", "api", "closed", "unknown"])
    zoo_model_specs_parser.add_argument("--output-json", type=Path)
    zoo_model_specs_parser.add_argument("--json", action="store_true")
    zoo_model_specs_parser.set_defaults(func=_dispatch_zoo_handler("_handle_zoo_model_specs"))

    zoo_model_show_parser = zoo_subparsers.add_parser(
        "model-show",
        help="Show one model-zoo entry using canonical id or registry alias",
    )
    zoo_model_show_parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MODEL_ZOO_DIR)
    zoo_model_show_parser.add_argument("--model-id", required=True)
    zoo_model_show_parser.add_argument("--include-manifest", action="store_true")
    zoo_model_show_parser.add_argument("--json", action="store_true")
    zoo_model_show_parser.set_defaults(func=_dispatch_zoo_handler("_handle_zoo_model_show"))

    zoo_model_download_parser = zoo_subparsers.add_parser(
        "model-download",
        help="Download, resume, or check Hugging Face checkpoints declared by model-zoo manifests",
    )
    zoo_model_download_parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MODEL_ZOO_DIR)
    zoo_model_download_parser.add_argument("--model-id")
    zoo_model_download_parser.add_argument("--repo-id", action="append", default=None)
    zoo_model_download_parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_HFD_ROOT)
    zoo_model_download_parser.add_argument("--execute", action="store_true")
    zoo_model_download_parser.add_argument("--disable-xet", action="store_true")
    zoo_model_download_parser.add_argument("--disable-hf-transfer", action="store_true")
    zoo_model_download_parser.add_argument("--timeout", type=int, default=None)
    zoo_model_download_parser.add_argument("--retries", type=int, default=0)
    zoo_model_download_parser.add_argument("--max-workers", type=int, default=1)
    zoo_model_download_parser.add_argument("--check-local", action="store_true")
    zoo_model_download_parser.add_argument("--allow-all-execute", action="store_true")
    zoo_model_download_parser.add_argument("--report-path", type=Path)
    zoo_model_download_parser.add_argument("--json", action="store_true")
    zoo_model_download_parser.set_defaults(func=_dispatch_zoo_handler("_handle_zoo_model_download"))

    zoo_embodied_assets_parser = zoo_subparsers.add_parser(
        "embodied-assets",
        help="Download or check official embodied action model assets with structured retry evidence",
        description="Download or check official embodied action model assets with structured retry evidence.",
    )
    zoo_embodied_assets_parser.add_argument("models", nargs="*", help="Model ids to download, or all.")
    zoo_embodied_assets_parser.add_argument("--hf-root", type=Path)
    zoo_embodied_assets_parser.add_argument("--asset-root", type=Path)
    zoo_embodied_assets_parser.add_argument("--openpi-root", type=Path)
    zoo_embodied_assets_parser.add_argument("--repos-root", type=Path)
    zoo_embodied_assets_parser.add_argument("--report-jsonl", type=Path)
    zoo_embodied_assets_parser.add_argument("--summary-json", type=Path)
    zoo_embodied_assets_parser.add_argument("--log-dir", type=Path)
    zoo_embodied_assets_parser.add_argument("--max-workers", type=int)
    zoo_embodied_assets_parser.add_argument("--url-timeout-seconds", type=int)
    zoo_embodied_assets_parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    zoo_embodied_assets_parser.add_argument("--plan-only", action="store_true")
    zoo_embodied_assets_parser.add_argument("--list", action="store_true")
    zoo_embodied_assets_parser.set_defaults(func=_dispatch_zoo_handler("_handle_zoo_embodied_assets"))

    zoo_benchmarks_parser = zoo_subparsers.add_parser("benchmarks", help="List benchmark-zoo entries")
    zoo_benchmarks_parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_BENCHMARK_ZOO_DIR)
    zoo_benchmarks_parser.add_argument("--integration-status", choices=["integrated", "planned", "blocked"])
    zoo_benchmarks_parser.add_argument("--source-status", choices=["open_source", "api", "closed", "unknown"])
    zoo_benchmarks_parser.add_argument(
        "--ready-now",
        action="store_true",
        help="Only show benchmarks with a verified full or bounded official one-command surface.",
    )
    zoo_benchmarks_parser.add_argument(
        "--official-ready",
        action="store_true",
        help="Only show benchmarks with a verified official runner surface.",
    )
    zoo_benchmarks_parser.add_argument(
        "--needs",
        action="append",
        help="Filter to benchmarks requiring all listed needs, comma-separated values allowed.",
    )
    zoo_benchmarks_parser.add_argument(
        "--show-manifest",
        action="store_true",
        help="Include manifest file paths in the human table.",
    )
    zoo_benchmarks_parser.add_argument("--json", action="store_true")
    zoo_benchmarks_parser.set_defaults(func=_dispatch_zoo_handler("_handle_zoo_benchmarks_list"))

    zoo_benchmark_specs_parser = zoo_subparsers.add_parser(
        "benchmark-specs",
        help="Export benchmark-zoo entries as public WorldFoundry BenchmarkSpec objects",
    )
    zoo_benchmark_specs_parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_BENCHMARK_ZOO_DIR)
    zoo_benchmark_specs_parser.add_argument("--benchmark-id", action="append")
    zoo_benchmark_specs_parser.add_argument("--integration-status", choices=["integrated", "planned", "blocked"])
    zoo_benchmark_specs_parser.add_argument("--source-status", choices=["open_source", "api", "closed", "unknown"])
    zoo_benchmark_specs_parser.add_argument("--output-json", type=Path)
    zoo_benchmark_specs_parser.add_argument("--json", action="store_true")
    zoo_benchmark_specs_parser.set_defaults(func=_dispatch_zoo_handler("_handle_zoo_benchmark_specs"))

    zoo_benchmark_show_parser = zoo_subparsers.add_parser(
        "benchmark-show",
        help="Show one benchmark-zoo entry using canonical id or registry alias",
    )
    zoo_benchmark_show_parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_BENCHMARK_ZOO_DIR)
    zoo_benchmark_show_parser.add_argument("--benchmark-id", required=True)
    zoo_benchmark_show_parser.add_argument("--include-spec", action="store_true")
    zoo_benchmark_show_parser.add_argument("--json", action="store_true")
    zoo_benchmark_show_parser.set_defaults(func=_dispatch_zoo_handler("_handle_zoo_benchmark_show"))

    zoo_benchmark_run_parser = zoo_subparsers.add_parser(
        "benchmark-run",
        help="Run a benchmark-zoo manifest benchmark evaluation",
    )
    zoo_benchmark_run_parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_BENCHMARK_ZOO_DIR)
    zoo_benchmark_run_parser.add_argument("--benchmark-id", required=True)
    zoo_benchmark_run_parser.add_argument(
        "--mode",
        choices=_BENCHMARK_RUN_MODE_CHOICES,
        default="official-run",
        help="Benchmark runner mode. Use official-run for integrated evaluators, official-validation for result import, or normalizer for scorecard normalization.",
    )
    zoo_benchmark_run_parser.add_argument("--output-dir", type=Path, required=True)
    zoo_benchmark_run_parser.add_argument("--generated-artifact-dir", type=Path)
    zoo_benchmark_run_parser.add_argument(
        "--metrics",
        action="append",
        help="Metric ids or dimensions to run; comma-separated values are accepted and the flag may repeat.",
    )
    zoo_benchmark_run_parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N benchmark samples for a bounded smoke run.",
    )
    zoo_benchmark_run_parser.add_argument(
        "--benchmark-data-root",
        type=Path,
        help="Optional local benchmark data root for integrated runners such as PhyGround and WorldModelBench.",
    )
    zoo_benchmark_run_parser.add_argument(
        "--prompt-manifest",
        type=Path,
        help="Optional prompt/question manifest used for generated-artifact coverage checks.",
    )
    zoo_benchmark_run_parser.add_argument(
        "--result-model-id",
        help="Optional model id used to filter multi-model official result dumps before normalization.",
    )
    zoo_benchmark_run_parser.add_argument(
        "--official-results-path",
        type=Path,
        help="Normalize a caller-provided official result JSON/JSONL/CSV/TSV file without executing upstream code.",
    )
    zoo_benchmark_run_parser.add_argument(
        "--score-dir",
        type=Path,
        help="Optional benchmark score directory for integrated metric aggregators such as CameraBench.",
    )
    zoo_benchmark_run_parser.add_argument("--timeout", type=float)
    zoo_benchmark_run_parser.add_argument("--workdir", type=Path)
    zoo_benchmark_run_parser.add_argument(
        "--env",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Environment override for official runtime modes. Repeatable.",
    )
    zoo_benchmark_run_parser.add_argument("--json", action="store_true")
    zoo_benchmark_run_parser.set_defaults(func=_dispatch_zoo_handler("_handle_zoo_benchmark_run"))
