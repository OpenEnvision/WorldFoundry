"""Strict validators for runtime profiles and data-backed pipeline references."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..pipelines.aliases import load_pipeline_alias_registry
from ..pipelines.bindings import (
    PipelineBindingRegistry,
    load_pipeline_binding_registry,
    resolve_pipeline_binding,
    runtime_profile_id,
)
from .assets import RuntimeAssetProfile, load_runtime_asset_profile_by_id
from .environments import RuntimeEnvironmentProfile, load_runtime_environment_profile_by_id
from .profiles import RuntimeProfile

# Recognized artifact kinds that a runtime profile may declare.
KNOWN_ARTIFACT_KINDS = frozenset(
    {
        "action_tokens",
        "action_trace",
        "blocked_plan",
        "benchmark_scorecard",
        "generated_3d_asset",
        "generated_4d_scene",
        "generated_artifact",
        "generated_audio",
        "generated_image",
        "generated_video",
        "generated_world",
        "evaluation_report",
        "memory_index",
        "metadata_profile",
        "model_response",
        "request_plan",
        "session_trace",
        "world_state",
    }
)


# ── Validation data model ────────────────────────────────────


@dataclass(frozen=True)
class RuntimeValidationIssue:
    """Validation issue or error returned by runtime validation checks.

    Attributes:
        code: Machine-readable issue identifier (e.g. ``"runtime_environment_missing"``).
        message: Human-readable description of the issue.
        field: Dot-separated field path that triggered the issue.
        severity: Issue severity level — ``"error"`` by default.
    """

    code: str
    message: str
    field: str = ""
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        """Convert the validation issue to a dictionary representation."""
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "severity": self.severity,
        }


# ── Public validation functions ──────────────────────────────


def validate_runtime_profile_references(
    profile: RuntimeProfile,
    *,
    environment_root: str | None = None,
    asset_root: str | None = None,
    binding_root: str | None = None,
    environments: Mapping[str, RuntimeEnvironmentProfile] | None = None,
    assets: Mapping[str, RuntimeAssetProfile] | None = None,
) -> tuple[RuntimeValidationIssue, ...]:
    """Validate strict target-profile references without importing model weights.

    Checks artifact defaults, environment, asset, and pipeline-binding
    references declared in ``profile.execution``, verifying that referenced
    profiles exist and their ``model_id`` matches.

    Args:
        profile: The :class:`RuntimeProfile` to validate.
        environment_root: Optional root override for environment manifest lookup.
        asset_root: Optional root override for asset manifest lookup.
        binding_root: Optional root override for pipeline binding lookup.
        environments: Optional pre-loaded environment profiles keyed by id.
            When given, lookups use this mapping instead of re-scanning the
            manifest tree per call (see :func:`validate_runtime_registry`).
        assets: Optional pre-loaded asset profiles keyed by id (same contract
            as ``environments``).

    Returns:
        A tuple of :class:`RuntimeValidationIssue` instances (may be empty).
    """

    issues = list(_validate_artifact_defaults(profile))
    environment_id = _text_or_none(profile.execution.get("environment"))
    asset_id = _text_or_none(profile.execution.get("assets"))
    binding_id = _text_or_none(profile.execution.get("pipeline_binding"))

    if environment_id:
        try:
            environment = _lookup_environment(environment_id, environments, environment_root)
            if environment.model_id != profile.model_id and environment.environment_id != environment_id:
                issues.append(
                    RuntimeValidationIssue(
                        code="runtime_environment_model_mismatch",
                        field="execution.environment",
                        message=(
                            f"runtime profile {profile.model_id!r} references environment {environment_id!r} "
                            f"for model {environment.model_id!r}"
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            issues.append(
                RuntimeValidationIssue(
                    code="runtime_environment_missing",
                    field="execution.environment",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )

    if asset_id:
        try:
            asset_profile = _lookup_asset(asset_id, assets, asset_root, profile)
            if asset_profile.model_id != profile.model_id and asset_profile.asset_profile_id != asset_id:
                issues.append(
                    RuntimeValidationIssue(
                        code="runtime_assets_model_mismatch",
                        field="execution.assets",
                        message=(
                            f"runtime profile {profile.model_id!r} references assets {asset_id!r} "
                            f"for model {asset_profile.model_id!r}"
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            issues.append(
                RuntimeValidationIssue(
                    code="runtime_assets_missing",
                    field="execution.assets",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )

    if binding_id:
        try:
            binding = resolve_pipeline_binding(binding_id, root=binding_root)
            if binding.model_id != profile.model_id:
                issues.append(
                    RuntimeValidationIssue(
                        code="pipeline_binding_model_mismatch",
                        field="execution.pipeline_binding",
                        message=(
                            f"runtime profile {profile.model_id!r} references binding {binding_id!r} "
                            f"for model {binding.model_id!r}"
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            issues.append(
                RuntimeValidationIssue(
                    code="pipeline_binding_missing",
                    field="execution.pipeline_binding",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )

    return tuple(issues)


def validate_pipeline_aliases_against_bindings(
    *,
    alias_root: str | None = None,
    binding_root: str | None = None,
) -> tuple[RuntimeValidationIssue, ...]:
    """Validate that data-backed aliases point at known pipeline bindings.

    Checks that each alias group's canonical ID resolves to a valid binding
    and that non-canonical aliases resolve to the same binding as the canonical.

    Args:
        alias_root: Optional root override for pipeline alias manifest lookup.
        binding_root: Optional root override for pipeline binding manifest lookup.

    Returns:
        A tuple of :class:`RuntimeValidationIssue` instances (may be empty).
    """

    issues: list[RuntimeValidationIssue] = []
    aliases = load_pipeline_alias_registry(alias_root)
    bindings = load_pipeline_binding_registry(binding_root)
    for group in aliases.list():
        try:
            canonical = bindings.get(group.canonical_id)
        except Exception as exc:  # noqa: BLE001
            issues.append(
                RuntimeValidationIssue(
                    code="pipeline_alias_unknown_canonical",
                    field="aliases",
                    message=f"{group.canonical_id!r}: {type(exc).__name__}: {exc}",
                )
            )
            continue
        for alias in group.aliases:
            try:
                resolved = bindings.get(alias)
            except KeyError:
                continue
            except Exception as exc:  # noqa: BLE001
                issues.append(
                    RuntimeValidationIssue(
                        code="pipeline_alias_unresolved",
                        field="aliases",
                        message=f"{alias!r}: {type(exc).__name__}: {exc}",
                    )
                )
                continue
            if resolved != canonical:
                issues.append(
                    RuntimeValidationIssue(
                        code="pipeline_alias_binding_mismatch",
                        field="aliases",
                        message=f"alias {alias!r} resolves to {resolved.model_id!r}, expected {canonical.model_id!r}",
                    )
                )
    return tuple(issues)


def validate_catalog_references(
    *,
    catalog_root: str | Path | None = None,
    binding_root: str | Path | None = None,
    profile_root: str | Path | None = None,
    catalog_entries: Iterable[Any] | None = None,
    bindings: PipelineBindingRegistry | None = None,
    runtime_profiles: Iterable[RuntimeProfile] | Mapping[str, RuntimeProfile] | None = None,
    runner_registry: Any = None,
) -> tuple[RuntimeValidationIssue, ...]:
    """Validate model-catalog references without importing model runtimes.

    Every explicitly declared ``pipeline_binding``, ``runtime_profile``, and
    ``runner_target`` on both top-level model entries and variants must resolve
    through the corresponding built-in registry.  Module-style runner targets
    use :meth:`ModelRunnerRegistry.resolve_key`, preserving the runner
    registry's lazy-import contract.

    Optional pre-loaded registries are accepted so aggregate validation and
    focused tests can avoid repeated manifest scans.
    """

    if catalog_entries is None:
        from ..catalog.zoo_registry import load_model_zoo_registry

        catalog_entries = load_model_zoo_registry(catalog_root).list()
    if bindings is None:
        bindings = load_pipeline_binding_registry(binding_root)
    if runtime_profiles is None:
        from .profiles import DEFAULT_RUNTIME_PROFILES_ROOT, load_runtime_profile_manifests

        runtime_profiles = load_runtime_profile_manifests(profile_root or DEFAULT_RUNTIME_PROFILES_ROOT)
    if runner_registry is None:
        from ..runners.registry import builtin_model_runner_registry

        runner_registry = builtin_model_runner_registry()

    profile_ids = _runtime_profile_reference_ids(runtime_profiles)
    issues: list[RuntimeValidationIssue] = []
    for entry in catalog_entries:
        references = ((f"catalog.{entry.model_id}", f"model catalog entry {entry.model_id!r}", entry),)
        variant_references = tuple(
            (
                f"catalog.{entry.model_id}.variants.{variant.variant_id}",
                f"model catalog entry {entry.model_id!r} variant {variant.variant_id!r}",
                variant,
            )
            for variant in entry.variants
        )
        for field_prefix, owner, reference in (*references, *variant_references):
            binding_id = _text_or_none(reference.pipeline_binding)
            if binding_id:
                try:
                    bindings.get(binding_id)
                except KeyError:
                    issues.append(
                        RuntimeValidationIssue(
                            code="catalog_pipeline_binding_missing",
                            field=f"{field_prefix}.pipeline_binding",
                            message=f"{owner} references unknown pipeline binding {binding_id!r}",
                        )
                    )

            profile_id = runtime_profile_id(reference.runtime_profile)
            if profile_id and profile_id not in profile_ids:
                issues.append(
                    RuntimeValidationIssue(
                        code="catalog_runtime_profile_missing",
                        field=f"{field_prefix}.runtime_profile",
                        message=f"{owner} references unknown runtime profile {profile_id!r}",
                    )
                )

            runner_target = _text_or_none(reference.runner_target)
            if runner_target:
                try:
                    runner_registry.resolve_key(runner_target)
                except (KeyError, ValueError):
                    issues.append(
                        RuntimeValidationIssue(
                            code="catalog_runner_target_unresolved",
                            field=f"{field_prefix}.runner_target",
                            message=f"{owner} references unresolved runner target {runner_target!r}",
                        )
                    )

    return tuple(issues)


def validate_runtime_registry(
    *,
    profile_root: str | Path | None = None,
) -> tuple[RuntimeValidationIssue, ...]:
    """Validate model-catalog, runtime-profile, and pipeline-alias references.

    Environments and assets are loaded once up front and shared across all
    catalog and per-profile checks.  This keeps reference validation linear in
    the number of manifests rather than re-scanning data trees per profile.
    """

    from .assets import load_runtime_asset_profiles
    from .environments import load_runtime_environment_profiles
    from .profiles import DEFAULT_RUNTIME_PROFILES_ROOT, load_runtime_profile_manifests

    issues: list[RuntimeValidationIssue] = []
    profiles = tuple(load_runtime_profile_manifests(profile_root or DEFAULT_RUNTIME_PROFILES_ROOT))
    issues.extend(validate_pipeline_aliases_against_bindings())
    issues.extend(validate_catalog_references(runtime_profiles=profiles))
    environments = load_runtime_environment_profiles()
    assets = load_runtime_asset_profiles(runtime_profiles={profile.model_id: profile for profile in profiles})
    for profile in profiles:
        issues.extend(validate_runtime_profile_references(profile, environments=environments, assets=assets))
    return tuple(issues)


# ── Private helpers ──────────────────────────────────────────


def _validate_artifact_defaults(profile: RuntimeProfile) -> tuple[RuntimeValidationIssue, ...]:
    """Validate that the artifact kind and filename settings are correct.

    Checks that ``artifact_kind`` is in ``KNOWN_ARTIFACT_KINDS``, that the
    filename is a relative path without ``..`` traversal, and that it
    includes a file extension.
    """
    issues: list[RuntimeValidationIssue] = []
    if profile.artifact_kind not in KNOWN_ARTIFACT_KINDS:
        issues.append(
            RuntimeValidationIssue(
                code="invalid_artifact_kind",
                field="artifact.kind",
                message=f"unknown artifact kind: {profile.artifact_kind!r}",
            )
        )
    filename = str(profile.artifact_filename or "")
    path = PurePosixPath(filename)
    if not filename:
        issues.append(
            RuntimeValidationIssue(
                code="invalid_artifact_filename",
                field="artifact.filename",
                message="artifact filename is required.",
            )
        )
    elif path.is_absolute() or ".." in path.parts:
        issues.append(
            RuntimeValidationIssue(
                code="invalid_artifact_filename",
                field="artifact.filename",
                message=f"artifact filename must be a relative path without parent traversal: {filename!r}",
            )
        )
    elif path.suffix == "":
        issues.append(
            RuntimeValidationIssue(
                code="invalid_artifact_filename",
                field="artifact.filename",
                message=f"artifact filename should include a file extension: {filename!r}",
            )
        )
    return tuple(issues)


def _text_or_none(value: Any) -> str | None:
    """Safely coerce any value to a stripped non-empty string, or return None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _runtime_profile_reference_ids(
    profiles: Iterable[RuntimeProfile] | Mapping[str, RuntimeProfile],
) -> frozenset[str]:
    """Return every key accepted as a catalog runtime-profile reference."""
    ids: set[str] = set()
    values: Iterable[RuntimeProfile]
    if isinstance(profiles, Mapping):
        ids.update(str(key) for key in profiles)
        values = profiles.values()
    else:
        values = profiles
    for profile in values:
        ids.add(profile.model_id)
        profile_id = _text_or_none(profile.execution.get("profile_id"))
        if profile_id:
            ids.add(profile_id)
    return frozenset(ids)


def _lookup_environment(
    environment_id: str,
    environments: Mapping[str, RuntimeEnvironmentProfile] | None,
    environment_root: str | None,
) -> RuntimeEnvironmentProfile:
    """Resolve an environment profile from the preloaded mapping or by re-scanning."""
    if environments is None:
        return load_runtime_environment_profile_by_id(environment_id, root=environment_root)
    if environment_id not in environments:
        raise KeyError(f"unknown runtime environment profile: {environment_id}")
    return environments[environment_id]


def _lookup_asset(
    asset_id: str,
    assets: Mapping[str, RuntimeAssetProfile] | None,
    asset_root: str | None,
    profile: RuntimeProfile,
) -> RuntimeAssetProfile:
    """Resolve an asset profile from the preloaded mapping or by re-scanning."""
    if assets is None:
        return load_runtime_asset_profile_by_id(
            asset_id,
            root=asset_root,
            runtime_profiles={profile.model_id: profile},
        )
    if asset_id not in assets:
        raise KeyError(f"unknown runtime asset profile: {asset_id}")
    return assets[asset_id]


__all__ = [
    "KNOWN_ARTIFACT_KINDS",
    "RuntimeValidationIssue",
    "validate_catalog_references",
    "validate_pipeline_aliases_against_bindings",
    "validate_runtime_profile_references",
    "validate_runtime_registry",
]
