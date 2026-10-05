"""Typed schema dataclasses for model-zoo catalog entries and their sub-structures.

Defines the frozen dataclasses — :class:`ModelSource`, :class:`CheckpointRef`,
:class:`DemoParitySpec`, :class:`ModelVariantSpec`, and :class:`ModelZooEntry` —
that represent a model's source availability, checkpoint references, demo parity,
variant-level integration specs, and the top-level zoo entry.  Each class
implements ``from_dict`` deserialization with extensive alias normalization so
that heterogeneous YAML manifests can be coerced into a consistent typed shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ...api.json_contract import JsonContract, require_mapping, to_plain
from ...utils import load_manifest

LOGGER = logging.getLogger(__name__)


# ── Canonical status sets ────────────────────────────────────

SOURCE_STATUSES = frozenset({"open_source", "api", "closed", "unknown"})
INTEGRATION_STATUSES = frozenset({"integrated", "planned", "blocked"})
DEMO_PARITY_STATUSES = frozenset({"verified", "pending", "not_applicable", "failed"})
RUNNER_ENTRY_KINDS = frozenset({"listed_only", "runner_candidate", "runnable_runner"})
_OPEN_SOURCE_ALIASES = frozenset(
    {
        "code_and_weights",
        "confirmed_official_code",
        "confirmed_official_code_and_data_in_github",
        "confirmed_official_code_and_hf_data",
        "confirmed_official_code_and_hf_metadata",
        "confirmed_official_code_and_partial_hf_data",
        "confirmed_official_code_model_and_data",
        "open",
        "open_weight",
        "open_weights",
        "open_source",
        "public",
        "source_available",
        "confirmed_official_github",
        "confirmed_public_hf",
    }
)
_UNKNOWN_SOURCE_ALIASES = frozenset(
    {
        "blocked_project_page_only",
        "confirmed_general_official_github_model_specific_source_pending",
        "open_weights_restricted",
        "paper_only",
        "unconfirmed",
    }
)
_API_ALIASES = frozenset({"api", "api_or_closed", "commercial_api", "restricted_api"})
_CLOSED_ALIASES = frozenset({"closed", "closed_source", "proprietary"})
# Registered integration-status vocabulary.  ``integrated`` means the in-tree
# integration surface (runner/pipeline wiring) exists; how far it has been
# validated is tracked separately by ``demo_parity``/``runner_parity``.  The
# checked-in catalog records that evidence directly in ``integration.status``
# strings, so every status that asserts an existing (at least statically
# validated) in-tree surface normalizes to ``integrated`` instead of silently
# degrading to ``planned`` and dropping the model from runnable listings.
_INTEGRATED_INTEGRATION_ALIASES = frozenset(
    {
        "runtime_ported",
        "route_ready",
        "runner_ready",
        "verified",
        "runnable",
        "component_verified",
        "checkpoint_gpu_validated",
        "all_released_checkpoints_gpu_validated",
        "checkpoint_backed_runtime_ready",
        "implemented_checkpoint_pending",
        "in_tree_checkpoint_gated_runtime",
        "in_tree_checkpoint_runtime",
        "in_tree_checkpoint_runtime_checkpoint_gpu_probe_verified",
        "in_tree_checkpoint_runtime_gpu_forward_server_init_verified",
        "in_tree_checkpoint_runtime_gpu_server_init_verified_rollout_pending",
        "in_tree_checkpoint_runtime_static_validated",
        "in_tree_checkpoint_runtime_static_verified",
        "in_tree_in_process_checkpoint_runtime",
        "in_tree_runtime_selected_checkpoint_gpu_validated",
        "in_tree_vendored_import_verified",
        "in_tree_vendored_checkpoint_pending",
        "in_tree_checkpoint_validation_pending",
        "in_tree_runtime_checkpoint_validation_pending",
    }
)

# ── Alias normalisation sets ─────────────────────────────────

_PENDING_INTEGRATION_ALIASES = frozenset(
    {
        "pending",
        "todo",
        "not_started",
        "not_applicable",
        # Metadata/staging states: catalog entry exists but no runnable
        # in-tree surface has landed (or its static validation is still in
        # progress), so they stay in the ``planned`` bucket.
        "metadata_only",
        "metadata_only_post_training_base",
        "checkpoint_asset_only",
        "checkpoint_assets_staged_cpu_schema_validated_gpu_pending",
        "entrypoint_profiled_runtime_port_pending",
        "in_tree_import_verification_in_progress",
    }
)
_PENDING_DEMO_ALIASES = frozenset(
    {
        "pending_checkpoint_and_demo",
        "pending_demo",
        "pending_parity",
        # Registered staging/component-level demo states from the checked-in
        # catalog.  They record partial evidence (implemented code, staged or
        # statically validated checkpoints, component-level checks) but NOT a
        # completed end-to-end demo parity run, so they normalize to
        # ``pending`` without triggering the unregistered-status warning.
        "checkpoint_environment_required",
        "checkpoint_required",
        "component_verified_worldgen_pending",
        "configured",
        "implemented",
        "implemented_checkpoint_pending",
        "implemented_hidden_weights_pending",
        "implemented_local_extension_pending",
        "implemented_pending_gpu_parity",
        "implemented_pending_native_gpu_parity",
        "implemented_public_checkpoint_not_staged",
        "in_tree_checkpoint_runtime",
        "in_tree_inference_ready_checkpoint_pending",
        "in_tree_vendor_ready_checkpoint_pending",
        "integrated",
        "native_checkpoint_layout_validated",
        "native_checkpoint_validated",
        "native_pipeline_cuda_artifact_pending",
        "not_recorded",
        "official_in_process_runtime_static_checkpoint_validation_complete",
        "partial",
        "partial_studio_visual_validation_verified",
        "planned",
        "runtime_ported",
        "static_runtime_verified",
        "static_runtime_verified_checkpoint_assets_staged",
        "structural_validation_passed",
    }
)
# Demo states that record a completed checkpoint-backed GPU demo/parity run in
# the catalog.  Only statuses that explicitly claim that end-to-end evidence
# normalize to ``verified``; component/static validation stays ``pending``.
_VERIFIED_DEMO_ALIASES = frozenset(
    {
        "full_demo_verified",
        "full_default_gpu_parity_verified",
        "passed",
        "checkpoint_verified",
        "checkpoint_backed_verified",
        "checkpoint_gpu_validated",
        "selected_checkpoint_gpu_validated",
        "all_declared_variants_checkpoint_gpu_validated",
        "all_released_checkpoints_gpu_validated",
        "robotwin_and_umi_checkpoints_gpu_validated",
        "converted_checkpoint_strict_restore_and_gpu_action_probe_validated",
    }
)

# ── Type aliases and shared helpers ──────────────────────────

JsonValue = Any
JsonSerializable = JsonContract
_to_plain = to_plain
_require_mapping = require_mapping


# ── Coercion helpers ─────────────────────────────────────────

def _optional_str(value: Any) -> str | None:
    """Coerce a value to string or return None if None."""
    if value is None:
        return None
    return str(value)


def _optional_command(value: Any) -> str | tuple[str, ...] | None:
    """Coerce a value to a command string or a tuple of command strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return str(value)


def _optional_float(value: Any) -> float | None:
    """Coerce a value to float or return None."""
    if value is None:
        return None
    return float(value)


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    """Coerce any scalar or iterable value to a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _tuple_of_artifacts(value: Any) -> tuple[JsonValue, ...]:
    """Coerce artifact value representations into a tuple of plain JsonValues."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return (_to_plain(value),)
    if isinstance(value, (list, tuple)):
        return tuple(_to_plain(item) for item in value)
    return (str(value),)


def _bool(value: Any) -> bool:
    """Coerce a value to boolean."""
    return bool(value)


def _validate_status(value: str, allowed: frozenset[str], context: str) -> str:
    """Validate that a status string is within a set of allowed values."""
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{context} must be one of: {allowed_values}. Got {value!r}.")
    return value


# ── Status normalisation ─────────────────────────────────────

def _normalize_source_status(value: Any) -> str:
    """Normalize and map varying source status strings/aliases to canonical states.

    Accepts raw strings, nested mappings (e.g. ``{"status": "open_source"}``),
    and well-known alias sets like ``"open_weights"`` → ``"open_source"``.
    """
    if isinstance(value, Mapping):
        value = (
            value.get("status")
            or value.get("source")
            or value.get("source_status")
            or value.get("availability")
            or value.get("release_type")
            or "unknown"
        )
    normalized = str(value or "unknown").strip().lower()
    if normalized in SOURCE_STATUSES:
        return normalized
    if normalized in _OPEN_SOURCE_ALIASES:
        return "open_source"
    if normalized in _UNKNOWN_SOURCE_ALIASES:
        return "unknown"
    if normalized in _API_ALIASES:
        return "api"
    if normalized in _CLOSED_ALIASES:
        return "closed"
    _warn_unknown_status_once("source_status", normalized, "unknown")
    return "unknown"


_WARNED_UNKNOWN_STATUSES: set[tuple[str, str]] = set()


def _warn_unknown_status_once(kind: str, raw_value: str, fallback: str) -> None:
    """Surface unknown status literals without flooding the log.

    The checked-in catalog currently carries dozens of freeform status strings
    (data cleanup is owned by the catalog data track), so the first unknown
    value logs a WARNING pointing at DEBUG for the full list; every unique
    value is logged once at DEBUG.
    """
    key = (kind, raw_value)
    if key in _WARNED_UNKNOWN_STATUSES:
        return
    first_unknown = not _WARNED_UNKNOWN_STATUSES
    _WARNED_UNKNOWN_STATUSES.add(key)
    if first_unknown:
        LOGGER.warning(
            "Model catalog contains unregistered status values (first: %s %r, normalized to %r). "
            "Enable DEBUG logging for the full list; register intentional values in schema.py.",
            kind,
            raw_value,
            fallback,
        )
    LOGGER.debug(
        "Unknown %s %r in model catalog; normalizing to %r.",
        kind,
        raw_value,
        fallback,
    )


def _normalize_integration_status(value: Any) -> str:
    """Normalize and map varying integration status strings/aliases to canonical states.

    Accepts raw strings, nested mappings, and aliases like ``"runtime_ported"``
    → ``"integrated"``, ``"pending"`` → ``"planned"``.  Unregistered values
    fall back to ``"planned"`` and are logged (see
    :func:`_warn_unknown_status_once`) so a typo cannot make a model silently
    disappear from runnable listings.
    """
    if isinstance(value, Mapping):
        value = value.get("status", "planned")
    normalized = str(value or "planned").strip().lower()
    if normalized in INTEGRATION_STATUSES:
        return normalized
    if normalized in _INTEGRATED_INTEGRATION_ALIASES:
        return "integrated"
    if normalized.startswith("blocked"):
        return "blocked"
    if normalized in _PENDING_INTEGRATION_ALIASES or normalized.startswith("pending"):
        return "planned"
    _warn_unknown_status_once("integration_status", normalized, "planned")
    return "planned"


def _normalize_demo_status(value: Any) -> str:
    """Normalize and map varying demo parity status strings/aliases to canonical states.

    Accepts raw strings, nested mappings, and aliases like
    ``"pending_demo"`` → ``"pending"``.  Unregistered values fall back to
    ``"pending"`` and are logged (see :func:`_warn_unknown_status_once`).
    """
    if isinstance(value, Mapping):
        value = value.get("status", "not_applicable")
    normalized = str(value or "not_applicable").strip().lower()
    if normalized in DEMO_PARITY_STATUSES:
        return normalized
    if normalized in _VERIFIED_DEMO_ALIASES:
        return "verified"
    if normalized in _PENDING_DEMO_ALIASES or normalized.startswith("pending"):
        return "pending"
    _warn_unknown_status_once("demo_status", normalized, "pending")
    return "pending"


# ── Text and URL helpers ─────────────────────────────────────

def _first_text(*values: Any) -> str | None:
    """Return the first non-empty string value found in the inputs."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _slug(value: Any) -> str | None:
    """Generate a clean alphanumeric URL slug from a string value."""
    text = _first_text(value)
    if not text:
        return None
    if "/" in text:
        text = text.strip("/").split("/")[-1]
    chars: list[str] = []
    for char in text.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "_", "-", ".", "/", "(", ")"} and (not chars or chars[-1] != "-"):
            chars.append("-")
    slug = "".join(chars).strip("-")
    return slug or None


def _repo_id_from_hf_url(value: Any) -> str | None:
    """Extract a Hugging Face repo ID from a full huggingface.co URL."""
    if not isinstance(value, str):
        return None
    marker = "huggingface.co/"
    if marker not in value:
        return None
    suffix = value.split(marker, 1)[1].strip("/")
    if suffix.startswith("spaces/"):
        return None
    parts = suffix.split("/")
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])


# ── Entry extraction helpers ─────────────────────────────────

def _github_url_from_sources(entry: Mapping[str, Any]) -> str | None:
    """Resolve the official GitHub repository URL from various manifest metadata structures."""
    direct = _first_text(entry.get("official_repo_url"))
    if direct:
        return direct

    source = entry.get("source")
    if isinstance(source, Mapping):
        direct = _first_text(source.get("official_repo_url"))
        if direct:
            return direct

    source_status = entry.get("source_status")
    if isinstance(source_status, Mapping):
        github = source_status.get("github")
        if isinstance(github, Mapping):
            direct = _first_text(github.get("url"))
            if direct:
                return direct

    official_sources = entry.get("official_sources")
    if isinstance(official_sources, Mapping):
        github = official_sources.get("github")
        if isinstance(github, Mapping):
            return _first_text(github.get("url"))
        if isinstance(github, str):
            return github
    sources = entry.get("sources")
    if isinstance(sources, Mapping):
        github = sources.get("github")
        if isinstance(github, Mapping):
            return _first_text(github.get("url"))
        if isinstance(github, str):
            return github
    return None


def _hf_source_status_allows_checkpoint(item: Mapping[str, Any]) -> bool:
    """Check if a Hugging Face source dictionary status allows loading checkpoints."""
    status = str(item.get("status", "confirmed") or "confirmed").strip().lower()
    blocked_statuses = {
        "access_required_or_private",
        "blocked",
        "not_found",
        "private",
        "unconfirmed",
    }
    return status not in blocked_statuses


def _hf_repo_ids_from_entry(entry: Mapping[str, Any]) -> list[str]:
    """Retrieve all Hugging Face repository IDs referenced inside a model entry.

    Inspects ``hf_repo_id``, ``source``, ``checkpoint``, ``checkpoints``,
    ``official_sources``, and ``sources`` sub-structures, collecting every
    distinct HuggingFace repo ID encountered.
    """
    repo_ids: list[str] = []

    # Inner helper: add a repo ID (from URL or raw text) to the dedup list.
    def add(value: Any) -> None:
        repo_id = _repo_id_from_hf_url(value) or _first_text(value)
        if repo_id and repo_id not in repo_ids:
            repo_ids.append(repo_id)

    add(entry.get("hf_repo_id"))

    source = entry.get("source")
    if isinstance(source, Mapping):
        add(source.get("hf_repo_id"))

    checkpoint = entry.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        add(checkpoint.get("hf_repo_id"))
        for repo in checkpoint.get("repos") or ():
            if isinstance(repo, Mapping):
                add(repo.get("id") or repo.get("repo_id"))
            else:
                add(repo)

    raw_checkpoints = entry.get("checkpoints")
    if isinstance(raw_checkpoints, Mapping):
        checkpoint_items = raw_checkpoints.values()
    else:
        checkpoint_items = raw_checkpoints or ()
    for checkpoint_item in checkpoint_items:
        if isinstance(checkpoint_item, Mapping):
            add(checkpoint_item.get("repo_id") or checkpoint_item.get("id"))
        else:
            add(checkpoint_item)

    # official_sources and sources share the same HF payload shape; keep the
    # precedence order (official first) while parsing both with one code path.
    for source_key in ("official_sources", "sources"):
        container = entry.get(source_key)
        if not isinstance(container, Mapping):
            continue
        huggingface = container.get("huggingface")
        if isinstance(huggingface, (list, tuple)):
            for item in huggingface:
                if isinstance(item, Mapping):
                    if _hf_source_status_allows_checkpoint(item):
                        add(item.get("repo_id") or item.get("url"))
                else:
                    add(item)
        elif isinstance(huggingface, Mapping):
            if _hf_source_status_allows_checkpoint(huggingface):
                add(huggingface.get("repo_id") or huggingface.get("url"))
        else:
            add(huggingface)

    return repo_ids


def _license_from_entry(entry: Mapping[str, Any]) -> str | None:
    """Resolve the software/data license identifier from a model entry."""
    direct = _first_text(entry.get("license"))
    if direct:
        return direct
    source = entry.get("source")
    if isinstance(source, Mapping):
        direct = _first_text(source.get("license"))
        if direct:
            return direct
    availability = entry.get("availability")
    if isinstance(availability, Mapping):
        direct = _first_text(availability.get("license"))
        if direct:
            return direct
    checkpoint = entry.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        repos = checkpoint.get("repos") or ()
        for repo in repos:
            if isinstance(repo, Mapping):
                direct = _first_text(repo.get("license"))
                if direct:
                    return direct
    return None


def _entry_notes(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """Retrieve and format a list of informational notes from a model entry."""
    notes = list(_tuple_of_str(entry.get("notes")))
    for key in ("confirmation",):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            notes.append(value.strip())
    integration = entry.get("integration")
    if isinstance(integration, Mapping):
        reason_prefix = "blocked" if _normalize_integration_status(integration) == "blocked" else "runtime_gated"
        for reason in _tuple_of_str(integration.get("blocked_reasons")):
            notes.append(f"{reason_prefix}: {reason}")
    return tuple(notes)


# ── Dataclass definitions ────────────────────────────────────

@dataclass(frozen=True)
class ModelSource(JsonSerializable):
    """Source availability and release metadata for a model.

    Attributes:
        status: Canonical source status (``"open_source"``, ``"api"``,
            ``"closed"``, or ``"unknown"``).
        official_repo_url: URL of the official repository (typically GitHub).
        hf_repo_id: Primary HuggingFace repository identifier.
        license: Software/data license identifier (e.g. ``"apache-2.0"``).
        requires_auth: Whether accessing the source requires authentication.
        notes: Additional informational notes about the source.
    """

    status: str = "unknown"
    official_repo_url: str | None = None
    hf_repo_id: str | None = None
    license: str | None = None
    requires_auth: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Post-initialization validation and coercion."""
        object.__setattr__(self, "status", _validate_status(str(self.status), SOURCE_STATUSES, "ModelSource.status"))
        object.__setattr__(self, "official_repo_url", _optional_str(self.official_repo_url))
        object.__setattr__(self, "hf_repo_id", _optional_str(self.hf_repo_id))
        object.__setattr__(self, "license", _optional_str(self.license))
        object.__setattr__(self, "requires_auth", _bool(self.requires_auth))
        object.__setattr__(self, "notes", _tuple_of_str(self.notes))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSource":
        """Deserialize a ModelSource instance from a dictionary."""
        source = _require_mapping(data, "ModelSource.from_dict")

        return cls(
            status=_normalize_source_status(source.get("status", source.get("source_status", "unknown"))),
            official_repo_url=source.get("official_repo_url"),
            hf_repo_id=source.get("hf_repo_id"),
            license=source.get("license"),
            requires_auth=source.get("requires_auth", False),
            notes=_tuple_of_str(source.get("notes")),
        )


@dataclass(frozen=True)
class CheckpointRef(JsonSerializable):
    """Hugging Face checkpoint repository reference with revision and licensing.

    Attributes:
        hf_repo_id: HuggingFace repository identifier (e.g. ``"org/model-name"``).
        revision: Git revision, SHA, or branch tag.
        license: License attached to the checkpoint weights.
        private: Whether the repository is private.
        gated: Gated-access policy value (may be ``True``, ``"false"``, etc.).
        path: Sub-path within the repository for specific files.
        filename: Specific filename within the repository.
        estimated_size_gb: Estimated download size in gigabytes.
        requires_auth: Whether downloading requires HuggingFace auth.
        notes: Additional informational notes about the checkpoint.
    """

    hf_repo_id: str | None = None
    revision: str | None = None
    license: str | None = None
    private: bool | None = None
    gated: JsonValue | None = None
    path: str | None = None
    filename: str | None = None
    estimated_size_gb: float | None = None
    requires_auth: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and construct attributes, establishing auth requirements."""
        object.__setattr__(self, "hf_repo_id", _optional_str(self.hf_repo_id))
        object.__setattr__(self, "revision", _optional_str(self.revision))
        object.__setattr__(self, "license", _optional_str(self.license))
        if self.private is not None:
            object.__setattr__(self, "private", _bool(self.private))
        object.__setattr__(self, "path", _optional_str(self.path))
        object.__setattr__(self, "filename", _optional_str(self.filename))
        object.__setattr__(self, "estimated_size_gb", _optional_float(self.estimated_size_gb))
        # NOTE: gated access implies auth even when requires_auth is not set explicitly.
        gated_requires_auth = self.gated not in (None, False, "false", "False", "none", "None")
        object.__setattr__(
            self,
            "requires_auth",
            _bool(self.requires_auth) or self.private is True or gated_requires_auth,
        )
        object.__setattr__(self, "notes", _tuple_of_str(self.notes))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckpointRef":
        """Deserialize a CheckpointRef from a mapping dictionary."""
        checkpoint = _require_mapping(data, "CheckpointRef.from_dict")
        return cls(
            hf_repo_id=checkpoint.get("hf_repo_id") or checkpoint.get("repo_id") or checkpoint.get("id"),
            revision=checkpoint.get("revision", checkpoint.get("sha", checkpoint.get("commit"))),
            license=checkpoint.get("license"),
            private=checkpoint.get("private"),
            gated=checkpoint.get("gated"),
            path=checkpoint.get("path"),
            filename=checkpoint.get("filename"),
            estimated_size_gb=checkpoint.get("estimated_size_gb"),
            requires_auth=checkpoint.get("requires_auth", False),
            notes=_tuple_of_str(checkpoint.get("notes")),
        )


# ── Checkpoint assembly helpers ───────────────────────────────

def _checkpoint_ref_from_repo_mapping(repo: Mapping[str, Any]) -> CheckpointRef:
    """Build a CheckpointRef from a repository sub-mapping structure."""
    return CheckpointRef.from_dict(
        {
            "hf_repo_id": repo.get("hf_repo_id") or repo.get("repo_id") or repo.get("id"),
            "revision": repo.get("revision", repo.get("sha", repo.get("commit"))),
            "license": repo.get("license"),
            "private": repo.get("private"),
            "gated": repo.get("gated"),
            "path": repo.get("path"),
            "filename": repo.get("filename"),
            "estimated_size_gb": repo.get("estimated_size_gb"),
            "requires_auth": repo.get("requires_auth", False),
            "notes": repo.get("notes"),
        }
    )


def _dedupe_checkpoint_refs(refs: list[CheckpointRef]) -> tuple[CheckpointRef, ...]:
    """Remove duplicate checkpoint references based on repository/revision coordinates."""
    deduped: list[CheckpointRef] = []
    seen: set[tuple[str | None, str | None, str | None, str | None]] = set()
    for ref in refs:
        key = (ref.hf_repo_id, ref.revision, ref.path, ref.filename)
        if not ref.hf_repo_id or key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


def _checkpoint_refs_from_entry(entry: Mapping[str, Any], checkpoint_data: Mapping[str, Any]) -> tuple[CheckpointRef, ...]:
    """Assemble all checkpoint references defined under various fields in a model entry."""
    refs: list[CheckpointRef] = []

    if "checkpoint_refs" in entry:
        for item in entry.get("checkpoint_refs") or ():
            if isinstance(item, Mapping):
                refs.append(CheckpointRef.from_dict(item))
        return _dedupe_checkpoint_refs(refs)

    if checkpoint_data.get("hf_repo_id") and not checkpoint_data.get("repos"):
        refs.append(CheckpointRef.from_dict(checkpoint_data))

    repos = checkpoint_data.get("repos")
    if isinstance(repos, (list, tuple)):
        for repo in repos:
            if isinstance(repo, Mapping):
                refs.append(_checkpoint_ref_from_repo_mapping(repo))
            else:
                refs.append(CheckpointRef(hf_repo_id=str(repo)))

    raw_checkpoints = entry.get("checkpoints")
    if isinstance(raw_checkpoints, Mapping):
        checkpoint_items = raw_checkpoints.values()
    else:
        checkpoint_items = raw_checkpoints or ()
    for item in checkpoint_items:
        if isinstance(item, Mapping):
            refs.append(_checkpoint_ref_from_repo_mapping(item))
        else:
            refs.append(CheckpointRef(hf_repo_id=str(item)))

    # official_sources historically accepts more HF key spellings than sources;
    # both share one parsing path with their accepted keys parameterized.
    for source_key, hf_keys in (
        ("official_sources", ("huggingface", "huggingface_models", "hf_models", "models")),
        ("sources", ("huggingface",)),
    ):
        container = entry.get(source_key)
        if not isinstance(container, Mapping):
            continue
        for key in hf_keys:
            values = container.get(key)
            items = values if isinstance(values, (list, tuple)) else (values,)
            for item in items:
                if isinstance(item, Mapping):
                    if _hf_source_status_allows_checkpoint(item):
                        refs.append(_checkpoint_ref_from_repo_mapping(item))
                elif item is not None:
                    refs.append(CheckpointRef(hf_repo_id=_repo_id_from_hf_url(item) or str(item)))

    if entry.get("hf_repo_id"):
        refs.append(CheckpointRef(hf_repo_id=str(entry["hf_repo_id"])))

    return _dedupe_checkpoint_refs(refs)


@dataclass(frozen=True)
class DemoParitySpec(JsonSerializable):
    """Specification for running and verifying a baseline model inference demo.

    Attributes:
        status: Canonical demo parity status (``"verified"``, ``"pending"``,
            ``"not_applicable"``, or ``"failed"``).
        demo_command: Shell command or command tuple used to run the demo.
        expected_artifacts: Artifact names/values the demo should produce.
        notes: Additional informational notes about the demo parity check.
    """

    status: str = "not_applicable"
    demo_command: str | tuple[str, ...] | None = None
    expected_artifacts: tuple[JsonValue, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and construct status, command, and notes attributes."""
        object.__setattr__(
            self,
            "status",
            _validate_status(str(self.status), DEMO_PARITY_STATUSES, "DemoParitySpec.status"),
        )
        object.__setattr__(self, "demo_command", _optional_command(self.demo_command))
        object.__setattr__(self, "expected_artifacts", _tuple_of_artifacts(self.expected_artifacts))
        object.__setattr__(self, "notes", _tuple_of_str(self.notes))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DemoParitySpec":
        """Deserialize a DemoParitySpec instance from a mapping dictionary."""
        demo = _require_mapping(data, "DemoParitySpec.from_dict")
        raw_status = demo.get("status", "not_applicable")
        notes = list(_tuple_of_str(demo.get("notes")))
        for reason in _tuple_of_str(demo.get("blocked_reasons")):
            notes.append(f"blocked: {reason}")
        return cls(
            status=_normalize_demo_status(raw_status),
            demo_command=demo.get("demo_command"),
            expected_artifacts=_tuple_of_artifacts(demo.get("expected_artifacts")),
            notes=tuple(notes),
        )


@dataclass(frozen=True)
class ModelVariantSpec(JsonSerializable):
    """Variant-level integration and runner specifications for a model family.

    Attributes:
        variant_id: Unique identifier for this variant within its model entry.
        task: Primary task this variant addresses (e.g. ``"video_generation"``).
        provider: Provider override for this variant.
        checkpoint_refs: HuggingFace checkpoint references specific to this variant.
        integration_status: Canonical integration status (``"integrated"``,
            ``"planned"``, or ``"blocked"``).
        runner_target: Dotted ``module:qualname`` import for the runner.
        pipeline_target: Dotted ``module:qualname`` import for the pipeline.
        pipeline_binding: Binding key linking the pipeline to the runner.
        runtime_profile: Runtime configuration profile identifier.
        demo_parity: Demo parity specification for this variant.
        runner_parity: Runner parity specification for this variant.
        min_vram_gb: Minimum VRAM in GB required to run this variant.
        requires_auth: Whether any checkpoint in this variant requires auth.
        notes: Additional informational notes about the variant.
    """

    variant_id: str
    task: str | None = None
    provider: str | None = None
    checkpoint_refs: tuple[CheckpointRef, ...] = ()
    integration_status: str = "planned"
    runner_target: str | None = None
    pipeline_target: str | None = None
    pipeline_binding: str | None = None
    runtime_profile: str | None = None
    demo_parity: DemoParitySpec = field(default_factory=DemoParitySpec)
    runner_parity: DemoParitySpec = field(default_factory=DemoParitySpec)
    min_vram_gb: float | None = None
    requires_auth: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate variant specs and establish authorization flags."""
        if not self.variant_id:
            raise ValueError("ModelVariantSpec.variant_id is required.")
        object.__setattr__(self, "variant_id", str(self.variant_id))
        object.__setattr__(self, "task", _optional_str(self.task))
        object.__setattr__(self, "provider", _optional_str(self.provider))
        object.__setattr__(
            self,
            "checkpoint_refs",
            tuple(item if isinstance(item, CheckpointRef) else CheckpointRef.from_dict(item) for item in self.checkpoint_refs),
        )
        object.__setattr__(
            self,
            "integration_status",
            _validate_status(str(self.integration_status), INTEGRATION_STATUSES, "ModelVariantSpec.integration_status"),
        )
        object.__setattr__(self, "runner_target", _optional_str(self.runner_target))
        object.__setattr__(self, "pipeline_target", _optional_str(self.pipeline_target))
        object.__setattr__(self, "pipeline_binding", _optional_str(self.pipeline_binding))
        object.__setattr__(self, "runtime_profile", _optional_str(self.runtime_profile))
        object.__setattr__(self, "min_vram_gb", _optional_float(self.min_vram_gb))
        object.__setattr__(
            self,
            "requires_auth",
            _bool(self.requires_auth) or any(ref.requires_auth for ref in self.checkpoint_refs),
        )
        object.__setattr__(self, "notes", _tuple_of_str(self.notes))

    @property
    def verification_status(self) -> str:
        """Get the variant verification status from runner parity."""
        return self.runner_parity.status

    @property
    def runner_entry_kind(self) -> str:
        """Get the classification of runner entry based on targets and integration status."""
        if not self.runner_target:
            return "listed_only"
        if self.integration_status == "integrated":
            return "runnable_runner"
        return "runner_candidate"

    @property
    def is_runnable_runner_entry(self) -> bool:
        """Check if this variant has a runnable runner target."""
        return self.runner_entry_kind == "runnable_runner"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelVariantSpec":
        """Deserialize a ModelVariantSpec from a mapping dictionary."""
        variant = _require_mapping(data, "ModelVariantSpec.from_dict")
        integration = variant.get("integration") if isinstance(variant.get("integration"), Mapping) else {}
        checkpoint_refs: list[CheckpointRef] = []
        for key in ("checkpoint_refs", "checkpoints"):
            for item in variant.get(key) or ():
                if isinstance(item, Mapping):
                    checkpoint_refs.append(_checkpoint_ref_from_repo_mapping(item))
                else:
                    checkpoint_refs.append(CheckpointRef(hf_repo_id=str(item)))
        checkpoint = variant.get("checkpoint")
        if isinstance(checkpoint, Mapping):
            checkpoint_refs.append(CheckpointRef.from_dict(checkpoint))
        elif isinstance(checkpoint, str) and checkpoint.strip():
            checkpoint_refs.append(CheckpointRef(hf_repo_id=checkpoint.strip()))

        task = variant.get("task")
        tasks = variant.get("tasks")
        if task is None:
            if isinstance(tasks, str):
                task = tasks
            elif isinstance(tasks, (list, tuple)):
                task = _first_text(*tasks)
        demo_raw = variant.get("demo_parity", {})
        runner_raw = variant.get("runner_parity", {})
        demo_data = demo_raw if isinstance(demo_raw, Mapping) else {"status": demo_raw}
        runner_data = runner_raw if isinstance(runner_raw, Mapping) else {"status": runner_raw}
        variant_id = (
            _first_text(variant.get("variant_id"), variant.get("id"))
            or _slug(variant.get("name"))
            or _slug(checkpoint)
        )
        return cls(
            variant_id=str(variant_id or ""),
            task=task,
            provider=variant.get("provider"),
            checkpoint_refs=_dedupe_checkpoint_refs(checkpoint_refs),
            integration_status=_normalize_integration_status(
                variant.get("integration", variant.get("integration_status", "planned"))
            ),
            runner_target=variant.get("runner_target") or integration.get("runner"),
            pipeline_target=variant.get("pipeline_target") or integration.get("pipeline_target"),
            pipeline_binding=variant.get("pipeline_binding") or integration.get("pipeline_binding"),
            runtime_profile=variant.get("runtime_profile") or integration.get("runtime_profile"),
            demo_parity=DemoParitySpec.from_dict(demo_data),
            runner_parity=DemoParitySpec.from_dict(runner_data),
            min_vram_gb=variant.get("min_vram_gb", variant.get("minimum_vram_gb")),
            requires_auth=variant.get("requires_auth", False),
            notes=_tuple_of_str(variant.get("notes")),
        )


@dataclass(frozen=True)
class ModelZooEntry(JsonSerializable):
    """Model-zoo catalog entry describing tasks, sources, checkpoints, variants, and integration.

    Attributes:
        model_id: Primary identifier for the model (required).
        name: Human-readable display name.
        aliases: Alternative identifiers for registry lookup.
        tasks: Supported task family names (e.g. ``"world_generation"``).
        output_artifacts: Expected output artifact types.
        provider: Provider or organisation name.
        source: :class:`ModelSource` describing availability and licensing.
        checkpoint: Primary :class:`CheckpointRef` for the default weights.
        checkpoint_refs: All :class:`CheckpointRef` instances for this entry.
        variants: :class:`ModelVariantSpec` instances for sub-configurations.
        integration_status: Canonical integration status across all variants.
        install_profile: Installation profile identifier.
        runner_target: Dotted ``module:qualname`` import for the default runner.
        pipeline_target: Dotted ``module:qualname`` import for the default pipeline.
        pipeline_binding: Binding key linking the default pipeline to the runner.
        runtime_profile: Runtime configuration profile identifier.
        min_vram_gb: Minimum VRAM in GB required to run this model.
        demo_parity: :class:`DemoParitySpec` for baseline demo verification.
        runner_parity: :class:`DemoParitySpec` for runner-level parity checks.
        notes: Additional informational notes about the model entry.
    """

    model_id: str
    name: str | None = None
    aliases: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    output_artifacts: tuple[str, ...] = ()
    provider: str | None = None
    source: ModelSource = field(default_factory=ModelSource)
    checkpoint: CheckpointRef = field(default_factory=CheckpointRef)
    checkpoint_refs: tuple[CheckpointRef, ...] = ()
    variants: tuple[ModelVariantSpec, ...] = ()
    integration_status: str = "planned"
    install_profile: str | None = None
    runner_target: str | None = None
    pipeline_target: str | None = None
    pipeline_binding: str | None = None
    runtime_profile: str | None = None
    min_vram_gb: float | None = None
    demo_parity: DemoParitySpec = field(default_factory=DemoParitySpec)
    runner_parity: DemoParitySpec = field(default_factory=DemoParitySpec)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and construct ModelZooEntry attributes."""
        if not self.model_id:
            raise ValueError("ModelZooEntry.model_id is required.")
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "name", _optional_str(self.name))
        object.__setattr__(self, "aliases", _tuple_of_str(self.aliases))
        object.__setattr__(self, "tasks", _tuple_of_str(self.tasks))
        object.__setattr__(self, "output_artifacts", _tuple_of_str(self.output_artifacts))
        object.__setattr__(self, "provider", _optional_str(self.provider))
        object.__setattr__(
            self,
            "checkpoint_refs",
            tuple(item if isinstance(item, CheckpointRef) else CheckpointRef.from_dict(item) for item in self.checkpoint_refs),
        )
        object.__setattr__(
            self,
            "variants",
            tuple(item if isinstance(item, ModelVariantSpec) else ModelVariantSpec.from_dict(item) for item in self.variants),
        )
        object.__setattr__(
            self,
            "integration_status",
            _validate_status(str(self.integration_status), INTEGRATION_STATUSES, "ModelZooEntry.integration_status"),
        )
        object.__setattr__(self, "install_profile", _optional_str(self.install_profile))
        object.__setattr__(self, "runner_target", _optional_str(self.runner_target))
        object.__setattr__(self, "pipeline_target", _optional_str(self.pipeline_target))
        object.__setattr__(self, "pipeline_binding", _optional_str(self.pipeline_binding))
        object.__setattr__(self, "runtime_profile", _optional_str(self.runtime_profile))
        object.__setattr__(self, "min_vram_gb", _optional_float(self.min_vram_gb))
        object.__setattr__(self, "notes", _tuple_of_str(self.notes))

    @property
    def source_status(self) -> str:
        """Get the model source status."""
        return self.source.status

    @property
    def official_repo_url(self) -> str | None:
        """Get the official repository URL."""
        return self.source.official_repo_url

    @property
    def hf_repo_id(self) -> str | None:
        """Get the primary Hugging Face repo ID."""
        if self.checkpoint.hf_repo_id:
            return self.checkpoint.hf_repo_id
        if self.checkpoint_refs:
            return self.checkpoint_refs[0].hf_repo_id
        return self.source.hf_repo_id

    @property
    def hf_repo_ids(self) -> tuple[str, ...]:
        """Compile all unique Hugging Face repository IDs associated with this entry."""
        ids: list[str] = []
        for ref in (*self.checkpoint_refs, self.checkpoint):
            if ref.hf_repo_id and ref.hf_repo_id not in ids:
                ids.append(ref.hf_repo_id)
        for variant in self.variants:
            for ref in variant.checkpoint_refs:
                if ref.hf_repo_id and ref.hf_repo_id not in ids:
                    ids.append(ref.hf_repo_id)
        if self.source.hf_repo_id and self.source.hf_repo_id not in ids:
            ids.append(self.source.hf_repo_id)
        return tuple(ids)

    @property
    def license(self) -> str | None:
        """Get the software/data license identifier."""
        return self.source.license

    @property
    def requires_auth(self) -> bool:
        """Check if any of the sources, checkpoints, or variants require authentication."""
        return (
            self.source.requires_auth
            or self.checkpoint.requires_auth
            or any(ref.requires_auth for ref in self.checkpoint_refs)
            or any(variant.requires_auth for variant in self.variants)
        )

    @property
    def estimated_size_gb(self) -> float | None:
        """Get the estimated checkpoint repository size in GB."""
        return self.checkpoint.estimated_size_gb

    @property
    def demo_command(self) -> str | tuple[str, ...] | None:
        """Get the baseline demo execution command."""
        return self.demo_parity.demo_command

    @property
    def expected_artifacts(self) -> tuple[JsonValue, ...]:
        """Get the expected output artifacts of the baseline demo."""
        return self.demo_parity.expected_artifacts

    @property
    def runner_demo_command(self) -> str | tuple[str, ...] | None:
        """Get the integrated runner demo command."""
        return self.runner_parity.demo_command

    @property
    def runner_expected_artifacts(self) -> tuple[JsonValue, ...]:
        """Get the expected output artifacts of the runner demo."""
        return self.runner_parity.expected_artifacts

    @property
    def verification_status(self) -> str:
        """Get the overall verification status from runner parity."""
        return self.runner_parity.status

    @property
    def runner_entry_kind(self) -> str:
        """Get the overall classification of the runner entry kind."""
        if self.runner_target and self.integration_status == "integrated":
            return "runnable_runner"
        if any(variant.is_runnable_runner_entry for variant in self.variants):
            return "runnable_runner"
        if self.runner_target or any(variant.runner_target for variant in self.variants):
            return "runner_candidate"
        return "listed_only"

    @property
    def is_runnable_runner_entry(self) -> bool:
        """Check if the entry has integrated and runnable runner targets."""
        return self.runner_entry_kind == "runnable_runner"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelZooEntry":
        """Deserialize a ModelZooEntry from a mapping dictionary."""
        entry = _require_mapping(data, "ModelZooEntry.from_dict")

        source_data = dict(_require_mapping(entry.get("source", {}), "ModelZooEntry.source"))
        source_data.setdefault("official_repo_url", _github_url_from_sources(entry))
        source_data.setdefault("license", _license_from_entry(entry))
        for key in ("official_repo_url", "hf_repo_id", "license", "requires_auth"):
            if key in entry and key not in source_data:
                source_data[key] = entry[key]
        if "source_status" in entry and "status" not in source_data:
            source_data["status"] = _normalize_source_status(entry["source_status"])
        elif "availability" in entry and "status" not in source_data:
            source_data["status"] = _normalize_source_status(entry["availability"])
        elif "release_type" in entry and "status" not in source_data:
            source_data["status"] = _normalize_source_status(entry["release_type"])

        raw_checkpoint = entry.get("checkpoint", {})
        if isinstance(raw_checkpoint, Mapping):
            checkpoint_data = dict(raw_checkpoint)
        elif isinstance(raw_checkpoint, str) and raw_checkpoint.strip():
            checkpoint_data = {"hf_repo_id": raw_checkpoint.strip()}
        else:
            checkpoint_data = {}
        hf_repo_ids = _hf_repo_ids_from_entry(entry)
        if hf_repo_ids and "hf_repo_id" not in checkpoint_data:
            checkpoint_data["hf_repo_id"] = hf_repo_ids[0]
        for key in ("hf_repo_id", "estimated_size_gb", "requires_auth"):
            if key in entry and key not in checkpoint_data:
                checkpoint_data[key] = entry[key]
        if "repos" in checkpoint_data:
            repos = checkpoint_data.get("repos") or ()
            for repo in repos:
                if isinstance(repo, Mapping):
                    checkpoint_data.setdefault("hf_repo_id", repo.get("id") or repo.get("repo_id"))
                    checkpoint_data.setdefault("revision", repo.get("sha"))
                    checkpoint_data.setdefault("requires_auth", repo.get("gated", repo.get("private", False)))
                    break
        checkpoint_refs = _checkpoint_refs_from_entry(entry, checkpoint_data)
        if checkpoint_refs:
            first_ref = next(
                (
                    ref
                    for ref in checkpoint_refs
                    if ref.revision or ref.license or ref.private is not None or ref.gated is not None
                ),
                checkpoint_refs[0],
            )
            checkpoint_data.setdefault("hf_repo_id", first_ref.hf_repo_id)
            checkpoint_data.setdefault("revision", first_ref.revision)
            checkpoint_data.setdefault("license", first_ref.license)
            checkpoint_data.setdefault("private", first_ref.private)
            checkpoint_data.setdefault("gated", first_ref.gated)
            checkpoint_data.setdefault("requires_auth", first_ref.requires_auth)

        demo_data = dict(_require_mapping(entry.get("demo_parity", {}), "ModelZooEntry.demo_parity"))
        for key in ("demo_command", "expected_artifacts"):
            if key in entry and key not in demo_data:
                demo_data[key] = entry[key]

        runner_parity_raw = entry.get("runner_parity", {})
        runner_data = dict(_require_mapping(runner_parity_raw, "ModelZooEntry.runner_parity"))
        for key in ("runner_demo_command",):
            if key in entry and "demo_command" not in runner_data:
                runner_data["demo_command"] = entry[key]
            if key in runner_data and "demo_command" not in runner_data:
                runner_data["demo_command"] = runner_data[key]
        if "runner_expected_artifacts" in entry and "expected_artifacts" not in runner_data:
            runner_data["expected_artifacts"] = entry["runner_expected_artifacts"]

        integration = entry.get("integration") if isinstance(entry.get("integration"), Mapping) else {}

        return cls(
            model_id=str(entry.get("model_id", entry.get("id", ""))),
            name=entry.get("name"),
            aliases=_tuple_of_str(entry.get("aliases", entry.get("alias"))),
            tasks=_tuple_of_str(entry.get("tasks", entry.get("task"))),
            output_artifacts=_tuple_of_str(entry.get("output_artifacts")),
            provider=entry.get("provider"),
            source=ModelSource.from_dict(source_data),
            checkpoint=CheckpointRef.from_dict(checkpoint_data),
            checkpoint_refs=checkpoint_refs,
            variants=tuple(
                ModelVariantSpec.from_dict(item)
                for item in (
                    *tuple(entry.get("variants") or ()),
                    *tuple(entry.get("modes") or ()),
                )
            ),
            integration_status=_normalize_integration_status(entry.get("integration", entry.get("integration_status", "planned"))),
            install_profile=entry.get("install_profile"),
            runner_target=entry.get("runner_target") or integration.get("runner"),
            pipeline_target=entry.get("pipeline_target") or integration.get("pipeline_target"),
            pipeline_binding=entry.get("pipeline_binding") or integration.get("pipeline_binding"),
            runtime_profile=entry.get("runtime_profile") or integration.get("runtime_profile"),
            min_vram_gb=entry.get("min_vram_gb", entry.get("minimum_vram_gb")),
            demo_parity=DemoParitySpec.from_dict(demo_data),
            runner_parity=DemoParitySpec.from_dict(runner_data),
            notes=_entry_notes(entry),
        )


# ── Loading utilities ────────────────────────────────────────

def iter_model_zoo_payloads(payload: Any) -> list[Mapping[str, Any]]:
    """Iterate over and validate varying styles of model-zoo payload structures, returning a flat list of dicts."""
    if isinstance(payload, Mapping):
        for key in ("models", "entries", "model_zoo", "manifests"):
            value = payload.get(key)
            if isinstance(value, list):
                return [_require_mapping(item, f"{key} item") for item in value]
        return [payload]
    if isinstance(payload, list):
        return [_require_mapping(item, "model_zoo list item") for item in payload]
    raise TypeError(f"model-zoo payload must be an object or list, got {type(payload).__name__}")


def load_entries(path: str | Path) -> tuple[ModelZooEntry, ...]:
    """Load, parse, and return all ModelZooEntry definitions from a manifest file.

    Schema validation errors are re-raised with the manifest path prepended.
    """
    resolved = Path(path)
    payload = load_manifest(resolved)
    entries: list[ModelZooEntry] = []
    for item in iter_model_zoo_payloads(payload):
        try:
            entries.append(ModelZooEntry.from_dict(item))
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{resolved}: {exc}") from exc
    return tuple(entries)


def select_default_variant(
    entry: ModelZooEntry,
    *,
    allow_runner_target_fallback: bool = False,
) -> ModelVariantSpec | None:
    """Resolve the synthetic default variant for runtime and manifest surfaces.

    Returns the first runnable runner variant, or ``None`` if the entry itself
    is already an integrated runnable runner.  When
    ``allow_runner_target_fallback`` is True, also considers variants that
    merely declare a ``runner_target`` without being fully integrated.

    Args:
        entry: The :class:`ModelZooEntry` to select a variant from.
        allow_runner_target_fallback: If True, fall back to variants with
            a ``runner_target`` even if not fully integrated.
    """

    if entry.runner_target and entry.integration_status == "integrated":
        return None
    runnable = next((item for item in entry.variants if item.is_runnable_runner_entry), None)
    if runnable is not None:
        return runnable
    if allow_runner_target_fallback:
        return next((item for item in entry.variants if item.runner_target), None)
    return None


__all__ = [
    "DEMO_PARITY_STATUSES",
    "INTEGRATION_STATUSES",
    "RUNNER_ENTRY_KINDS",
    "SOURCE_STATUSES",
    "CheckpointRef",
    "DemoParitySpec",
    "JsonSerializable",
    "ModelSource",
    "ModelVariantSpec",
    "ModelZooEntry",
    "iter_model_zoo_payloads",
    "load_entries",
    "select_default_variant",
]
