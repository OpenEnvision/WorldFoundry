from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.api.json_contract import JsonContract, require_mapping, to_plain
from worldfoundry.evaluation.utils import load_manifest

SOURCE_STATUSES = frozenset({"open_source", "api", "closed", "unknown"})
OPEN_SOURCE_STATUSES = frozenset(
    {
        "stable",
        "experimental",
        "preflight_only",
        "normalizer_only",
        "gated",
        "planned",
        "in_tree_runtime",
        "in_tree_result_normalizer",
        "in_tree_artifact_scores_ready",
    }
)
INTEGRATION_STATUSES = frozenset({"integrated", "planned", "blocked"})
VERIFICATION_STATUSES = frozenset({"verified", "pending", "not_applicable", "failed", "normalizer_only"})
MATURITY_STATUSES = frozenset({"verified_runner", "contract_ready", "planned", "blocked"})
_OPEN_SOURCE_ALIASES = frozenset(
    {
        "confirmed_official_code",
        "confirmed_official_code_and_data_in_github",
        "confirmed_official_code_and_hf_data",
        "confirmed_official_code_and_hf_dataset",
        "confirmed_official_code_and_hf_datasets",
        "confirmed_official_code_and_hf_metadata",
        "confirmed_official_code_and_hf_testset_task_mismatch",
        "confirmed_official_code_but_not_benchmark_dataset",
        "confirmed_official_code_no_official_hf_dataset",
        "confirmed_official_code_and_partial_hf_data",
        "confirmed_official_code_model_and_data",
        "confirmed_public_hf",
        "confirmed_public_hf_dataset",
        "open",
        "open_source",
        "public",
        "source_available",
    }
)
_UNKNOWN_SOURCE_ALIASES = frozenset(
    {
        "blocked_project_page_only",
        "paper_only",
        "unconfirmed",
    }
)
_API_ALIASES = frozenset({"api", "commercial_api", "restricted_api"})
_CLOSED_ALIASES = frozenset({"closed", "closed_source", "proprietary"})
_PENDING_INTEGRATION_ALIASES = frozenset({"pending", "todo", "not_started", "not_applicable"})
_PENDING_VERIFICATION_ALIASES = frozenset({"pending_runner", "pending_validation", "pending_verification"})
_OFFICIAL_DATASET_SOURCE_KEYS = ("huggingface_dataset", "huggingface_datasets", "hf_datasets", "datasets")

JsonValue = Any
JsonSerializable = JsonContract
_to_plain = to_plain
_require_mapping = require_mapping


def _optional_str(value: Any) -> str | None:
    """Coerces the value to a string or returns None if it is None.

    Args:
        value: Input value of any type or None.

    Returns:
        The coerced string or None.
    """
    if value is None:
        return None
    return str(value)


def _optional_command(value: Any) -> str | tuple[str, ...] | None:
    """Coerces command inputs into a single string command, a tuple of argument strings, or None.

    Args:
        value: Command representation (str, list, tuple, or None).

    Returns:
        The normalized command, or None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return str(value)


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    """Coerces any input into a tuple of strings.

    Args:
        value: Input sequence, single string, or None.

    Returns:
        A tuple of string items.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _tuple_of_artifacts(value: Any) -> tuple[JsonValue, ...]:
    """Coerces artifact inputs into a tuple of normalized JsonValue items.

    Args:
        value: Raw artifact description.

    Returns:
        A tuple of normalized JsonValue artifacts.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return (_to_plain(value),)
    if isinstance(value, (list, tuple)):
        return tuple(_to_plain(item) for item in value)
    return (str(value),)


def _tuple_of_command_specs(value: Any) -> tuple[JsonValue, ...]:
    """Coerces command spec specifications into a standardized command spec tuple.

    Args:
        value: Command spec object or sequence.

    Returns:
        A tuple of standardized command spec items.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return (_to_plain(value),)
    if isinstance(value, (list, tuple)):
        if not value:
            return ()
        if all(isinstance(item, str) for item in value):
            return (tuple(str(item) for item in value),)
        return tuple(_to_plain(item) for item in value)
    return (str(value),)


def _bool(value: Any) -> bool:
    """Safely casts any value to boolean.

    Args:
        value: Value to cast.

    Returns:
        The boolean value.
    """
    return bool(value)


def _metric_spec(value: Any) -> "BenchmarkMetricSpec":
    """Coerces a raw input into a BenchmarkMetricSpec object.

    Args:
        value: A metric ID string, raw mapping, or existing spec.

    Returns:
        A BenchmarkMetricSpec object.
    """
    if isinstance(value, BenchmarkMetricSpec):
        return value
    if isinstance(value, str):
        return BenchmarkMetricSpec(metric_id=value)
    return BenchmarkMetricSpec.from_dict(_require_mapping(value, "BenchmarkMetricSpec item"))


def _validate_status(value: str, allowed: frozenset[str], context: str) -> str:
    """Validates that a status string is within the allowed set.

    Args:
        value: The status string to test.
        allowed: The allowed set of status values.
        context: Context string used in compiling the exception message.

    Returns:
        The validated status string.

    Raises:
        ValueError: If value is not allowed.
    """
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{context} must be one of: {allowed_values}. Got {value!r}.")
    return value


def _first_text(*values: Any) -> str | None:
    """Returns the first non-empty text string from a list of arguments.

    Args:
        *values: Arguments to check.

    Returns:
        The first non-empty string, or None.
    """
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _default_ready_now_command(benchmark_id: str) -> str:
    """Generates the default shell command for performing a full ready now run.

    Args:
        benchmark_id: The canonical benchmark ID.

    Returns:
        The execution command string.
    """
    return (
        f"worldfoundry-eval run --benchmark {benchmark_id} "
        f"--output-dir tmp/benchmark_eval/{benchmark_id} --json"
    )


def _normalize_source_status(value: Any) -> str:
    """Normalizes various source status representations or aliases into a canonical status code.

    Args:
        value: Input source status descriptor.

    Returns:
        The canonical source status code.
    """
    if isinstance(value, Mapping):
        value = (
            value.get("status")
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
    return "unknown"


def _normalize_integration_status(value: Any) -> str:
    """Normalizes integration status value or mapping into a canonical integration status string.

    Args:
        value: Input integration status.

    Returns:
        The canonical integration status.
    """
    if isinstance(value, Mapping):
        value = value.get("status", "planned")
    normalized = str(value or "planned").strip().lower()
    if normalized in INTEGRATION_STATUSES:
        return normalized
    if normalized.startswith("blocked"):
        return "blocked"
    if normalized in _PENDING_INTEGRATION_ALIASES or normalized.startswith("pending"):
        return "planned"
    return "planned"


def _normalize_verification_status(value: Any) -> str:
    """Normalizes verification status value or mapping into a canonical verification status string.

    Args:
        value: Input verification status.

    Returns:
        The canonical verification status.
    """
    if isinstance(value, Mapping):
        value = value.get("status", "pending")
    normalized = str(value or "pending").strip().lower()
    if normalized == "normalizer":
        return "normalizer_only"
    if normalized in VERIFICATION_STATUSES:
        return normalized
    if normalized in _PENDING_VERIFICATION_ALIASES or normalized.startswith("pending"):
        return "pending"
    return "pending"


def _normalize_open_source_status(value: Any) -> str:
    """Normalizes open source status value or mapping into a canonical open source status string.

    Args:
        value: Input open source status.

    Returns:
        The canonical open source status.
    """
    if isinstance(value, Mapping):
        value = value.get("status", "planned")
    normalized = str(value or "planned").strip().lower().replace("-", "_")
    if normalized in OPEN_SOURCE_STATUSES:
        return normalized
    if normalized == "preflight":
        return "preflight_only"
    if normalized == "normalizer":
        return "normalizer_only"
    return "planned"


def _normalize_maturity_status(value: Any) -> str:
    """Normalizes maturity status value or mapping into a canonical maturity status string.

    Args:
        value: Input maturity status.

    Returns:
        The canonical maturity status.
    """
    normalized = str(value or "planned").strip().lower().replace("-", "_")
    if normalized in MATURITY_STATUSES:
        return normalized
    if normalized in {"integrated", "verified", "validated"}:
        return "verified_runner"
    if normalized in {"preflight_only", "normalizer_only", "contract", "contract_only"}:
        return "contract_ready"
    if normalized.startswith("blocked"):
        return "blocked"
    return "planned"


def _github_url_from_sources(entry: Mapping[str, Any]) -> str | None:
    """Extracts a GitHub repository URL from various catalog entry source paths.

    Args:
        entry: Input catalog entry mapping.

    Returns:
        The resolved GitHub URL string, or None if not found.
    """
    direct = _first_text(entry.get("official_repo_url"))
    if direct:
        return direct

    source = entry.get("source")
    if isinstance(source, Mapping):
        direct = _first_text(source.get("official_repo_url"))
        if direct:
            return direct

    official_sources = entry.get("official_sources")
    if isinstance(official_sources, Mapping):
        github = official_sources.get("github")
        items = github if isinstance(github, (list, tuple)) else (github,)
        for item in items:
            if isinstance(item, Mapping):
                direct = _first_text(item.get("url"))
                if direct:
                    return direct
            else:
                direct = _first_text(item)
                if direct:
                    return direct
    return None


def _paper_or_project_url_from_sources(entry: Mapping[str, Any]) -> str | None:
    """Extracts a paper or project page URL from various catalog entry source paths.

    Args:
        entry: Input catalog entry mapping.

    Returns:
        The resolved paper or project URL string, or None if not found.
    """
    direct = _first_text(entry.get("paper_url"), entry.get("project_page"))
    if direct:
        return direct
    source = entry.get("source")
    if isinstance(source, Mapping):
        direct = _first_text(source.get("paper_url"), source.get("project_page"))
        if direct:
            return direct
    official_sources = entry.get("official_sources")
    if isinstance(official_sources, Mapping):
        return _first_text(official_sources.get("paper_url"), official_sources.get("project_page"))
    return None


def _hf_dataset_id_from_url(value: Any) -> str | None:
    """Parses and extracts a Hugging Face dataset repository identifier from a URL.

    Args:
        value: Input URL string or object.

    Returns:
        The extracted HF dataset ID (e.g. 'owner/repo'), or None if invalid.
    """
    if not isinstance(value, str):
        return None
    marker = "huggingface.co/datasets/"
    if marker not in value:
        return None
    suffix = value.split(marker, 1)[1].strip("/")
    parts = suffix.split("/")
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])


def _mapping_has_direct_dataset_ref_shape(data: Mapping[str, Any]) -> bool:
    """Detects whether a dictionary mapping represents a direct dataset reference format.

    Args:
        data: The dictionary mapping to inspect.

    Returns:
        True if the mapping contains keys typical of a dataset reference, False otherwise.
    """
    return any(
        key in data
        for key in (
            "hf_dataset_id",
            "repo_id",
            "id",
            "url",
            "not_applicable",
        )
    )


def _official_dataset_source_items(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    """Retrieves all official dataset sources listed in an entry mapping.

    Args:
        entry: Input catalog entry mapping.

    Returns:
        A tuple of raw official dataset source specifications.
    """
    official_sources = entry.get("official_sources")
    if not isinstance(official_sources, Mapping):
        return ()
    items: list[Any] = []
    for key in _OFFICIAL_DATASET_SOURCE_KEYS:
        values = official_sources.get(key)
        if isinstance(values, (list, tuple)):
            items.extend(values)
        elif values is not None:
            items.append(values)
    return tuple(items)


def _hf_dataset_ids_from_entry(entry: Mapping[str, Any]) -> list[str]:
    """Aggregates all unique Hugging Face dataset IDs referenced across various parts of a catalog entry.

    Args:
        entry: Input catalog entry mapping.

    Returns:
        A list of parsed HF dataset ID strings.
    """
    dataset_ids: list[str] = []

    def add(value: Any) -> None:
        dataset_id = _hf_dataset_id_from_url(value) or _first_text(value)
        if dataset_id and dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)

    add(entry.get("hf_dataset_id"))
    dataset = entry.get("dataset")
    if isinstance(dataset, Mapping):
        add(dataset.get("hf_dataset_id"))

    for item in _official_dataset_source_items(entry):
        if isinstance(item, Mapping):
            add(item.get("repo_id") or item.get("id") or item.get("url"))
        else:
            add(item)
    return dataset_ids


def _requires_auth_from_dataset_sources(entry: Mapping[str, Any]) -> bool:
    """Determines if any of the entry's dataset sources require authentication.

    Args:
        entry: Input catalog entry mapping.

    Returns:
        True if authentication is required by any dataset source, False otherwise.
    """
    if bool(entry.get("requires_auth", False)):
        return True
    dataset = entry.get("dataset")
    if isinstance(dataset, Mapping) and bool(dataset.get("requires_auth", False)):
        return True
    for item in _official_dataset_source_items(entry):
        if not isinstance(item, Mapping):
            continue
        if item.get("private") is True:
            return True
        gated = item.get("gated")
        if gated not in (None, False, "false", "False", "none", "None"):
            return True
    return False


def _entry_notes(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """Gathers and aggregates notes and blocker explanations for an entry mapping.

    Args:
        entry: Input catalog entry mapping.

    Returns:
        A tuple of explanatory note strings.
    """
    notes = list(_tuple_of_str(entry.get("notes")))
    integration = entry.get("integration")
    if isinstance(integration, Mapping):
        for reason in _tuple_of_str(integration.get("blocked_reasons")):
            notes.append(f"blocked: {reason}")
    return tuple(notes)


def _entry_requires(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """Compiles all environmental requirement requirements specified for an entry.

    Args:
        entry: Input catalog entry mapping.

    Returns:
        A tuple of environmental requirement strings.
    """
    requires = list(_tuple_of_str(entry.get("requires")))
    runner = entry.get("runner")
    if isinstance(runner, Mapping):
        requires.extend(_tuple_of_str(runner.get("required_env")))
    return tuple(dict.fromkeys(requires))


def _entry_blockers(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """Compiles all blocker descriptions declared in an entry's manifest or integration section.

    Args:
        entry: Input catalog entry mapping.

    Returns:
        A tuple of unique blocker descriptor strings.
    """
    blockers = list(_tuple_of_str(entry.get("blockers")))
    integration = entry.get("integration")
    if isinstance(integration, Mapping):
        blockers.extend(_tuple_of_str(integration.get("blocked_reasons")))
    return tuple(dict.fromkeys(blockers))


def _entry_bool_flag(entry: Mapping[str, Any], key: str) -> bool:
    """Extract an entry-level evidence flag with legacy nested fallback.

    Args:
        entry: Input catalog entry mapping.
        key: The flag key name to search.

    Returns:
        The extracted boolean flag value.
    """
    # Entry-level flags are the public claim boundary.  Older manifests stored
    # these values under integration.evidence, so retain that form only when a
    # top-level declaration is absent.  In particular, bounded component
    # evidence must never override an explicit full-benchmark ``false``.
    if key in entry:
        return _bool(entry[key])
    integration = entry.get("integration")
    if isinstance(integration, Mapping):
        evidence = integration.get("evidence")
        if isinstance(evidence, Mapping) and key in evidence:
            return _bool(evidence[key])
    return _bool(entry.get(key, False))


@dataclass(frozen=True)
class BenchmarkSource(JsonSerializable):
    status: str = "unknown"
    official_repo_url: str | None = None
    paper_url: str | None = None
    requires_auth: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _validate_status(str(self.status), SOURCE_STATUSES, "BenchmarkSource.status"))
        object.__setattr__(self, "official_repo_url", _optional_str(self.official_repo_url))
        object.__setattr__(self, "paper_url", _optional_str(self.paper_url))
        object.__setattr__(self, "requires_auth", _bool(self.requires_auth))
        object.__setattr__(self, "notes", _tuple_of_str(self.notes))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkSource":
        source = _require_mapping(data, "BenchmarkSource.from_dict")
        status = _normalize_source_status(source.get("status", source.get("source_status", "unknown")))
        official_repo_url = source.get("official_repo_url")
        if status == "unknown" and official_repo_url:
            status = "open_source"
        return cls(
            status=status,
            official_repo_url=official_repo_url,
            paper_url=source.get("paper_url"),
            requires_auth=source.get("requires_auth", False),
            notes=_tuple_of_str(source.get("notes")),
        )


@dataclass(frozen=True)
class BenchmarkDatasetRef(JsonSerializable):
    hf_dataset_id: str | None = None
    revision: str | None = None
    license: str | None = None
    private: bool | None = None
    gated: JsonValue | None = None
    split: str | None = None
    path: str | None = None
    not_applicable: bool = False
    reason: str | None = None
    requires_auth: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hf_dataset_id", _optional_str(self.hf_dataset_id))
        object.__setattr__(self, "revision", _optional_str(self.revision))
        object.__setattr__(self, "license", _optional_str(self.license))
        if self.private is not None:
            object.__setattr__(self, "private", _bool(self.private))
        object.__setattr__(self, "not_applicable", _bool(self.not_applicable))
        object.__setattr__(self, "reason", _optional_str(self.reason))
        object.__setattr__(self, "split", _optional_str(self.split))
        object.__setattr__(self, "path", _optional_str(self.path))
        gated_requires_auth = self.gated not in (None, False, "false", "False", "none", "None")
        object.__setattr__(
            self,
            "requires_auth",
            _bool(self.requires_auth) or self.private is True or gated_requires_auth,
        )
        object.__setattr__(self, "notes", _tuple_of_str(self.notes))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkDatasetRef":
        """Instantiates a BenchmarkDatasetRef from a raw dictionary mapping.

        Args:
            data: Input dictionary configuration.

        Returns:
            A BenchmarkDatasetRef instance.
        """
        dataset = _require_mapping(data, "BenchmarkDatasetRef.from_dict")
        return cls(
            hf_dataset_id=dataset.get("hf_dataset_id") or dataset.get("repo_id") or dataset.get("id"),
            revision=dataset.get("revision", dataset.get("sha", dataset.get("commit"))),
            license=dataset.get("license"),
            private=dataset.get("private"),
            gated=dataset.get("gated"),
            split=dataset.get("split"),
            path=dataset.get("path"),
            not_applicable=dataset.get("not_applicable", False),
            reason=dataset.get("reason"),
            requires_auth=dataset.get("requires_auth", False),
            notes=_tuple_of_str(dataset.get("notes")),
        )


def _dedupe_dataset_refs(refs: list[BenchmarkDatasetRef]) -> tuple[BenchmarkDatasetRef, ...]:
    """Dedupes dataset references by checking combination of ID, revision, split and path.

    Args:
        refs: List of BenchmarkDatasetRef to deduplicate.

    Returns:
        A tuple of unique BenchmarkDatasetRef objects.
    """
    deduped: list[BenchmarkDatasetRef] = []
    seen: set[tuple[str | None, str | None, str | None, str | None]] = set()
    for ref in refs:
        key = (ref.hf_dataset_id, ref.revision, ref.split, ref.path)
        if not ref.hf_dataset_id or key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


def _dataset_refs_from_entry(entry: Mapping[str, Any], dataset_data: Mapping[str, Any]) -> tuple[BenchmarkDatasetRef, ...]:
    """Extracts and compiles all unique dataset references declared within a catalog entry mapping.

    Args:
        entry: Input catalog entry mapping.
        dataset_data: The fallback dataset data mapping.

    Returns:
        A tuple of unique BenchmarkDatasetRef objects.
    """
    refs: list[BenchmarkDatasetRef] = []

    if "dataset_refs" in entry:
        for item in entry.get("dataset_refs") or ():
            if isinstance(item, Mapping):
                refs.append(BenchmarkDatasetRef.from_dict(item))
        return _dedupe_dataset_refs(refs)

    data_refs = entry.get("data_refs")
    if isinstance(data_refs, Mapping):
        for item in data_refs.get("dataset_refs") or ():
            if isinstance(item, Mapping):
                refs.append(BenchmarkDatasetRef.from_dict(item))
        if _mapping_has_direct_dataset_ref_shape(data_refs):
            refs.append(BenchmarkDatasetRef.from_dict(data_refs))

    has_official_dataset_sources = False
    official_sources = entry.get("official_sources")
    if isinstance(official_sources, Mapping):
        has_official_dataset_sources = any(
            official_sources.get(key) is not None
            for key in _OFFICIAL_DATASET_SOURCE_KEYS
        )

    if dataset_data.get("hf_dataset_id") and not has_official_dataset_sources:
        refs.append(BenchmarkDatasetRef.from_dict(dataset_data))

    for item in _official_dataset_source_items(entry):
        if isinstance(item, Mapping):
            refs.append(BenchmarkDatasetRef.from_dict(item))
        else:
            refs.append(BenchmarkDatasetRef(hf_dataset_id=_hf_dataset_id_from_url(item) or str(item)))

    if entry.get("hf_dataset_id"):
        refs.append(BenchmarkDatasetRef(hf_dataset_id=str(entry["hf_dataset_id"])))

    return _dedupe_dataset_refs(refs)


@dataclass(frozen=True)
class BenchmarkMetricSpec(JsonSerializable):
    """Specification of a specific evaluation metric produced by an external benchmark.

    Attributes:
        metric_id: Canonical identifier of this metric.
        name: Optional human-readable display name.
        description: Optional textual description.
        higher_is_better: Whether higher values of this metric represent better performance.
        raw_metric_name: Name used in the upstream benchmark's raw output.
        leaderboard_key: Key under which this metric is published on the leaderboard.
        normalizer: Canonical name of normalizer function to scale values.
        aggregator: Accumulation strategy (e.g. 'mean', 'sum').
        output_unit: Output measurement unit symbol/name.
        primary: If True, indicates a primary/core metric.
        weight: Coefficient/weight applied in compound score calculations.
        official_results: Optional baseline results of reference models.
        notes: Explanatory annotations.
    """
    metric_id: str
    name: str | None = None
    description: str | None = None
    higher_is_better: bool | None = None
    raw_metric_name: str | None = None
    leaderboard_key: str | None = None
    normalizer: str | None = None
    aggregator: str = "mean"
    output_unit: str | None = None
    primary: bool = False
    weight: float = 1.0
    official_results: Mapping[str, JsonValue] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.metric_id:
            raise ValueError("BenchmarkMetricSpec.metric_id is required.")
        object.__setattr__(self, "metric_id", str(self.metric_id))
        object.__setattr__(self, "name", _optional_str(self.name))
        object.__setattr__(self, "description", _optional_str(self.description))
        if self.higher_is_better is not None:
            object.__setattr__(self, "higher_is_better", _bool(self.higher_is_better))
        object.__setattr__(self, "raw_metric_name", _optional_str(self.raw_metric_name))
        object.__setattr__(self, "leaderboard_key", _optional_str(self.leaderboard_key))
        object.__setattr__(self, "normalizer", _optional_str(self.normalizer))
        object.__setattr__(self, "aggregator", str(self.aggregator or "mean"))
        object.__setattr__(self, "output_unit", _optional_str(self.output_unit))
        object.__setattr__(self, "primary", _bool(self.primary))
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(
            self,
            "official_results",
            _to_plain(self.official_results) if isinstance(self.official_results, Mapping) else {},
        )
        object.__setattr__(self, "notes", _tuple_of_str(self.notes))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkMetricSpec":
        """Loads a BenchmarkMetricSpec from a raw dictionary mapping.

        Args:
            data: Input dictionary mapping.

        Returns:
            A BenchmarkMetricSpec instance.
        """
        metric = _require_mapping(data, "BenchmarkMetricSpec.from_dict")
        return cls(
            metric_id=str(metric.get("metric_id", metric.get("id", ""))),
            name=metric.get("name"),
            description=metric.get("description"),
            higher_is_better=metric.get("higher_is_better"),
            raw_metric_name=metric.get("raw_metric_name", metric.get("raw_name")),
            leaderboard_key=metric.get("leaderboard_key", metric.get("leaderboard")),
            normalizer=metric.get("normalizer"),
            aggregator=metric.get("aggregator", "mean"),
            output_unit=metric.get("output_unit", metric.get("unit")),
            primary=metric.get("primary", False),
            weight=metric.get("weight", 1.0),
            official_results=_require_mapping(
                metric.get("official_results", metric.get("official_result", {})),
                "BenchmarkMetricSpec.official_results",
            ),
            notes=_tuple_of_str(metric.get("notes")),
        )


@dataclass(frozen=True)
class BenchmarkRunnerSpec(JsonSerializable):
    """Specification of execution runtime environment and dependency configurations for benchmark execution.

    Attributes:
        install_profile: Named installation environment profile template.
        runner_target: Executable script or package entry point.
        run_command: Default command line to execute benchmark runs.
        validation_command: Command to validate output metrics.
        repo_url: URL to the official benchmark codebase repository.
        repo_revision: Checkout git revision/commit/tag.
        clone_dir: Destination path inside virtual environment.
        install_commands: Ordered instructions for workspace initialization and setup.
        env: Process level environment variable assignments.
        assets: Required data files/checkpoints mapped to fetch locations.
        dependency_profile: System libraries dependency list.
        runtime: Nested engine specific runtime properties.
        expected_artifacts: Output artifacts path specifications.
        verification_status: Local validation and deployment status code.
        notes: Explanatory annotations.
    """
    install_profile: str | None = None
    runner_target: str | None = None
    run_command: str | tuple[str, ...] | None = None
    validation_command: str | tuple[str, ...] | None = None
    repo_url: str | None = None
    repo_revision: str | None = None
    clone_dir: str | None = None
    install_commands: tuple[JsonValue, ...] = ()
    env: Mapping[str, JsonValue] = field(default_factory=dict)
    assets: Mapping[str, JsonValue] = field(default_factory=dict)
    dependency_profile: str | None = None
    runtime: Mapping[str, JsonValue] = field(default_factory=dict)
    expected_artifacts: tuple[JsonValue, ...] = ()
    verification_status: str = "pending"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "install_profile", _optional_str(self.install_profile))
        object.__setattr__(self, "runner_target", _optional_str(self.runner_target))
        object.__setattr__(self, "run_command", _optional_command(self.run_command))
        object.__setattr__(self, "validation_command", _optional_command(self.validation_command))
        object.__setattr__(self, "repo_url", _optional_str(self.repo_url))
        object.__setattr__(self, "repo_revision", _optional_str(self.repo_revision))
        object.__setattr__(self, "clone_dir", _optional_str(self.clone_dir))
        object.__setattr__(self, "install_commands", _tuple_of_command_specs(self.install_commands))
        object.__setattr__(self, "env", _to_plain(self.env) if isinstance(self.env, Mapping) else {})
        object.__setattr__(self, "assets", _to_plain(self.assets) if isinstance(self.assets, Mapping) else {})
        object.__setattr__(self, "dependency_profile", _optional_str(self.dependency_profile))
        object.__setattr__(self, "runtime", _to_plain(self.runtime) if isinstance(self.runtime, Mapping) else {})
        object.__setattr__(self, "expected_artifacts", _tuple_of_artifacts(self.expected_artifacts))
        object.__setattr__(
            self,
            "verification_status",
            _validate_status(str(self.verification_status), VERIFICATION_STATUSES, "BenchmarkRunnerSpec.verification_status"),
        )
        object.__setattr__(self, "notes", _tuple_of_str(self.notes))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkRunnerSpec":
        """Constructs a BenchmarkRunnerSpec from a raw dictionary mapping.

        Args:
            data: Input dictionary configuration mapping.

        Returns:
            A BenchmarkRunnerSpec instance.
        """
        runner = _require_mapping(data, "BenchmarkRunnerSpec.from_dict")
        runtime = _require_mapping(runner.get("runtime", {}), "BenchmarkRunnerSpec.runtime")
        notes = list(_tuple_of_str(runner.get("notes")))
        for reason in _tuple_of_str(runner.get("blocked_reasons")):
            notes.append(f"blocked: {reason}")
        install_commands = runner.get("install_commands", runtime.get("install_commands"))
        if install_commands is None and runtime.get("setup_command") is not None:
            install_commands = [runtime["setup_command"]]
        return cls(
            install_profile=runner.get("install_profile"),
            runner_target=runner.get("runner_target"),
            run_command=runner.get("run_command"),
            validation_command=runner.get("validation_command"),
            repo_url=runner.get("repo_url", runtime.get("repo_url", runtime.get("official_repo_url"))),
            repo_revision=runner.get("repo_revision", runtime.get("repo_revision", runtime.get("revision"))),
            clone_dir=runner.get("clone_dir", runtime.get("clone_dir", runtime.get("root"))),
            install_commands=install_commands,
            env=_require_mapping(runner.get("env", runtime.get("env", {})), "BenchmarkRunnerSpec.env"),
            assets=_require_mapping(runner.get("assets", runtime.get("assets", {})), "BenchmarkRunnerSpec.assets"),
            dependency_profile=runner.get("dependency_profile", runtime.get("dependency_profile")),
            runtime=runtime,
            expected_artifacts=_tuple_of_artifacts(runner.get("expected_artifacts")),
            verification_status=_normalize_verification_status(
                runner.get("verification_status", runner.get("status", "pending"))
            ),
            notes=tuple(notes),
        )


@dataclass(frozen=True)
class BenchmarkZooEntry(JsonSerializable):
    """A strongly-typed, validated container mapping an auto-discovered benchmark entry in benchmark_zoo.

    Attributes:
        benchmark_id: Canonical unique name of the benchmark.
        name: Human-readable name.
        aliases: Registered lookup aliases.
        contract_validation_command: Deprecated legacy fixture command retained for old manifests.
        ready_now_command: One-click runner execution instruction.
        one_click_command: Copy of ready_now_command kept for backwards compatibility.
        domains: Task categories (e.g. 'video_generation', 'robotics').
        modalities: Media inputs/outputs formats (e.g. 'video', 'text').
        tags: Categorization tags.
        source: BenchmarkSource code repository configuration.
        dataset: Main dataset reference object.
        dataset_refs: Sequence of auxiliary dataset reference objects.
        integration_status: Status code representing framework readiness.
        runner: Detailed environment execution details spec.
        metrics: Complete sequence of compiled metrics.
        open_source_status: Detailed release status of the upstream codebase.
        release_status: Synonym of open_source_status.
        maturity: Execution maturity categorization label.
        official_benchmark_verified: True if official test outputs are validated locally.
        integration_evidence: True if successful local run report has been created.
        leaderboard_valid: True if ready to be listed in scorecards.
        base_model_dependencies: Sequence of prerequisite large foundation models.
        optional_base_model_dependencies: Non-critical prereq models.
        requires: Environmental/Hardware dependency keywords.
        blockers: Explanations of outstanding engineering barriers.
        data_refs: Miscellaneous dataset reference properties.
        runner_availability: Dynamic verification evidence markers.
        notes: General developer annotations.
    """
    benchmark_id: str
    name: str | None = None
    aliases: tuple[str, ...] = ()
    contract_validation_command: str | None = None
    ready_now_command: str | None = None
    one_click_command: str | None = None
    domains: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    source: BenchmarkSource = field(default_factory=BenchmarkSource)
    dataset: BenchmarkDatasetRef = field(default_factory=BenchmarkDatasetRef)
    dataset_refs: tuple[BenchmarkDatasetRef, ...] = ()
    integration_status: str = "planned"
    runner: BenchmarkRunnerSpec = field(default_factory=BenchmarkRunnerSpec)
    metrics: tuple[BenchmarkMetricSpec, ...] = ()
    open_source_status: str = "planned"
    release_status: str = "planned"
    maturity: str = "planned"
    official_benchmark_verified: bool = False
    integration_evidence: bool = False
    leaderboard_valid: bool = False
    base_model_dependencies: tuple[str, ...] = ()
    optional_base_model_dependencies: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    data_refs: Mapping[str, JsonValue] = field(default_factory=dict)
    runner_availability: Mapping[str, JsonValue] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("BenchmarkZooEntry.benchmark_id is required.")
        object.__setattr__(self, "benchmark_id", str(self.benchmark_id))
        object.__setattr__(self, "name", _optional_str(self.name))
        object.__setattr__(self, "aliases", _tuple_of_str(self.aliases))
        object.__setattr__(
            self,
            "contract_validation_command",
            _optional_str(self.contract_validation_command),
        )
        object.__setattr__(self, "ready_now_command", _optional_str(self.ready_now_command))
        object.__setattr__(
            self,
            "one_click_command",
            _optional_str(self.one_click_command),
        )
        object.__setattr__(self, "domains", _tuple_of_str(self.domains))
        object.__setattr__(self, "modalities", _tuple_of_str(self.modalities))
        object.__setattr__(self, "tags", _tuple_of_str(self.tags))
        object.__setattr__(
            self,
            "dataset_refs",
            tuple(
                item if isinstance(item, BenchmarkDatasetRef) else BenchmarkDatasetRef.from_dict(item)
                for item in self.dataset_refs
            ),
        )
        object.__setattr__(
            self,
            "integration_status",
            _validate_status(str(self.integration_status), INTEGRATION_STATUSES, "BenchmarkZooEntry.integration_status"),
        )
        object.__setattr__(self, "metrics", tuple(_metric_spec(item) for item in self.metrics))
        object.__setattr__(
            self,
            "open_source_status",
            _validate_status(
                str(self.open_source_status),
                OPEN_SOURCE_STATUSES,
                "BenchmarkZooEntry.open_source_status",
            ),
        )
        object.__setattr__(
            self,
            "release_status",
            _validate_status(
                str(self.release_status),
                OPEN_SOURCE_STATUSES,
                "BenchmarkZooEntry.release_status",
            ),
        )
        object.__setattr__(
            self,
            "maturity",
            _validate_status(str(self.maturity), MATURITY_STATUSES, "BenchmarkZooEntry.maturity"),
        )
        object.__setattr__(self, "official_benchmark_verified", _bool(self.official_benchmark_verified))
        object.__setattr__(self, "integration_evidence", _bool(self.integration_evidence))
        object.__setattr__(self, "leaderboard_valid", _bool(self.leaderboard_valid))
        object.__setattr__(self, "base_model_dependencies", _tuple_of_str(self.base_model_dependencies))
        object.__setattr__(
            self,
            "optional_base_model_dependencies",
            _tuple_of_str(self.optional_base_model_dependencies),
        )
        object.__setattr__(self, "requires", _tuple_of_str(self.requires))
        object.__setattr__(self, "blockers", _tuple_of_str(self.blockers))
        object.__setattr__(self, "data_refs", dict(_require_mapping(self.data_refs, "BenchmarkZooEntry.data_refs")))
        object.__setattr__(
            self,
            "runner_availability",
            dict(_require_mapping(self.runner_availability, "BenchmarkZooEntry.runner_availability")),
        )
        object.__setattr__(self, "notes", _tuple_of_str(self.notes))
        official_ready = (
            self.integration_status == "integrated"
            and self.runner.verification_status == "verified"
            and self.runner.runner_target is not None
            and self.leaderboard_valid
        )
        if official_ready and self.ready_now_command is None:
            object.__setattr__(self, "ready_now_command", _default_ready_now_command(str(self.benchmark_id)))
        if official_ready and self.one_click_command is None:
            object.__setattr__(self, "one_click_command", self.ready_now_command)

    @property
    def source_status(self) -> str:
        """Returns the canonical status of the benchmark's upstream source code."""
        return self.source.status

    @property
    def official_repo_url(self) -> str | None:
        """Returns the official code repository URL if defined."""
        return self.source.official_repo_url

    @property
    def paper_url(self) -> str | None:
        """Returns the publication paper or project page URL if defined."""
        return self.source.paper_url

    @property
    def hf_dataset_id(self) -> str | None:
        """Returns the canonical Hugging Face dataset identifier if defined."""
        if self.dataset.hf_dataset_id:
            return self.dataset.hf_dataset_id
        if self.dataset_refs:
            return self.dataset_refs[0].hf_dataset_id
        return None

    @property
    def hf_dataset_ids(self) -> tuple[str, ...]:
        """Returns all referenced Hugging Face dataset identifiers."""
        ids: list[str] = []
        for ref in (*self.dataset_refs, self.dataset):
            if ref.hf_dataset_id and ref.hf_dataset_id not in ids:
                ids.append(ref.hf_dataset_id)
        return tuple(ids)

    @property
    def requires_auth(self) -> bool:
        """Returns whether access to the codebase or datasets requires credentials."""
        return self.source.requires_auth or self.dataset.requires_auth or any(ref.requires_auth for ref in self.dataset_refs)

    @property
    def install_profile(self) -> str | None:
        """Returns the name of target virtual environment deployment profile."""
        return self.runner.install_profile

    @property
    def runner_target(self) -> str | None:
        """Returns the execution entry point / package runner target."""
        return self.runner.runner_target

    @property
    def run_command(self) -> str | tuple[str, ...] | None:
        """Returns default execution command line."""
        return self.runner.run_command

    @property
    def validation_command(self) -> str | tuple[str, ...] | None:
        """Returns default validation command line."""
        return self.runner.validation_command

    @property
    def expected_artifacts(self) -> tuple[JsonValue, ...]:
        """Returns a list of expected output files / artifact configurations."""
        return self.runner.expected_artifacts

    @property
    def runner_runtime(self) -> Mapping[str, JsonValue]:
        """Returns nested engine runtime properties."""
        return self.runner.runtime

    @property
    def verification_status(self) -> str:
        """Returns local deployment/verification status code."""
        return self.runner.verification_status

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkZooEntry":
        """Loads a BenchmarkZooEntry from a raw dictionary mapping, normalizing and parsing nested fields.

        Args:
            data: Input configuration dictionary.

        Returns:
            A fully constructed and validated BenchmarkZooEntry.
        """
        entry = _require_mapping(data, "BenchmarkZooEntry.from_dict")

        source_data = dict(_require_mapping(entry.get("source", {}), "BenchmarkZooEntry.source"))
        source_data.setdefault("official_repo_url", _github_url_from_sources(entry))
        source_data.setdefault("paper_url", _paper_or_project_url_from_sources(entry))
        for key in ("official_repo_url", "paper_url", "requires_auth"):
            if key in entry and key not in source_data:
                source_data[key] = entry[key]
        if "source_status" in entry and "status" not in source_data:
            source_data["status"] = entry["source_status"]
        elif "status" in entry and "status" not in source_data:
            source_data["status"] = entry["status"]

        dataset_data = dict(_require_mapping(entry.get("dataset", {}), "BenchmarkZooEntry.dataset"))
        hf_dataset_ids = _hf_dataset_ids_from_entry(entry)
        if hf_dataset_ids and "hf_dataset_id" not in dataset_data:
            dataset_data["hf_dataset_id"] = hf_dataset_ids[0]
        if _requires_auth_from_dataset_sources(entry) and "requires_auth" not in dataset_data:
            dataset_data["requires_auth"] = True
        for key in ("hf_dataset_id", "revision", "split", "path", "requires_auth"):
            if key in entry and key not in dataset_data:
                dataset_data[key] = entry[key]
        dataset_refs = _dataset_refs_from_entry(entry, dataset_data)
        if dataset_refs and "hf_dataset_id" not in dataset_data:
            first_ref = dataset_refs[0]
            dataset_data.setdefault("hf_dataset_id", first_ref.hf_dataset_id)
            dataset_data.setdefault("revision", first_ref.revision)
            dataset_data.setdefault("license", first_ref.license)
            dataset_data.setdefault("private", first_ref.private)
            dataset_data.setdefault("gated", first_ref.gated)
            dataset_data.setdefault("requires_auth", first_ref.requires_auth)

        runner_data = dict(_require_mapping(entry.get("runner", {}), "BenchmarkZooEntry.runner"))
        for key in ("install_profile", "runner_target", "run_command", "validation_command", "runtime", "expected_artifacts"):
            if key in entry and key not in runner_data:
                runner_data[key] = entry[key]
        if "verification_status" in entry and "verification_status" not in runner_data:
            runner_data["verification_status"] = entry["verification_status"]
        integration = entry.get("integration")
        if isinstance(integration, Mapping):
            for key in ("run_command", "validation_command", "runtime", "expected_artifacts"):
                if key in integration and key not in runner_data:
                    runner_data[key] = integration[key]

        metrics = tuple(BenchmarkMetricSpec.from_dict(item) for item in entry.get("metrics", ()))

        release_status = _normalize_open_source_status(
            entry.get("release_status", entry.get("open_source_status", "planned"))
        )

        return cls(
            benchmark_id=str(entry.get("benchmark_id", entry.get("id", ""))),
            name=entry.get("name"),
            aliases=tuple(dict.fromkeys((*_tuple_of_str(entry.get("aliases")), *_tuple_of_str(entry.get("alias"))))),
            contract_validation_command=entry.get("contract_validation_command"),
            ready_now_command=entry.get("ready_now_command"),
            one_click_command=entry.get("one_click_command"),
            domains=_tuple_of_str(entry.get("domains", entry.get("domain"))),
            modalities=_tuple_of_str(entry.get("modalities", entry.get("modality"))),
            tags=tuple(dict.fromkeys((*_tuple_of_str(entry.get("tags")), *_tuple_of_str(entry.get("benchmark_kind"))))),
            source=BenchmarkSource.from_dict(source_data),
            dataset=BenchmarkDatasetRef.from_dict(dataset_data),
            dataset_refs=dataset_refs,
            integration_status=_normalize_integration_status(entry.get("integration", entry.get("integration_status", "planned"))),
            runner=BenchmarkRunnerSpec.from_dict(runner_data),
            metrics=metrics,
            open_source_status=_normalize_open_source_status(entry.get("open_source_status", release_status)),
            release_status=release_status,
            maturity=_normalize_maturity_status(entry.get("maturity", entry.get("integration", "planned"))),
            official_benchmark_verified=_entry_bool_flag(entry, "official_benchmark_verified"),
            integration_evidence=_entry_bool_flag(entry, "integration_evidence"),
            leaderboard_valid=_entry_bool_flag(entry, "leaderboard_valid"),
            base_model_dependencies=_tuple_of_str(entry.get("base_model_dependencies")),
            optional_base_model_dependencies=_tuple_of_str(entry.get("optional_base_model_dependencies")),
            requires=_entry_requires(entry),
            blockers=_entry_blockers(entry),
            data_refs=_require_mapping(entry.get("data_refs", {}), "BenchmarkZooEntry.data_refs"),
            runner_availability=_require_mapping(
                entry.get("runner_availability", {}),
                "BenchmarkZooEntry.runner_availability",
            ),
            notes=_entry_notes(entry),
        )


def iter_benchmark_zoo_payloads(payload: Any) -> list[Mapping[str, Any]]:
    """Polymorphically extracts list of benchmark mappings from a heterogeneous JSON payload.

    Args:
        payload: The raw parsed mapping or list.

    Returns:
        A list of parsed dictionary mappings representing benchmark-zoo entries.

    Raises:
        TypeError: If the payload is of an unsupported type.
    """
    if isinstance(payload, Mapping):
        for key in ("benchmarks", "entries", "benchmark_zoo", "manifests"):
            value = payload.get(key)
            if isinstance(value, list):
                return [_require_mapping(item, f"{key} item") for item in value]
        return [payload]
    if isinstance(payload, list):
        return [_require_mapping(item, "benchmark_zoo list item") for item in payload]
    raise TypeError(f"benchmark-zoo payload must be an object or list, got {type(payload).__name__}")


def load_entries(path: str | Path) -> tuple[BenchmarkZooEntry, ...]:
    """Loads and instantiates a collection of BenchmarkZooEntry objects from a manifest file.

    Args:
        path: Path to the YAML or JSON manifest file.

    Returns:
        A tuple of validated BenchmarkZooEntry objects.
    """
    payload = load_manifest(Path(path))
    return tuple(BenchmarkZooEntry.from_dict(item) for item in iter_benchmark_zoo_payloads(payload))
