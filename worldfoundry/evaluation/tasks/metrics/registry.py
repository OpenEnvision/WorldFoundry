"""Metric registry, module protocol, artifact helpers, and offline evaluators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from worldfoundry.evaluation.api import (
    AggregateResult,
    ArtifactRef,
    GenerationRequest,
    GenerationResult,
    MetricResult,
    MetricSpec,
    is_generation_result_successful,
)
from worldfoundry.evaluation.api.artifacts import local_path_for_uri
from worldfoundry.evaluation.api.json_contract import to_plain
from worldfoundry.evaluation.api.metrics import aggregate_mean

# ---------------------------------------------------------------------------
# Metric module protocol (one-folder-per-metric packages)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricModuleSpec:
    """Declarative metadata exported by one-folder-per-metric modules."""

    id: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    family: str = "distribution"
    required_artifacts: tuple[str, ...] = ()
    higher_is_better: bool | None = True
    tags: tuple[str, ...] = ()
    parameterized_prefix: str | None = None
    implementation: str | None = None

    def registry_entry(self) -> "MetricRegistryEntry":
        return MetricRegistryEntry(
            id=self.id,
            aliases=self.aliases,
            description=self.description,
            family=self.family,
            parameterized_prefix=self.parameterized_prefix,
            required_artifacts=self.required_artifacts,
            higher_is_better=self.higher_is_better,
            tags=self.tags,
        )

    @property
    def spec(self) -> MetricSpec:
        return self.registry_entry().spec


@runtime_checkable
class MetricModule(Protocol):
    """Minimum surface for auto-discovered metric packages."""

    METRIC_MODULE: MetricModuleSpec

    def compute(self, *args: Any, **kwargs: Any) -> Any:
        ...


def metric_module_from_globals(
    *,
    metric_id: str,
    aliases: tuple[str, ...] = (),
    description: str = "",
    family: str = "distribution",
    required_artifacts: tuple[str, ...] = (),
    higher_is_better: bool | None = True,
    tags: tuple[str, ...] = (),
    parameterized_prefix: str | None = None,
    implementation: str | None = None,
) -> MetricModuleSpec:
    return MetricModuleSpec(
        id=metric_id,
        aliases=aliases,
        description=description,
        family=family,
        required_artifacts=required_artifacts,
        higher_is_better=higher_is_better,
        tags=tags,
        parameterized_prefix=parameterized_prefix,
        implementation=implementation,
    )


# ---------------------------------------------------------------------------
# Artifact normalization
# ---------------------------------------------------------------------------

def _metric_key(value: str) -> str:
    return value.strip().casefold().replace("_", "-")


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, ArtifactRef):
        return value.to_dict()
    if isinstance(value, Mapping):
        return value
    return None


def _artifact_record_from_mapping(value: Mapping[str, Any], default_kind: str = "generated_artifact") -> dict[str, Any]:
    uri = value.get("uri") or value.get("path") or value.get("resolved_path") or value.get("file") or value.get("filename")
    kind = value.get("kind") or value.get("name") or value.get("key") or default_kind
    return {
        "uri": None if uri is None else str(uri),
        "kind": str(kind),
        "exists": value.get("exists"),
        "metadata": to_plain(value),
    }


def _artifact_record_from_path(value: str | Path, default_kind: str = "generated_artifact") -> dict[str, Any]:
    uri = str(value)
    local_path = local_path_for_uri(uri)
    return {
        "uri": uri,
        "kind": default_kind,
        "exists": None if local_path is None else local_path.is_file(),
        "metadata": {},
    }


def _artifact_records_from_sequence(value: Sequence[Any], default_kind: str = "generated_artifact") -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for item in value:
        mapping = _as_mapping(item)
        if mapping is not None:
            records.append(_artifact_record_from_mapping(mapping, default_kind=default_kind))
        elif isinstance(item, (str, Path)):
            records.append(_artifact_record_from_path(item, default_kind=default_kind))
    return tuple(records)


def normalize_artifact_records(value: Any) -> tuple[dict[str, Any], ...]:
    mapping = _as_mapping(value)
    if mapping is not None:
        artifacts = mapping.get("artifacts")
        generated_files = mapping.get("generated_files")
        if isinstance(artifacts, Mapping):
            return tuple(
                _artifact_record_from_mapping(
                    item if isinstance(item, Mapping) else {"uri": item, "kind": name},
                    default_kind=str(name),
                )
                for name, item in artifacts.items()
            )
        if isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes)):
            return _artifact_records_from_sequence(artifacts)
        if isinstance(generated_files, Sequence) and not isinstance(generated_files, (str, bytes)):
            return _artifact_records_from_sequence(generated_files)
        if any(key in mapping for key in ("uri", "path", "resolved_path", "file", "filename")):
            return (_artifact_record_from_mapping(mapping),)
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _artifact_records_from_sequence(value)
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_dir():
            return tuple(_artifact_record_from_path(item) for item in sorted(path.rglob("*")) if item.is_file())
        return (_artifact_record_from_path(value),)
    return ()


def _artifact_exists(record: Mapping[str, Any], base_dir: str | Path | None = None) -> bool:
    exists = record.get("exists")
    if exists is not None:
        return bool(exists)
    uri = record.get("uri")
    if not uri:
        return False
    path = local_path_for_uri(str(uri), base_dir=base_dir)
    return True if path is None else path.is_file()


def _artifact_aliases(record: Mapping[str, Any]) -> set[str]:
    aliases = {str(record.get("kind") or "")}
    uri = record.get("uri")
    if uri:
        path = Path(str(uri))
        aliases.update({str(uri), path.name, path.stem})
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("name", "key", "kind"):
            if metadata.get(key):
                aliases.add(str(metadata[key]))
    return {_metric_key(alias) for alias in aliases if alias}


def missing_artifacts(
    required_artifacts: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    *,
    base_dir: str | Path | None = None,
) -> tuple[str, ...]:
    available: set[str] = set()
    for record in records:
        if _artifact_exists(record, base_dir=base_dir):
            available.update(_artifact_aliases(record))
    return tuple(name for name in required_artifacts if _metric_key(name) not in available)


# ---------------------------------------------------------------------------
# Offline existing-results metrics
# ---------------------------------------------------------------------------

def _numeric_container_values(container: Any) -> dict[str, float]:
    if not isinstance(container, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in container.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _numeric_values(result: GenerationResult) -> dict[str, float]:
    values: dict[str, float] = {}
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    extra = metadata.get("extra") if isinstance(metadata.get("extra"), Mapping) else {}
    for container in (
        metadata.get("metrics"),
        metadata.get("scores"),
        extra.get("metrics"),
        extra.get("scores"),
        extra,
    ):
        values.update(_numeric_container_values(container))
    for key, value in _numeric_container_values(result.timings).items():
        values[f"timing:{key}"] = value
    return values


def _is_failed(result: GenerationResult) -> bool:
    return not is_generation_result_successful(result)


#: Canonical metric ids `BuiltinExistingResultsMetric.__call__` actually implements.
OFFLINE_COMPUTABLE_METRIC_IDS: frozenset[str] = frozenset(
    {"artifact_count", "required_artifacts_present", "numeric"}
)
#: Parameterized prefixes `BuiltinExistingResultsMetric.__call__` actually implements.
OFFLINE_COMPUTABLE_METRIC_PREFIXES: tuple[str, ...] = ("has_artifact:", "numeric:")


def is_offline_computable_metric_id(canonical_metric_id: str) -> bool:
    """Return True when the offline existing-results evaluator can compute this id."""
    return canonical_metric_id in OFFLINE_COMPUTABLE_METRIC_IDS or canonical_metric_id.startswith(
        OFFLINE_COMPUTABLE_METRIC_PREFIXES
    )


class BuiltinExistingResultsMetric:
    """Metric callable for scoring materialized generation outputs."""

    name = "existing_results_metrics"
    version = "3"
    higher_is_better = None

    def __init__(self, metrics: Sequence[str] = (), required_artifacts: Sequence[str] = ()) -> None:
        validation = validate_metric_ids(tuple(str(item) for item in (metrics or ())), raise_on_error=True)
        self.metrics = tuple(item["canonical_metric_id"] for item in validation["metrics"])
        # Fail fast on registered-but-not-offline-computable ids (e.g. fid, cmmd,
        # vqa_score).  Accepting them here would silently drop the metric from
        # the scorecard, which is worse than refusing the run request.
        unsupported = tuple(metric for metric in self.metrics if not is_offline_computable_metric_id(metric))
        if unsupported:
            raise MetricRegistryError(
                "metrics not computable by the offline existing-results evaluator: "
                f"{', '.join(sorted(unsupported))}. This evaluator only computes "
                "artifact_count, required_artifacts_present, numeric, "
                "has_artifact:<name>, and numeric:<name>; other registered metric "
                "ids need the benchmark in-tree evaluator or their metric package "
                "compute() API."
            )
        self.required_artifacts = tuple(str(item) for item in (required_artifacts or ()))
        self.parameters = {"metrics": self.metrics}
        self.metric_ids = tuple(dict.fromkeys([
            *(key.removeprefix("numeric:") for key in self.metrics),
            *(f"has_artifact:{key}" for key in self.required_artifacts),
            *(["required_artifacts_present"] if self.required_artifacts else []),
            "generation_success",
        ]))

    def resolve_numeric_fields(self, results: Sequence[GenerationResult]) -> "BuiltinExistingResultsMetric":
        """Expand the numeric wildcard once for this run without mutating a reusable metric."""
        if "numeric" not in self.metrics:
            return self
        fields = sorted({
            key for result in results if is_generation_result_successful(result)
            for key in _numeric_values(result)
        })
        if not fields:
            return self
        metrics = tuple(dict.fromkeys(
            value for key in self.metrics
            for value in ([f"numeric:{field}" for field in fields] if key == "numeric" else [key])
        ))
        return BuiltinExistingResultsMetric(metrics, self.required_artifacts)

    def metric_definition(self, metric_id: str) -> dict[str, Any]:
        """Keep numeric field selection separate from artifact checks and metric groups."""
        artifact_metric = (
            metric_id in {"artifact_count", "required_artifacts_present", "generation_success"}
            or metric_id.startswith("has_artifact:")
        )
        direction = True if artifact_metric else None
        if not artifact_metric:
            try:
                direction = default_metric_registry().get(metric_id).higher_is_better
            except KeyError:
                pass
        return {
            "parameters": {} if artifact_metric else {"source": metric_id},
            "required_artifacts": self.required_artifacts if metric_id == "required_artifacts_present" else (),
            "higher_is_better": direction,
        }

    def compute_sample(self, request: GenerationRequest, result: GenerationResult) -> list[MetricResult]:
        values = self(request, result)["metrics"]
        rows = [
            MetricResult(sample_id=request.sample_id, metric_id=key, raw_value=value)
            for key, value in values.items()
        ]
        missing = [
            key.removeprefix("numeric:") for key in self.metrics
            if key.startswith("numeric:") and key.removeprefix("numeric:") not in values
        ]
        if "numeric" in self.metrics and not _numeric_values(result):
            missing.append("numeric")
        rows.extend(
            MetricResult(sample_id=request.sample_id, metric_id=key, valid=False, coverage=0,
                         skip_reason="numeric value not emitted by runner")
            for key in missing
        )
        return rows

    def aggregate(self, results: Sequence[MetricResult]) -> AggregateResult:
        return aggregate_mean(results[0].metric_id if results else self.name, results)

    def __call__(self, request: GenerationRequest, result: GenerationResult) -> dict[str, Any]:
        del request
        artifacts = set(result.artifacts)
        numeric_values = _numeric_values(result)
        values: dict[str, float] = {}

        for metric in self.metrics:
            if metric == "artifact_count":
                values["artifact_count"] = float(len(artifacts))
            elif metric == "required_artifacts_present":
                values["required_artifacts_present"] = self._required_artifacts_present(artifacts)
            elif metric == "numeric":
                values.update(numeric_values)
            elif metric.startswith("has_artifact:"):
                artifact_name = metric.split(":", 1)[1]
                values[f"has_artifact:{artifact_name}"] = 1.0 if artifact_name in artifacts else 0.0
            elif metric.startswith("numeric:"):
                metric_name = metric.split(":", 1)[1]
                if metric_name in numeric_values:
                    values[metric_name] = numeric_values[metric_name]

        for artifact_name in self.required_artifacts:
            values.setdefault(f"has_artifact:{artifact_name}", 1.0 if artifact_name in artifacts else 0.0)
        if self.required_artifacts:
            values.setdefault("required_artifacts_present", self._required_artifacts_present(artifacts))
        values.setdefault("generation_success", 0.0 if _is_failed(result) else 1.0)
        return {"metrics": values}

    def _required_artifacts_present(self, artifacts: set[str]) -> float:
        if not self.required_artifacts:
            return 1.0
        return 1.0 if all(name in artifacts for name in self.required_artifacts) else 0.0


# ---------------------------------------------------------------------------
# Auto-discovered metric packages
# ---------------------------------------------------------------------------

_DISCOVERABLE_METRIC_PACKAGES: tuple[str, ...] = (
    "worldfoundry.evaluation.tasks.metrics.inception_score",
    "worldfoundry.evaluation.tasks.metrics.fid",
    "worldfoundry.evaluation.tasks.metrics.kid",
    "worldfoundry.evaluation.tasks.metrics.precision_recall",
    "worldfoundry.evaluation.tasks.metrics.improved_precision_recall",
    "worldfoundry.evaluation.tasks.metrics.fwd",
    "worldfoundry.evaluation.tasks.metrics.ppl",
    "worldfoundry.evaluation.tasks.metrics.mind",
    "worldfoundry.evaluation.tasks.metrics.fvd",
    "worldfoundry.evaluation.tasks.metrics.fvmd",
    "worldfoundry.evaluation.tasks.metrics.lpips",
    "worldfoundry.evaluation.tasks.metrics.ssim",
    "worldfoundry.evaluation.tasks.metrics.ms_ssim",
    "worldfoundry.evaluation.tasks.metrics.psnr",
    "worldfoundry.evaluation.tasks.metrics.fsim",
    "worldfoundry.evaluation.tasks.metrics.rke",
    "worldfoundry.evaluation.tasks.metrics.fdd",
    "worldfoundry.evaluation.tasks.metrics.cis",
    "worldfoundry.evaluation.tasks.metrics.rnd",
    "worldfoundry.evaluation.tasks.metrics.lqs",
    "worldfoundry.evaluation.tasks.metrics.semsr",
    "worldfoundry.evaluation.tasks.metrics.irs",
    "worldfoundry.evaluation.tasks.metrics.cas",
    "worldfoundry.evaluation.tasks.metrics.manipulation_direction",
    "worldfoundry.evaluation.tasks.metrics.vs_similarity",
    "worldfoundry.evaluation.tasks.metrics.quality_loss",
    "worldfoundry.evaluation.tasks.metrics.object_wise_consistency",
    "worldfoundry.evaluation.tasks.metrics.facesim_cur",
    "worldfoundry.evaluation.tasks.metrics.sadpad",
    "worldfoundry.evaluation.tasks.metrics.opens2v",
    "worldfoundry.evaluation.tasks.metrics.laion_aesthetic",
)


def load_metric_module_spec(module_name: str) -> MetricModuleSpec:
    module = import_module(module_name)
    spec = getattr(module, "METRIC_MODULE", None)
    if spec is None:
        raise AttributeError(f"{module_name} must export METRIC_MODULE")
    if not isinstance(spec, MetricModuleSpec):
        raise TypeError(f"{module_name}.METRIC_MODULE must be MetricModuleSpec")
    return spec


def load_metric_module_specs(module_name: str) -> tuple[MetricModuleSpec, ...]:
    module = import_module(module_name)
    multi = getattr(module, "METRIC_MODULES", None)
    if multi is not None:
        specs = tuple(multi)
        for spec in specs:
            if not isinstance(spec, MetricModuleSpec):
                raise TypeError(f"{module_name}.METRIC_MODULES entries must be MetricModuleSpec")
        return specs
    return (load_metric_module_spec(module_name),)


def discover_metric_registry_entries(
    packages: Sequence[str] = _DISCOVERABLE_METRIC_PACKAGES,
    *,
    skip_unavailable: bool = False,
) -> tuple[MetricRegistryEntry, ...]:
    entries: list[MetricRegistryEntry] = []
    for package in packages:
        try:
            specs = load_metric_module_specs(package)
        except (ImportError, ModuleNotFoundError):
            if skip_unavailable:
                continue
            raise
        for spec in specs:
            entries.append(spec.registry_entry())
    return tuple(entries)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class MetricRegistryError(ValueError):
    """Base error for metric-registry failures."""


class DuplicateMetricRegistryKeyError(MetricRegistryError):
    """Raised when a metric registry key is registered more than once."""


class UnknownMetricRegistryKeyError(KeyError):
    """Raised when a metric id cannot be resolved."""


def _normalise_key(value: str, field_name: str = "metric key") -> str:
    """Normalize and validate a metric registry lookup key."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value.casefold().replace("_", "-")


def _parameterized_suffix(entry: "MetricRegistryEntry", metric_id: str) -> str | None:
    """Return parameterized suffix when ``metric_id`` matches ``entry`` prefix."""
    if not entry.parameterized_prefix:
        return None
    prefix, separator, suffix = metric_id.partition(":")
    if not separator or not suffix:
        return None
    expected_prefix = entry.parameterized_prefix.rstrip(":")
    if _normalise_key(prefix, "metric prefix") == _normalise_key(expected_prefix, "metric prefix"):
        return suffix
    return None


@dataclass(frozen=True)
class MetricRegistryEntry:
    """Declarative metric metadata (aliases, orientation, artifact requirements)."""

    id: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    family: str = "builtin"
    parameterized_prefix: str | None = None
    required_artifacts: tuple[str, ...] = ()
    higher_is_better: bool | None = True
    tags: tuple[str, ...] = ()

    def keys(self) -> tuple[str, ...]:
        """Return canonical id plus aliases for registry indexing."""
        seen: set[str] = set()
        ordered: list[str] = []
        for key in (self.id, *self.aliases):
            normalized = _normalise_key(key)
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(key)
        return tuple(ordered)

    @property
    def spec(self) -> MetricSpec:
        """Export public :class:`MetricSpec` for this entry."""
        # Only the offline-computable builtin ids are implemented by
        # BuiltinExistingResultsMetric; claiming it for every registered entry
        # (fid, cmmd, vqa_score, ...) would misdescribe ~55 metric packages.
        offline = is_offline_computable_metric_id(self.id) or (
            self.parameterized_prefix is not None
            and self.parameterized_prefix in OFFLINE_COMPUTABLE_METRIC_PREFIXES
        )
        return MetricSpec(
            id=self.id,
            aliases=self.aliases,
            description=self.description,
            family=self.family,
            required_artifacts=self.required_artifacts,
            higher_is_better=self.higher_is_better,
            implementation=(
                "worldfoundry.evaluation.tasks.metrics.registry:BuiltinExistingResultsMetric" if offline else None
            ),
            tags=self.tags,
            metadata={
                "parameterized_prefix": self.parameterized_prefix,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry as a metric spec dict."""
        payload = self.spec.to_dict()
        payload["parameterized_prefix"] = self.parameterized_prefix
        return payload


# ---------------------------------------------------------------------------
# Built-in metric entries
# ---------------------------------------------------------------------------

BUILTIN_METRIC_REGISTRY_ENTRIES: tuple[MetricRegistryEntry, ...] = (
    MetricRegistryEntry(
        id="artifact_count",
        aliases=("artifact-count",),
        description="Counts output artifacts present on each generation result.",
        family="existing_results",
        higher_is_better=True,
        tags=("builtin", "result_only"),
    ),
    MetricRegistryEntry(
        id="required_artifacts_present",
        aliases=("required-artifacts-present",),
        description="Returns 1 when every required artifact name is present, otherwise 0.",
        family="existing_results",
        higher_is_better=True,
        tags=("builtin", "result_only"),
    ),
    MetricRegistryEntry(
        id="has_artifact",
        description="Parameterized metric: has_artifact:<artifact_name>.",
        family="existing_results",
        parameterized_prefix="has_artifact:",
        higher_is_better=True,
        tags=("builtin", "result_only", "parameterized"),
    ),
    MetricRegistryEntry(
        id="numeric",
        description="Emits all numeric metrics/scores/timings found on a generation result.",
        family="existing_results",
        higher_is_better=True,
        tags=("builtin", "result_only"),
    ),
    MetricRegistryEntry(
        id="numeric_value",
        aliases=("numeric-field",),
        description="Parameterized metric: numeric:<metric_name>.",
        family="existing_results",
        parameterized_prefix="numeric:",
        higher_is_better=True,
        tags=("builtin", "result_only", "parameterized"),
    ),
    MetricRegistryEntry(
        id="vqa_score",
        aliases=("vqascore", "vqa-score"),
        description="VQAScore image/video-text alignment scorer from runners/_scorers/vqa_score.",
        family="scorer",
        tags=("scorer", "multimodal", "vqa"),
    ),
    MetricRegistryEntry(
        id="clip_score",
        aliases=("clipscore", "clip-score"),
        description="CLIPScore image/video-text similarity scorer from runners/_scorers/clip_score.",
        family="scorer",
        tags=("scorer", "multimodal", "clip"),
    ),
    MetricRegistryEntry(
        id="itm_score",
        aliases=("itmscore", "itm-score"),
        description="ITMScore image/video-text matching scorer from runners/_scorers/itm_score.",
        family="scorer",
        tags=("scorer", "multimodal", "itm"),
    ),
    MetricRegistryEntry(
        id="cmmd",
        aliases=("clip-mmd", "clip_mmd"),
        description="CLIP Maximum Mean Discrepancy (CVPR 2024 CMMD metric).",
        family="distribution",
        higher_is_better=False,
        tags=("distribution", "image_generation", "clip"),
    ),
    MetricRegistryEntry(
        id="dino_similarity",
        aliases=("dino-similarity", "dino_sim"),
        description="Cosine similarity between DINOv2 embeddings (pairwise).",
        family="perceptual",
        higher_is_better=True,
        tags=("perceptual", "condition_consistency", "dino"),
    ),
    MetricRegistryEntry(
        id="jedi",
        aliases=("jedi-mmd", "video-jedi"),
        description="JEDi (Video JEPA Distance) distribution metric for generated video sets.",
        family="distribution",
        higher_is_better=False,
        tags=("distribution", "video_generation", "jedi"),
    ),
    MetricRegistryEntry(
        id="clean_fid",
        aliases=("clean-fid", "improved-fid"),
        description="Clean-FID / Improved FID with proper image resizing (lower is better).",
        family="distribution",
        higher_is_better=False,
        tags=("distribution", "image_generation", "fid_family"),
    ),
    MetricRegistryEntry(
        id="dreamsim",
        aliases=("dream-sim",),
        description="DreamSim perceptual distance between image pairs (lower is more similar).",
        family="perceptual",
        higher_is_better=False,
        tags=("perceptual", "condition_consistency"),
    ),
    MetricRegistryEntry(
        id="vendi_score",
        aliases=("vendi-score", "vendi"),
        description="Vendi Score measuring diversity of feature embeddings (higher is better).",
        family="distribution",
        higher_is_better=True,
        tags=("distribution", "image_generation", "diversity"),
    ),
    MetricRegistryEntry(
        id="rarity_score",
        aliases=("rarity-score", "rs"),
        description="Rarity Score measuring uncommonness of synthesized images (higher is rarer).",
        family="distribution",
        higher_is_better=True,
        tags=("distribution", "image_generation", "diversity"),
    ),
    MetricRegistryEntry(
        id="facescore",
        aliases=("face-score", "face_score"),
        description="FaceScore face quality reward model (OPPO-Mente-Lab).",
        family="scorer",
        higher_is_better=True,
        tags=("scorer", "face", "quality"),
    ),
    MetricRegistryEntry(
        id="artscore",
        aliases=("art-score", "art_score"),
        description="ArtScore artness evaluation model.",
        family="scorer",
        higher_is_better=True,
        tags=("scorer", "aesthetics", "art"),
    ),
    MetricRegistryEntry(
        id="trend",
        aliases=("trend-jsd", "trend_jsd"),
        description="TREND metric via TGND parameter JSD on Inception embeddings (lower is better).",
        family="distribution",
        higher_is_better=False,
        tags=("distribution", "image_generation"),
    ),
    MetricRegistryEntry(
        id="fld",
        aliases=("feature-likelihood-divergence", "fls"),
        description="Feature Likelihood Divergence for generative model evaluation (lower is better).",
        family="distribution",
        higher_is_better=False,
        tags=("distribution", "image_generation", "diversity"),
    ),
    MetricRegistryEntry(
        id="multimodal_mid",
        aliases=("mid-metric", "mid_metric", "mutual-information-divergence"),
        description="Mutual Information Divergence for text-image alignment (Naver mid.metric; not MIND).",
        family="distribution",
        higher_is_better=True,
        tags=("distribution", "multimodal", "clip"),
    ),
    MetricRegistryEntry(
        id="fjd",
        aliases=("frechet-joint-distance", "frechet_joint_distance"),
        description="Fréchet Joint Distance for conditional generative models (lower is better).",
        family="distribution",
        higher_is_better=False,
        tags=("distribution", "conditional_generation"),
    ),
    MetricRegistryEntry(
        id="crosslid",
        aliases=("cross-lid", "cross_lid"),
        description="Cross Local Intrinsic Dimensionality diversity metric (lower is more diverse).",
        family="distribution",
        higher_is_better=False,
        tags=("distribution", "image_generation", "diversity"),
    ),
    MetricRegistryEntry(
        id="cpbd",
        aliases=("cpbd-sharpness",),
        description="CPBD cumulative probability of blur detection sharpness metric (higher is sharper).",
        family="perceptual",
        higher_is_better=True,
        tags=("perceptual", "no_reference", "blur"),
    ),
    MetricRegistryEntry(
        id="cfid",
        aliases=("conditional-fid", "conditional_fid"),
        description="Conditional Fréchet Inception Distance (Soloveitchik paired CFID).",
        family="distribution",
        higher_is_better=False,
        tags=("distribution", "conditional_generation"),
    ),
    MetricRegistryEntry(
        id="ssd",
        aliases=("semantic-similarity-distance", "semantic_similarity_distance"),
        description="Semantic Similarity Distance for text-to-image evaluation (lower is better).",
        family="distribution",
        higher_is_better=False,
        tags=("distribution", "text_to_image", "multimodal"),
    ),
    MetricRegistryEntry(
        id="linear_separability",
        aliases=("linear-separability", "ls-metric", "stylegan-ls"),
        description="StyleGAN linear separability score from latent SVM confusion matrix.",
        family="distribution",
        higher_is_better=True,
        tags=("distribution", "image_generation", "generative_model"),
    ),
    MetricRegistryEntry(
        id="mask_accuracy",
        aliases=("mask-accuracy", "mask_acc"),
        description="Pixel-wise mask accuracy between predicted and ground-truth masks.",
        family="perceptual",
        higher_is_better=True,
        tags=("perceptual", "segmentation", "condition_consistency"),
    ),
    MetricRegistryEntry(
        id="object_detection",
        aliases=("object-detection", "object-detection-rate", "detection-success-rate"),
        description="WorldScore-style object detection phrase-matching success rate.",
        family="perceptual",
        higher_is_better=True,
        tags=("perceptual", "detection", "worldscore"),
    ),
)


class MetricRegistry:
    """Registry of declarative metric ids and offline metric factories."""

    def __init__(
        self,
        entries: Sequence[MetricRegistryEntry | Mapping[str, Any]] = (),
        *,
        include_builtins: bool = True,
    ) -> None:
        """Initialize registry with optional entries and built-ins."""
        self._entries: dict[str, MetricRegistryEntry] = {}
        self._by_key: dict[str, MetricRegistryEntry] = {}
        self._discoverable = include_builtins
        self._discovered = False
        if include_builtins:
            for entry in BUILTIN_METRIC_REGISTRY_ENTRIES:
                self.register(entry)
        for entry in entries:
            self.register(entry)

    def register(
        self,
        entry: MetricRegistryEntry | Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> MetricRegistryEntry:
        """Register one metric entry and its aliases."""
        if not isinstance(entry, MetricRegistryEntry):
            entry = MetricRegistryEntry(
                id=str(entry["id"]),
                aliases=tuple(str(item) for item in entry.get("aliases", ())),
                description=str(entry.get("description", "")),
                family=str(entry.get("family", "custom")),
                parameterized_prefix=entry.get("parameterized_prefix"),
                required_artifacts=tuple(str(item) for item in entry.get("required_artifacts", ())),
                higher_is_better=entry.get("higher_is_better", True),
                tags=tuple(str(item) for item in entry.get("tags", ())),
            )
        normalized_keys = tuple(_normalise_key(key) for key in entry.keys())
        entry_key = _normalise_key(entry.id)
        if not replace:
            for key in normalized_keys:
                if key in self._by_key:
                    raise DuplicateMetricRegistryKeyError(f"metric registry key already exists: {key}")
        else:
            existing = self._entries.get(entry_key)
            for key in normalized_keys:
                owner = self._by_key.get(key)
                if owner is not None and owner is not existing:
                    raise DuplicateMetricRegistryKeyError(f"metric registry key already exists: {key}")
            if existing is not None:
                for key in tuple(_normalise_key(key) for key in existing.keys()):
                    if self._by_key.get(key) is existing:
                        self._by_key.pop(key)
        for key in normalized_keys:
            self._by_key[key] = entry
        self._entries[entry_key] = entry
        return entry

    def list(self, *, include_discovered: bool = False) -> tuple[MetricRegistryEntry, ...]:
        """Return registered entries in insertion order."""
        if include_discovered:
            self._register_discoverable_entries()
        return tuple(self._entries.values())

    def get(self, metric_id: str) -> MetricRegistryEntry:
        """Resolve ``metric_id`` or alias to a registry entry."""
        return self.resolve_key(metric_id)

    def _resolve_key_or_none(self, metric_id: str) -> MetricRegistryEntry | None:
        """Resolve ``metric_id`` without raising for unknown ids."""
        key = _normalise_key(metric_id, "metric_id")
        if key in self._by_key:
            return self._by_key[key]
        for entry in self._entries.values():
            if _parameterized_suffix(entry, metric_id) is not None:
                return entry
        self._register_discoverable_entries()
        if key in self._by_key:
            return self._by_key[key]
        for entry in self._entries.values():
            if _parameterized_suffix(entry, metric_id) is not None:
                return entry
        return None

    def _register_discoverable_entries(self) -> None:
        """Register optional metric-package metadata only when needed."""
        if self._discovered or not self._discoverable:
            return
        self._discovered = True
        for entry in discover_metric_registry_entries(skip_unavailable=True):
            if self._resolve_key_or_none(entry.id) is None:
                self.register(entry)

    def resolve_key(self, metric_id: str) -> MetricRegistryEntry:
        """Resolve ``metric_id`` or raise :class:`UnknownMetricRegistryKeyError`."""
        entry = self._resolve_key_or_none(metric_id)
        if entry is not None:
            return entry
        raise UnknownMetricRegistryKeyError(f"unknown metric id: {metric_id!r}")

    def canonical_metric_id(self, metric_id: str) -> str:
        """Map alias or parameterized id to canonical registry id."""
        entry = self.resolve_key(metric_id)
        suffix = _parameterized_suffix(entry, metric_id)
        if suffix is not None:
            return f"{entry.parameterized_prefix}{suffix}"
        return entry.id

    def create_existing_results_metric(
        self,
        metrics: Sequence[str] = (),
        required_artifacts: Sequence[str] = (),
    ) -> BuiltinExistingResultsMetric | None:
        """Build :class:`BuiltinExistingResultsMetric` from metric id list."""
        if not metrics and not required_artifacts:
            return None
        for metric in metrics:
            self.resolve_key(str(metric))
        return BuiltinExistingResultsMetric(metrics=metrics, required_artifacts=required_artifacts)

    def validate_ids(self, metrics: Sequence[str]) -> dict[str, Any]:
        """Bulk-validate metric ids and return resolved/unknown lists."""
        unknown: list[str] = []
        resolved: list[dict[str, Any]] = []
        for metric in metrics:
            metric_id = str(metric)
            entry = self._resolve_key_or_none(metric_id)
            if entry is None:
                unknown.append(metric_id)
                continue
            suffix = _parameterized_suffix(entry, metric_id)
            canonical = f"{entry.parameterized_prefix}{suffix}" if suffix is not None else entry.id
            resolved.append(
                {
                    "metric_id": metric_id,
                    "canonical_metric_id": canonical,
                    "registry_id": entry.id,
                    "parameterized": suffix is not None,
                }
            )
        return {
            "ok": not unknown,
            "metrics": resolved,
            "unknown_metrics": unknown,
        }


# ---------------------------------------------------------------------------
# Module-level registry helpers
# ---------------------------------------------------------------------------

_DEFAULT_METRIC_REGISTRY = MetricRegistry()


def default_metric_registry() -> MetricRegistry:
    """Return process-wide default :class:`MetricRegistry`."""
    return _DEFAULT_METRIC_REGISTRY


def list_metric_registry_entries() -> tuple[MetricRegistryEntry, ...]:
    """List all registered metric entries."""
    return default_metric_registry().list(include_discovered=True)


def create_existing_results_metric(
    metrics: Sequence[str] = (),
    required_artifacts: Sequence[str] = (),
) -> BuiltinExistingResultsMetric | None:
    """Create offline metric evaluator via the default registry."""
    return default_metric_registry().create_existing_results_metric(metrics, required_artifacts)


def validate_metric_ids(
    metrics: Sequence[str],
    *,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Validate metric ids via the default registry."""
    payload = default_metric_registry().validate_ids(metrics)
    if raise_on_error and not payload["ok"]:
        metric = payload["unknown_metrics"][0]
        raise UnknownMetricRegistryKeyError(
            "unsupported built-in metric "
            f"{metric!r}; use artifact_count, required_artifacts_present, "
            "has_artifact:<name>, numeric, or numeric:<name>"
        )
    return payload


__all__ = [
    "BUILTIN_METRIC_REGISTRY_ENTRIES",
    "BuiltinExistingResultsMetric",
    "DuplicateMetricRegistryKeyError",
    "MetricModule",
    "MetricModuleSpec",
    "MetricRegistry",
    "MetricRegistryEntry",
    "MetricRegistryError",
    "OFFLINE_COMPUTABLE_METRIC_IDS",
    "OFFLINE_COMPUTABLE_METRIC_PREFIXES",
    "UnknownMetricRegistryKeyError",
    "is_offline_computable_metric_id",
    "create_existing_results_metric",
    "default_metric_registry",
    "discover_metric_registry_entries",
    "list_metric_registry_entries",
    "load_metric_module_spec",
    "load_metric_module_specs",
    "metric_module_from_globals",
    "missing_artifacts",
    "normalize_artifact_records",
    "validate_metric_ids",
]
