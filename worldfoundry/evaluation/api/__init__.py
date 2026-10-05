"""Core stdlib-only evaluation API contracts."""

from .artifacts import (
    ARTIFACT_REF_SCHEMA_VERSION,
    ArtifactRef,
    coerce_artifact_refs,
    enrich_artifact_ref,
    local_path_for_uri,
    restore_artifact_refs,
)
from .generation import (
    GENERATION_REQUEST_SCHEMA_VERSION,
    GENERATION_RESULT_SCHEMA_VERSION,
    GENERATION_SUCCESS_STATUSES,
    GENERATION_UNKNOWN_STATUS,
    GenerationRequest,
    GenerationResult,
    is_generation_result_successful,
    is_generation_status_successful,
    normalize_generation_status,
)
from .metrics import (
    AGGREGATE_RESULT_SCHEMA_VERSION,
    METRIC_RESULT_SCHEMA_VERSION,
    METRIC_SPEC_SCHEMA_VERSION,
    AggregateResult,
    BatchMetric,
    Metric,
    MetricResult,
    MetricSpec,
    align_batch_metric_outputs,
)
from .models import (
    WORLD_MODEL_CONFIG_SCHEMA_VERSION,
    WorldModelConfig,
    WorldModelRunner,
)
from .tasks import (
    BENCHMARK_SPEC_SCHEMA_VERSION,
    EVALUATION_PROTOCOL_SCHEMA_VERSION,
    WORLD_TASK_CONFIG_SCHEMA_VERSION,
    BenchmarkSpec,
    EvaluationProtocolSpec,
    WorldTaskConfig,
)
from .world_model_manifest import WORLD_MODEL_MANIFEST_SCHEMA_VERSION, WorldModelManifest

__all__ = [
    "AGGREGATE_RESULT_SCHEMA_VERSION",
    "ARTIFACT_REF_SCHEMA_VERSION",
    "BENCHMARK_SPEC_SCHEMA_VERSION",
    "EVALUATION_PROTOCOL_SCHEMA_VERSION",
    "GENERATION_REQUEST_SCHEMA_VERSION",
    "GENERATION_RESULT_SCHEMA_VERSION",
    "GENERATION_SUCCESS_STATUSES",
    "GENERATION_UNKNOWN_STATUS",
    "METRIC_RESULT_SCHEMA_VERSION",
    "METRIC_SPEC_SCHEMA_VERSION",
    "WORLD_MODEL_CONFIG_SCHEMA_VERSION",
    "WORLD_MODEL_MANIFEST_SCHEMA_VERSION",
    "WORLD_TASK_CONFIG_SCHEMA_VERSION",
    "AggregateResult",
    "ArtifactRef",
    "BatchMetric",
    "BenchmarkSpec",
    "EvaluationProtocolSpec",
    "GenerationRequest",
    "GenerationResult",
    "Metric",
    "MetricResult",
    "MetricSpec",
    "WorldModelRunner",
    "WorldModelConfig",
    "WorldModelManifest",
    "WorldTaskConfig",
    "align_batch_metric_outputs",
    "coerce_artifact_refs",
    "enrich_artifact_ref",
    "is_generation_result_successful",
    "is_generation_status_successful",
    "local_path_for_uri",
    "normalize_generation_status",
    "restore_artifact_refs",
]
