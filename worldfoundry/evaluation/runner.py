"""In-process evaluation runners.

Provides lazy loading for embodied run symbols to optimize lean CLI entry paths.
"""

# ruff: noqa: F822 - embodied symbols listed in __all__ are provided lazily by __getattr__.

from worldfoundry.evaluation.api import GenerationRequest

from .tasks.execution.orchestration.cache import (
    GENERATION_CACHE_MODES,
    GENERATION_RESULT_CACHE_SCHEMA_VERSION,
    CacheKey,
    GenerationCacheRecord,
    GenerationCacheStats,
    GenerationResultCache,
    cache_paths_from_stats,
    canonical_json_bytes,
    canonical_json_dumps,
    file_sha256,
    generation_cache_hit_metadata,
    generation_cache_payload,
    generation_request_cacheable,
    json_sha256,
    make_cache_key,
    make_generation_cache_key,
    normalize_generation_cache_mode,
    normalize_json,
    run_generation_with_cache,
    sha256_hex,
)
from .tasks.execution.orchestration.contract import (
    ContractRunner,
    ContractRunRequest,
    ContractRunResult,
    execute_contract_run,
    run_contract,
)
from .tasks.execution.orchestration.evaluate import (
    EVALUATE_RUN_REQUEST_SCHEMA_VERSION,
    EVALUATE_RUN_RESULT_SCHEMA_VERSION,
    BuiltinExistingResultsMetric,
    EvaluateRunRequest,
    EvaluateRunResult,
    execute_evaluate_run,
    run_evaluate,
)
from .tasks.execution.orchestration.existing_results import (
    ExistingResultsRunner,
    ExistingResultsRunRequest,
    ExistingResultsRunResult,
    execute_existing_results,
    run_existing_results,
)
from .tasks.execution.orchestration.fidelity import (
    EVALUATION_PROVENANCE_SCHEMA_VERSION,
    EvaluationFidelity,
)
from .tasks.execution.orchestration.materialize import (
    DEFAULT_CONTROL_KEYS,
    MATERIALIZED_REQUESTS_SCHEMA_VERSION,
    MaterializedRequests,
    materialize_generation_requests,
    materialize_requests,
    materialize_requests_from_benchmark,
    materialize_requests_from_dataset_manifest,
)
from .tasks.execution.orchestration.model_benchmark import (
    MODEL_BENCHMARK_RESULT_SCHEMA_VERSION,
    MODEL_BENCHMARK_RUN_SCHEMA_VERSION,
    ModelBenchmarkRunRequest,
    ModelBenchmarkRunResult,
    run_model_benchmark,
)
from .tasks.execution.orchestration.model_benchmark_suite import (
    MODEL_BENCHMARK_SUITE_RESULT_SCHEMA_VERSION,
    MODEL_BENCHMARK_SUITE_SCHEMA_VERSION,
    ModelBenchmarkSuiteRequest,
    ModelBenchmarkSuiteResult,
    get_model_benchmark_suite_preset,
    list_model_benchmark_suite_presets,
    run_model_benchmark_suite,
)
from .tasks.execution.orchestration.plan import (
    RUN_PLAN_SCHEMA_VERSION,
    RunPlan,
    build_run_plan,
    build_run_plan_from_task_registry,
    evaluate_request_from_run_plan,
    load_run_plan,
    validate_run_plan,
    write_run_plan,
)
from .tasks.execution.orchestration.service import (
    GenerateAndScoreIntent,
    ModelBenchmarkIntent,
    PreparedEvaluation,
    ReproduceIntent,
    ReproductionRecipe,
    ScoreArtifactsIntent,
    ScoreResultsIntent,
    execute_evaluation,
    execute_prepared_evaluation,
    prepare_evaluation,
)


def __getattr__(name: str):
    """Lazily expose embodied run symbols to avoid import side-effects in lean CLI entrypaths."""
    if name in {
        "VLA_VA_WAM_RUN_REQUEST_SCHEMA_VERSION",
        "VlaVaWamRunRequest",
        "build_vla_va_wam_evaluate_request",
        "execute_vla_va_wam_run",
        "run_vla_va_wam",
    }:
        from .tasks.embodied import evaluate as run_embodied_evaluate

        value = getattr(run_embodied_evaluate, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CacheKey",
    "ContractRunRequest",
    "ContractRunResult",
    "ContractRunner",
    "EVALUATE_RUN_REQUEST_SCHEMA_VERSION",
    "EVALUATE_RUN_RESULT_SCHEMA_VERSION",
    "EVALUATION_PROVENANCE_SCHEMA_VERSION",
    "MATERIALIZED_REQUESTS_SCHEMA_VERSION",
    "MODEL_BENCHMARK_RESULT_SCHEMA_VERSION",
    "MODEL_BENCHMARK_RUN_SCHEMA_VERSION",
    "MODEL_BENCHMARK_SUITE_RESULT_SCHEMA_VERSION",
    "MODEL_BENCHMARK_SUITE_SCHEMA_VERSION",
    "RUN_PLAN_SCHEMA_VERSION",
    "VLA_VA_WAM_RUN_REQUEST_SCHEMA_VERSION",
    "BuiltinExistingResultsMetric",
    "DEFAULT_CONTROL_KEYS",
    "GENERATION_CACHE_MODES",
    "GENERATION_RESULT_CACHE_SCHEMA_VERSION",
    "EvaluateRunRequest",
    "EvaluateRunResult",
    "EvaluationFidelity",
    "ExistingResultsRunRequest",
    "ExistingResultsRunResult",
    "ExistingResultsRunner",
    "GenerationCacheRecord",
    "GenerationCacheStats",
    "GenerationResultCache",
    "GenerationRequest",
    "GenerateAndScoreIntent",
    "MaterializedRequests",
    "ModelBenchmarkRunRequest",
    "ModelBenchmarkRunResult",
    "ModelBenchmarkIntent",
    "ModelBenchmarkSuiteRequest",
    "ModelBenchmarkSuiteResult",
    "RunPlan",
    "PreparedEvaluation",
    "ReproduceIntent",
    "ReproductionRecipe",
    "ScoreArtifactsIntent",
    "ScoreResultsIntent",
    "VlaVaWamRunRequest",
    "cache_paths_from_stats",
    "canonical_json_bytes",
    "canonical_json_dumps",
    "build_vla_va_wam_evaluate_request",
    "execute_contract_run",
    "execute_evaluate_run",
    "execute_existing_results",
    "execute_evaluation",
    "execute_prepared_evaluation",
    "execute_vla_va_wam_run",
    "file_sha256",
    "build_run_plan",
    "build_run_plan_from_task_registry",
    "get_model_benchmark_suite_preset",
    "generation_cache_hit_metadata",
    "generation_cache_payload",
    "generation_request_cacheable",
    "json_sha256",
    "list_model_benchmark_suite_presets",
    "make_cache_key",
    "make_generation_cache_key",
    "normalize_json",
    "normalize_generation_cache_mode",
    "prepare_evaluation",
    "evaluate_request_from_run_plan",
    "load_run_plan",
    "run_contract",
    "run_evaluate",
    "run_generation_with_cache",
    "run_model_benchmark",
    "run_model_benchmark_suite",
    "run_vla_va_wam",
    "run_existing_results",
    "sha256_hex",
    "validate_run_plan",
    "write_run_plan",
    "materialize_generation_requests",
    "materialize_requests",
    "materialize_requests_from_benchmark",
    "materialize_requests_from_dataset_manifest",
]
