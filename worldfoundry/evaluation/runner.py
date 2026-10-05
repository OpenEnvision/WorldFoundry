"""In-process evaluation runners (lazy facade).

Pure re-export barrel over ``tasks.execution.orchestration`` and the embodied
evaluate module.  Every symbol is exposed lazily (PEP 562) so lean CLI entry
paths pay no orchestration import cost until a runner symbol is actually used.
"""

# ruff: noqa: F822 - symbols listed in __all__ are provided lazily by __getattr__.

from importlib import import_module
from typing import Any

from worldfoundry.evaluation.api import GenerationRequest

_ORCHESTRATION = "worldfoundry.evaluation.tasks.execution.orchestration"
_EMBODIED_EVALUATE = "worldfoundry.evaluation.tasks.embodied.evaluate"

# Symbol name -> providing module. Kept sorted by module for review-ability.
_LAZY_EXPORTS: dict[str, str] = {
    # ── orchestration.cache ───────────────────────────────────
    "GENERATION_CACHE_MODES": f"{_ORCHESTRATION}.cache",
    "GENERATION_RESULT_CACHE_SCHEMA_VERSION": f"{_ORCHESTRATION}.cache",
    "CacheKey": f"{_ORCHESTRATION}.cache",
    "GenerationCacheRecord": f"{_ORCHESTRATION}.cache",
    "GenerationCacheStats": f"{_ORCHESTRATION}.cache",
    "GenerationResultCache": f"{_ORCHESTRATION}.cache",
    "cache_paths_from_stats": f"{_ORCHESTRATION}.cache",
    "canonical_json_bytes": f"{_ORCHESTRATION}.cache",
    "canonical_json_dumps": f"{_ORCHESTRATION}.cache",
    "file_sha256": f"{_ORCHESTRATION}.cache",
    "generation_cache_hit_metadata": f"{_ORCHESTRATION}.cache",
    "generation_cache_payload": f"{_ORCHESTRATION}.cache",
    "generation_request_cacheable": f"{_ORCHESTRATION}.cache",
    "json_sha256": f"{_ORCHESTRATION}.cache",
    "make_cache_key": f"{_ORCHESTRATION}.cache",
    "make_generation_cache_key": f"{_ORCHESTRATION}.cache",
    "normalize_generation_cache_mode": f"{_ORCHESTRATION}.cache",
    "normalize_json": f"{_ORCHESTRATION}.cache",
    "run_generation_with_cache": f"{_ORCHESTRATION}.cache",
    "sha256_hex": f"{_ORCHESTRATION}.cache",
    # ── orchestration.contract ────────────────────────────────
    "ContractRunner": f"{_ORCHESTRATION}.contract",
    "ContractRunRequest": f"{_ORCHESTRATION}.contract",
    "ContractRunResult": f"{_ORCHESTRATION}.contract",
    "execute_contract_run": f"{_ORCHESTRATION}.contract",
    "run_contract": f"{_ORCHESTRATION}.contract",
    # ── orchestration.evaluate ────────────────────────────────
    "EVALUATE_RUN_REQUEST_SCHEMA_VERSION": f"{_ORCHESTRATION}.evaluate",
    "EVALUATE_RUN_RESULT_SCHEMA_VERSION": f"{_ORCHESTRATION}.evaluate",
    "BuiltinExistingResultsMetric": f"{_ORCHESTRATION}.evaluate",
    "EvaluateRunRequest": f"{_ORCHESTRATION}.evaluate",
    "EvaluateRunResult": f"{_ORCHESTRATION}.evaluate",
    "execute_evaluate_run": f"{_ORCHESTRATION}.evaluate",
    "run_evaluate": f"{_ORCHESTRATION}.evaluate",
    # ── orchestration.existing_results ────────────────────────
    "ExistingResultsRunner": f"{_ORCHESTRATION}.existing_results",
    "ExistingResultsRunRequest": f"{_ORCHESTRATION}.existing_results",
    "ExistingResultsRunResult": f"{_ORCHESTRATION}.existing_results",
    "execute_existing_results": f"{_ORCHESTRATION}.existing_results",
    "run_existing_results": f"{_ORCHESTRATION}.existing_results",
    # ── orchestration.fidelity ────────────────────────────────
    "EVALUATION_PROVENANCE_SCHEMA_VERSION": f"{_ORCHESTRATION}.fidelity",
    "EvaluationFidelity": f"{_ORCHESTRATION}.fidelity",
    # ── orchestration.materialize ─────────────────────────────
    "DEFAULT_CONTROL_KEYS": f"{_ORCHESTRATION}.materialize",
    "MATERIALIZED_REQUESTS_SCHEMA_VERSION": f"{_ORCHESTRATION}.materialize",
    "MaterializedRequests": f"{_ORCHESTRATION}.materialize",
    "materialize_generation_requests": f"{_ORCHESTRATION}.materialize",
    "materialize_requests": f"{_ORCHESTRATION}.materialize",
    "materialize_requests_from_benchmark": f"{_ORCHESTRATION}.materialize",
    "materialize_requests_from_dataset_manifest": f"{_ORCHESTRATION}.materialize",
    # ── orchestration.model_benchmark ─────────────────────────
    "MODEL_BENCHMARK_RESULT_SCHEMA_VERSION": f"{_ORCHESTRATION}.model_benchmark",
    "MODEL_BENCHMARK_RUN_SCHEMA_VERSION": f"{_ORCHESTRATION}.model_benchmark",
    "ModelBenchmarkRunRequest": f"{_ORCHESTRATION}.model_benchmark",
    "ModelBenchmarkRunResult": f"{_ORCHESTRATION}.model_benchmark",
    "run_model_benchmark": f"{_ORCHESTRATION}.model_benchmark",
    # ── orchestration.model_benchmark_suite ───────────────────
    "MODEL_BENCHMARK_SUITE_RESULT_SCHEMA_VERSION": f"{_ORCHESTRATION}.model_benchmark_suite",
    "MODEL_BENCHMARK_SUITE_SCHEMA_VERSION": f"{_ORCHESTRATION}.model_benchmark_suite",
    "ModelBenchmarkSuiteRequest": f"{_ORCHESTRATION}.model_benchmark_suite",
    "ModelBenchmarkSuiteResult": f"{_ORCHESTRATION}.model_benchmark_suite",
    "get_model_benchmark_suite_preset": f"{_ORCHESTRATION}.model_benchmark_suite",
    "list_model_benchmark_suite_presets": f"{_ORCHESTRATION}.model_benchmark_suite",
    "run_model_benchmark_suite": f"{_ORCHESTRATION}.model_benchmark_suite",
    # ── orchestration.plan ────────────────────────────────────
    "RUN_PLAN_SCHEMA_VERSION": f"{_ORCHESTRATION}.plan",
    "RunPlan": f"{_ORCHESTRATION}.plan",
    "build_run_plan": f"{_ORCHESTRATION}.plan",
    "build_run_plan_from_task_registry": f"{_ORCHESTRATION}.plan",
    "evaluate_request_from_run_plan": f"{_ORCHESTRATION}.plan",
    "load_run_plan": f"{_ORCHESTRATION}.plan",
    "validate_run_plan": f"{_ORCHESTRATION}.plan",
    "write_run_plan": f"{_ORCHESTRATION}.plan",
    # ── orchestration.service ─────────────────────────────────
    "GenerateAndScoreIntent": f"{_ORCHESTRATION}.service",
    "ModelBenchmarkIntent": f"{_ORCHESTRATION}.service",
    "PreparedEvaluation": f"{_ORCHESTRATION}.service",
    "ReproduceIntent": f"{_ORCHESTRATION}.service",
    "ReproductionRecipe": f"{_ORCHESTRATION}.service",
    "ScoreArtifactsIntent": f"{_ORCHESTRATION}.service",
    "ScoreResultsIntent": f"{_ORCHESTRATION}.service",
    "execute_evaluation": f"{_ORCHESTRATION}.service",
    "execute_prepared_evaluation": f"{_ORCHESTRATION}.service",
    "prepare_evaluation": f"{_ORCHESTRATION}.service",
    # ── tasks.embodied.evaluate ───────────────────────────────
    "VLA_VA_WAM_RUN_REQUEST_SCHEMA_VERSION": _EMBODIED_EVALUATE,
    "VlaVaWamRunRequest": _EMBODIED_EVALUATE,
    "build_vla_va_wam_evaluate_request": _EMBODIED_EVALUATE,
    "execute_vla_va_wam_run": _EMBODIED_EVALUATE,
    "run_vla_va_wam": _EMBODIED_EVALUATE,
}


def __getattr__(name: str) -> Any:
    """Lazily import and cache runner symbols on first attribute access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily provided symbols in ``dir()`` output."""
    return sorted({*globals(), *_LAZY_EXPORTS})


# NOTE (EF-14): generic cache/hash helpers (sha256_hex, normalize_json,
# canonical_json_bytes/dumps, file_sha256, json_sha256) intentionally stay out
# of __all__ — their canonical public home is the orchestration cache module /
# api.json_contract. They remain importable from here for compatibility.
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
    "build_vla_va_wam_evaluate_request",
    "execute_contract_run",
    "execute_evaluate_run",
    "execute_existing_results",
    "execute_evaluation",
    "execute_prepared_evaluation",
    "execute_vla_va_wam_run",
    "build_run_plan",
    "build_run_plan_from_task_registry",
    "get_model_benchmark_suite_preset",
    "generation_cache_hit_metadata",
    "generation_cache_payload",
    "generation_request_cacheable",
    "list_model_benchmark_suite_presets",
    "make_cache_key",
    "make_generation_cache_key",
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
    "validate_run_plan",
    "write_run_plan",
    "materialize_generation_requests",
    "materialize_requests",
    "materialize_requests_from_benchmark",
    "materialize_requests_from_dataset_manifest",
]
