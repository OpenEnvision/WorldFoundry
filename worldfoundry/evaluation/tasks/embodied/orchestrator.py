"""Async orchestrator for WorldFoundry embodied closed-loop evaluations."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult, MetricResult
from worldfoundry.evaluation.tasks.embodied.config_loader import load_canonical_embodied_config
from worldfoundry.evaluation.tasks.embodied.docker_runner import inside_docker, run_embodied_via_docker
from worldfoundry.evaluation.tasks.embodied.materialize_rollouts import materialize_embodied_rollout_requests
from worldfoundry.evaluation.tasks.embodied.metrics import metric_suite
from worldfoundry.evaluation.tasks.embodied.rollout_runner import build_embodied_closed_loop_runner
from worldfoundry.evaluation.tasks.execution.orchestration.evaluate import (
    EVALUATE_RUN_RESULT_SCHEMA_VERSION,
    EvaluateRunResult,
)
from worldfoundry.evaluation.tasks.execution.orchestration.existing_results import run_existing_results
from worldfoundry.evaluation.utils import write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbodiedOrchestratorResult:
    """Result object returned by embodied orchestrator runs."""

    evaluate_result: EvaluateRunResult
    eval_id: str
    output_dir: Path
    raw_results: tuple[GenerationResult, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = self.evaluate_result.to_dict()
        payload["eval_id"] = self.eval_id
        return payload


def _shard_requests(
    requests: Sequence[GenerationRequest],
    *,
    shard_id: int | None,
    num_shards: int | None,
) -> tuple[GenerationRequest, ...]:
    if shard_id is None and num_shards is None:
        return tuple(requests)
    if shard_id is None or num_shards is None:
        raise ValueError("shard_id and num_shards must be provided together")
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards})")
    return tuple(request for index, request in enumerate(requests) if index % num_shards == shard_id)


def _metric_callable(metric_ids: Sequence[str] | None = None):
    metrics = metric_suite(metric_ids or ("generation_success", "task_success", "success_rate"), track="vla")

    def compute(request: GenerationRequest, result: GenerationResult) -> list[MetricResult]:
        return [metric.compute_sample(request, result) for metric in metrics]

    return compute


class EmbodiedEvalOrchestrator:
    """Runs embodied benchmark configs with async episode isolation and progress files."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        shard_id: int | None = None,
        num_shards: int | None = None,
        eval_id: str | None = None,
        no_save: bool = False,
    ) -> None:
        self.config = dict(config)
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.eval_id = eval_id or str(uuid.uuid4())
        self.no_save = bool(no_save)
        self.root_output_dir = Path(self.config.get("output_dir", "./results")).resolve()
        self.output_dir = (
            self.root_output_dir / f"shard{shard_id}of{num_shards}"
            if shard_id is not None and num_shards is not None
            else self.root_output_dir
        )
        self.progress_path = self.root_output_dir / self._progress_name()

    def _progress_name(self) -> str:
        if self.shard_id is not None and self.num_shards is not None:
            return f"embodied_shard{self.shard_id}of{self.num_shards}.progress"
        return "embodied.progress"

    def _update_progress(self, completed: int, total: int, errors: int) -> None:
        self.root_output_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.progress_path.with_suffix(".tmp")
        write_json(tmp, {"completed": completed, "total": total, "errors": errors, "eval_id": self.eval_id})
        tmp.replace(self.progress_path)

    async def run(self) -> EmbodiedOrchestratorResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_cfg = dict(self.config.get("model") or {})
        model_id = str(model_cfg.get("id") or self.config.get("model_id") or "openvla")
        model_parameters = dict(model_cfg.get("parameters") or {})
        server_cfg = dict(self.config.get("server") or {})
        server_url = server_cfg.get("url")

        all_requests: list[GenerationRequest] = []
        all_results: list[GenerationResult] = []
        benchmark_contexts: list[dict[str, Any]] = []
        total_requests = 0
        for bench_cfg in self.config.get("benchmarks") or ():
            requests = materialize_embodied_rollout_requests(bench_cfg)
            total_requests += len(_shard_requests(requests, shard_id=self.shard_id, num_shards=self.num_shards))
        self._update_progress(0, total_requests, 0)

        completed = 0
        errors = 0
        for bench_cfg in self.config.get("benchmarks") or ():
            benchmark_id = str(bench_cfg.get("benchmark_id") or bench_cfg.get("id") or "libero")
            benchmark_params = dict(bench_cfg.get("params") or {})
            run_parameters = {
                **model_parameters,
                "benchmark_id": benchmark_id,
                "benchmark_kwargs": benchmark_params,
            }
            requests = _shard_requests(
                materialize_embodied_rollout_requests(bench_cfg),
                shard_id=self.shard_id,
                num_shards=self.num_shards,
            )
            runner = build_embodied_closed_loop_runner(
                model_id,
                benchmark_id,
                run_parameters,
                server_url=str(server_url) if server_url else None,
                zero_policy=bool(model_parameters.get("zero_policy")),
            )
            benchmark_contexts.append({"benchmark_id": benchmark_id, "params": benchmark_params, "request_count": len(requests)})
            try:
                for request in requests:
                    result_batch = await asyncio.to_thread(runner.generate, [request])
                    result = tuple(result_batch)[0]
                    all_requests.append(request)
                    all_results.append(result)
                    completed += 1
                    if result.error:
                        errors += 1
                    self._update_progress(completed, total_requests, errors)
            finally:
                runner.cleanup()

        if self.progress_path.exists():
            self.progress_path.unlink()

        if self.no_save:
            minimal = self._minimal_result(len(all_requests), errors)
            return EmbodiedOrchestratorResult(
                evaluate_result=minimal,
                eval_id=self.eval_id,
                output_dir=self.output_dir,
                raw_results=tuple(all_results),
            )

        metric_ids = tuple(str(item) for item in self.config.get("metric_ids") or ("task_success", "success_rate"))
        existing_result = run_existing_results(
            output_dir=self.output_dir,
            requests=all_requests,
            results=all_results,
            metric=_metric_callable(metric_ids),
            benchmark={
                "suite": "vla_va_wam",
                "benchmark_name": self.config.get("id", "embodied_eval"),
                "benchmark_id": ",".join(item["benchmark_id"] for item in benchmark_contexts) or "embodied",
                "task_type": "embodied_closed_loop",
                "evaluation_protocol": "worldfoundry_embodied_async",
                "official_runtime_executed": True,
                "normalizer_only": False,
                "integration_evidence": True,
            },
            model={
                "model_type": "embodied_policy",
                "model_id": model_id,
                "model_name": model_id,
                "server_url": server_url,
            },
            dataset={
                "dataset_id": self.config.get("id", "embodied_eval"),
                "name": self.config.get("id", "embodied_eval"),
                "split": "closed_loop",
                "sample_count": len(all_requests),
            },
            run_id=self.eval_id,
            run_metadata={
                "schema_version": "worldfoundry-embodied-orchestrator-run",
                "delegate_runner": "worldfoundry.embodied-orchestrator",
                "benchmarks": benchmark_contexts,
                "shard": (
                    {"id": self.shard_id, "total": self.num_shards}
                    if self.shard_id is not None and self.num_shards is not None
                    else None
                ),
            },
        )
        evaluate_result = EvaluateRunResult(
            schema_version=EVALUATE_RUN_RESULT_SCHEMA_VERSION,
            mode="model",
            delegate_runner="worldfoundry.embodied-orchestrator",
            status=existing_result.status,
            exit_code=existing_result.exit_code,
            output_dir=existing_result.output_dir,
            manifest_path=existing_result.manifest_path,
            execution_plan_path=existing_result.execution_plan_path,
            scorecard_path=existing_result.scorecard_path,
            sample_count=existing_result.sample_count,
            successful_sample_count=existing_result.successful_sample_count,
            failed_sample_count=existing_result.failed_sample_count,
            artifact_count=existing_result.artifact_count,
        )
        return EmbodiedOrchestratorResult(
            evaluate_result=evaluate_result,
            eval_id=self.eval_id,
            output_dir=self.output_dir,
            raw_results=tuple(all_results),
        )

    def _minimal_result(self, sample_count: int, errors: int) -> EvaluateRunResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        placeholder = self.output_dir / "no_save.json"
        write_json(placeholder, {"eval_id": self.eval_id, "sample_count": sample_count, "errors": errors})
        return EvaluateRunResult(
            schema_version=EVALUATE_RUN_RESULT_SCHEMA_VERSION,
            mode="model",
            delegate_runner="worldfoundry.embodied-orchestrator",
            status="completed_with_failures" if errors else "succeeded",
            exit_code=0,
            output_dir=self.output_dir,
            manifest_path=placeholder,
            execution_plan_path=placeholder,
            scorecard_path=placeholder,
            sample_count=sample_count,
            successful_sample_count=sample_count - errors,
            failed_sample_count=errors,
            artifact_count=0,
        )


async def run_embodied_eval(config: Mapping[str, Any], **kwargs: Any) -> EmbodiedOrchestratorResult:
    """Async convenience wrapper."""
    return await EmbodiedEvalOrchestrator(config, **kwargs).run()


def run_embodied_eval_sync(config: Mapping[str, Any], **kwargs: Any) -> EmbodiedOrchestratorResult:
    """Synchronous wrapper for CLI entry points."""
    return asyncio.run(run_embodied_eval(config, **kwargs))


def _docker_result(output_dir: str | Path, exit_code: int) -> EmbodiedOrchestratorResult:
    root = Path(output_dir).resolve()
    scorecard = root / "scorecard.json"
    result = EvaluateRunResult(
        schema_version=EVALUATE_RUN_RESULT_SCHEMA_VERSION,
        mode="model",
        delegate_runner="worldfoundry.embodied-docker",
        status="succeeded" if exit_code == 0 else "failed",
        exit_code=exit_code,
        output_dir=root,
        manifest_path=root / "run_manifest.json",
        execution_plan_path=root / "execution_plan.json",
        scorecard_path=scorecard,
        sample_count=0,
        successful_sample_count=0,
        failed_sample_count=0,
        artifact_count=0,
    )
    return EmbodiedOrchestratorResult(evaluate_result=result, eval_id="", output_dir=root, raw_results=())


def run_embodied_eval_config(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    server_url: str | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
    eval_id: str | None = None,
    no_docker: bool = False,
    no_save: bool = False,
    pull_docker: bool = False,
) -> EmbodiedOrchestratorResult:
    """Run an embodied eval YAML through Docker or the native async orchestrator."""
    config = load_canonical_embodied_config(config_path, output_dir=output_dir, server_url=server_url)
    use_docker = bool((config.get("docker") or {}).get("image")) and not no_docker and not inside_docker()
    if use_docker:
        exit_code = run_embodied_via_docker(
            config,
            shard_id=shard_id,
            num_shards=num_shards,
            eval_id=eval_id,
            no_save=no_save,
            pull=pull_docker,
        )
        return _docker_result(config.get("output_dir", "./results"), exit_code)
    return run_embodied_eval_sync(
        config,
        shard_id=shard_id,
        num_shards=num_shards,
        eval_id=eval_id,
        no_save=no_save,
    )


__all__ = [
    "EmbodiedEvalOrchestrator",
    "EmbodiedOrchestratorResult",
    "run_embodied_eval",
    "run_embodied_eval_config",
    "run_embodied_eval_sync",
]
