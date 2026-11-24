"""Embodied closed-loop rollout runner.

Binds an in-tree policy adapter to a native simulator and executes step-by-step
rollouts, returning standardized :class:`GenerationResult` payloads with
``vla_va_wam`` scorecard metadata.

Sections:

* **EmbodiedClosedLoopRunner** — :class:`WorldModelRunner` orchestrator.
* **Spec validation** — policy/simulator action and observation checks.
* **Factories** — convenience builders for common benchmark ids.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from worldfoundry.evals.api import (
    GenerationRequest,
    GenerationResult,
    WorldModelConfig,
    WorldModelRunner,
)
from worldfoundry.evals.tasks.embodied.contracts import (
    CAPABILITY_MULTIMODAL_OBSERVATION,
    CAPABILITY_SESSION_CONTROL,
    CAPABILITY_VLA_ACTION_PREDICTION,
    CAPABILITY_VLA_POLICY_ROLLOUT,
)
from worldfoundry.evals.tasks.embodied.policy_adapter import build_policy_adapter, normalize_action_payload
from worldfoundry.evals.tasks.embodied.simulators.registry import (
    get_simulator_entry,
    resolve_simulator_class,
)
from worldfoundry.evals.tasks.embodied.simulators.specs import DimSpec, check_specs

logger = logging.getLogger(__name__)


def _maybe_spec(obj: Any, method_name: str) -> Mapping[str, DimSpec | Mapping[str, Any]]:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return {}
    try:
        value = method()
    except NotImplementedError:
        return {}
    return dict(value or {}) if isinstance(value, Mapping) else {}


def validate_policy_simulator_specs(policy_adapter: Any, benchmark: Any) -> list[str]:
    """Return warning strings for policy/simulator action and observation mismatches."""
    policy_action = _maybe_spec(policy_adapter, "get_action_spec")
    policy_observation = _maybe_spec(policy_adapter, "get_observation_spec")
    simulator_action = _maybe_spec(benchmark, "get_action_spec")
    simulator_observation = _maybe_spec(benchmark, "get_observation_spec")
    return check_specs(policy_action, simulator_action, policy_observation, simulator_observation)


class EmbodiedClosedLoopRunner(WorldModelRunner):
    """Native closed-loop rollout runner for embodied benchmarks.

    Execution flow:

    * Resolve simulator class from :mod:`simulators.registry`.
    * For each :class:`GenerationRequest`, reset env and step until done.
    * Query policy for actions; package episode metrics into ``GenerationResult``.
    """

    def __init__(
        self,
        model_id: str,
        benchmark_id: str,
        policy_runner: Any = None,
        benchmark_kwargs: Mapping[str, Any] | None = None,
        capabilities: Collection[str] | None = None,
    ) -> None:
        """Initialize runner with model id, benchmark id, and optional policy."""
        self._model_id = str(model_id)
        self._benchmark_id = str(benchmark_id)
        self._policy_runner = policy_runner
        self._benchmark_kwargs = dict(benchmark_kwargs or {})
        self._capabilities = set(
            capabilities
            or {
                "closed_loop_rollout",
                "native_embodied_eval",
                CAPABILITY_VLA_ACTION_PREDICTION,
                CAPABILITY_VLA_POLICY_ROLLOUT,
                CAPABILITY_SESSION_CONTROL,
                CAPABILITY_MULTIMODAL_OBSERVATION,
            }
        )

        self._benchmark_instance: Any = None
        self._benchmark_instance_key: str | None = None
        self._benchmark_class: type | None = None
        self._task_cache: tuple[Mapping[str, Any], ...] | None = None
        self._spec_warnings: list[str] = []
        self._load_benchmark_class()

    @property
    def model_id(self) -> str:  # ty: ignore[invalid-property-type-override]  # Identity is read-only configuration.
        """Configured model identifier."""
        return self._model_id

    @property
    def capabilities(self) -> Collection[str]:  # ty: ignore[invalid-property-type-override]  # Capability tags are read-only configuration.
        """Supported runner capability tags."""
        return self._capabilities

    def _load_benchmark_class(self) -> None:
        """Load simulator class from the embodied registry."""
        entry = get_simulator_entry(self._benchmark_id)
        if entry is None:
            logger.error("Unsupported benchmark_id in embodied rollout runner: %s", self._benchmark_id)
            return
        try:
            self._benchmark_class = resolve_simulator_class(self._benchmark_id)
            logger.info("Loaded embodied simulator class: %s", entry.import_path)
        except (ImportError, AttributeError, KeyError, TypeError) as exc:
            logger.error("Failed to load embodied simulator class for %s: %s", self._benchmark_id, exc)

    def _get_benchmark_instance(self, task_name: str) -> Any:
        """Get or create cached simulator instance for ``task_name``."""
        if self._benchmark_class is None:
            raise RuntimeError(f"StepBenchmark class for '{self._benchmark_id}' was not loaded.")

        if self._benchmark_instance is not None and self._benchmark_instance_key == task_name:
            return self._benchmark_instance

        # Release previous simulator when switching tasks.
        if self._benchmark_instance is not None:
            try:
                self._benchmark_instance.cleanup()
            except Exception as exc:
                logger.warning("Error during benchmark cleanup: %s", exc)

        kwargs = dict(self._benchmark_kwargs)
        if self._constructor_accepts("task_name"):
            kwargs["task_name"] = task_name

        if self._benchmark_id == "robotwin" and "task_config" not in kwargs:
            kwargs["task_config"] = "demo_clean"

        self._benchmark_instance = self._benchmark_class(**kwargs)
        self._benchmark_instance_key = task_name
        self._task_cache = None
        self._spec_warnings = self._validate_specs(self._benchmark_instance)
        return self._benchmark_instance

    def _constructor_accepts(self, name: str) -> bool:
        if self._benchmark_class is None:
            return False
        try:
            signature = inspect.signature(self._benchmark_class.__init__)
        except (TypeError, ValueError):
            return False
        params = signature.parameters
        return name in params or any(item.kind == inspect.Parameter.VAR_KEYWORD for item in params.values())

    @classmethod
    def from_config(cls, config: WorldModelConfig) -> EmbodiedClosedLoopRunner:
        """Construct runner from :class:`WorldModelConfig`."""
        params = dict(config.parameters or {})
        benchmark_id = params.get("benchmark_id", "robotwin")
        benchmark_kwargs = params.get("benchmark_kwargs", {})
        policy_runner = params.get("policy_runner")
        server_url = params.get("server_url")
        if policy_runner is None and str(config.model_id).strip().lower() not in {"", "zero", "zero-policy"}:
            from worldfoundry.evals.tasks.embodied.policy_adapter import build_policy_adapter

            policy_runner = build_policy_adapter(config.model_id, params, server_url=server_url)
        capabilities = params.get("capabilities")
        return cls(
            model_id=config.model_id,
            benchmark_id=benchmark_id,
            policy_runner=policy_runner,
            benchmark_kwargs=benchmark_kwargs,
            capabilities=capabilities,
        )

    def _infer_action(self, obs: dict[str, Any], task_instruction: str) -> dict[str, Any]:
        """Predict actions from observation dict and task instruction."""
        if self._policy_runner is not None:
            if hasattr(self._policy_runner, "predict"):
                return normalize_action_payload(self._policy_runner.predict(obs, task_instruction))
            if callable(self._policy_runner):
                return normalize_action_payload(self._policy_runner(obs, task_instruction))

        # Deterministic zero policy for wiring tests when no real policy is supplied.
        if self._benchmark_id == "ai2thor":
            return {"token": "forward"}
        action_dim = 14 if self._benchmark_id == "robotwin" else 7
        return {"actions": [0.0] * action_dim}

    def _request_controls(self, request: GenerationRequest) -> dict[str, Any]:
        controls = dict(request.controls or {})
        sample_controls = controls.get("sample_controls")
        if isinstance(sample_controls, Mapping):
            controls.update(sample_controls)
        controls.update(dict(request.generation_kwargs or {}))
        return controls

    def _request_value(self, request: GenerationRequest, *keys: str, default: Any = None) -> Any:
        inputs = dict(request.inputs or {})
        controls = self._request_controls(request)
        for key in keys:
            if key in inputs:
                return inputs[key]
            if key in controls:
                return controls[key]
        return default

    def _benchmark_tasks(self, benchmark: Any) -> tuple[Mapping[str, Any], ...]:
        if self._task_cache is not None:
            return self._task_cache
        get_tasks = getattr(benchmark, "get_tasks", None)
        if not callable(get_tasks):
            self._task_cache = ()
            return self._task_cache
        try:
            tasks = tuple(task for task in get_tasks() if isinstance(task, Mapping))
        except Exception as exc:
            logger.debug("Benchmark %s did not expose tasks: %s", self._benchmark_id, exc)
            tasks = ()
        self._task_cache = tasks
        return tasks

    @staticmethod
    def _task_index_from_name(task_name: str) -> int | None:
        match = re.search(r"(?:task[_-]?|/)(\d+)$", task_name)
        if match:
            return int(match.group(1))
        return None

    def _select_benchmark_task(self, benchmark: Any, request: GenerationRequest) -> Mapping[str, Any]:
        tasks = self._benchmark_tasks(benchmark)
        if not tasks:
            return {}

        raw_task_id = self._request_value(request, "task_id", "task_index", default=None)
        if raw_task_id is None:
            raw_task_id = self._task_index_from_name(request.task_name)
        if raw_task_id is not None:
            try:
                task_id = int(raw_task_id)
            except (TypeError, ValueError):
                task_id = None
            if task_id is not None:
                for task in tasks:
                    if int(task.get("task_id", -1)) == task_id:
                        return task
                if 0 <= task_id < len(tasks):
                    return tasks[task_id]

        requested = {request.task_name}
        for key in ("task_name", "language_instruction", "instruction"):
            value = self._request_value(request, key, default=None)
            if value is not None:
                requested.add(str(value))
        for task in tasks:
            if str(task.get("name")) in requested or str(task.get("suite")) in requested:
                return task
        return tasks[0]

    def _task_spec(self, benchmark: Any, request: GenerationRequest, index: int) -> dict[str, Any]:
        selected = dict(self._select_benchmark_task(benchmark, request))
        seed = self._request_value(request, "seed", "reset_seed", default=100000 + index)
        episode_idx = self._request_value(request, "episode_idx", "episode_id", default=index)
        instruction = self._request_value(
            request,
            "language_instruction",
            "instruction",
            default=selected.get("name") or f"Perform the {request.task_name} task.",
        )
        task_spec = {
            **selected,
            "name": str(selected.get("name") or request.task_name),
            "suite": str(selected.get("suite") or self._benchmark_kwargs.get("suite") or self._benchmark_id),
            "seed": int(seed),
            "episode_idx": int(episode_idx),
            "instruction": str(instruction),
            "request_task_name": request.task_name,
        }
        if "task_id" not in task_spec:
            maybe_task_id = self._request_value(request, "task_id", "task_index", default=None)
            if maybe_task_id is not None:
                task_spec["task_id"] = int(maybe_task_id)
        return task_spec

    def _validate_specs(self, benchmark: Any) -> list[str]:
        if self._policy_runner is None:
            return []
        try:
            warnings = validate_policy_simulator_specs(self._policy_runner, benchmark)
            for warning in warnings:
                logger.warning("Embodied policy/simulator spec mismatch: %s", warning)
            return list(warnings)
        except Exception as exc:
            logger.debug("Embodied spec validation skipped: %s", exc)
            return []

    def generate(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationResult]:
        """Run closed-loop rollouts for a batch of requests."""
        if self._benchmark_class is None:
            raise RuntimeError(f"StepBenchmark class for '{self._benchmark_id}' was not loaded.")

        results: list[GenerationResult] = []
        for index, request in enumerate(requests):
            task_name = request.task_name
            try:
                benchmark = self._get_benchmark_instance(task_name)

                task_spec = self._task_spec(benchmark, request, index)

                logger.info("Starting native closed-loop rollout. task=%s, seed=%d", task_name, task_spec["seed"])
                start_episode = getattr(self._policy_runner, "start_episode", None)
                if callable(start_episode):
                    start_episode(task_spec)

                # Reset → step loop → harvest episode metrics.
                raw_obs = benchmark.reset(task_spec)
                done = False
                steps = 0
                step_result = None

                # Read max steps from metadata
                max_steps_meta = benchmark.get_metadata().get("max_steps", 400)
                try:
                    max_steps = int(max_steps_meta)
                except (ValueError, TypeError):
                    max_steps = 400

                # Step until done or max_steps.
                while not done and steps < max_steps:
                    # Translate raw sensor data into clean, structured spec format
                    obs_spec = benchmark.make_obs(raw_obs, task_spec)
                    task_desc = obs_spec.get("task_description", task_spec["instruction"])

                    # Policy inference and env step.
                    action_pred = self._infer_action(obs_spec, task_desc)
                    step_result = benchmark.step(action_pred)
                    raw_obs = step_result.obs
                    done = benchmark.check_done(step_result)
                    steps += 1

                # Harvest episode metrics from the final step.
                episode_result = (
                    benchmark.get_step_result(step_result)
                    if step_result is not None
                    else {"success": False, "steps": 0, "failure_reason": "no_steps_executed"}
                )
                success_val = 1.0 if episode_result.get("success", False) else 0.0
                end_episode = getattr(self._policy_runner, "end_episode", None)
                if callable(end_episode):
                    end_episode(dict(episode_result))

                # Package step metrics directly into the vla_va_wam scorecard structure
                vla_va_wam_payload = {
                    "metrics": {
                        "success_rate": success_val,
                        "task_success": success_val,
                        **{
                            k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                            for k, v in episode_result.items()
                        },
                    }
                }

                results.append(
                    GenerationResult(
                        sample_id=request.sample_id,
                        request_id=request.request_id,
                        model_id=self._model_id,
                        status="success",
                        metadata={
                            "vla_va_wam": vla_va_wam_payload,
                            "steps_run": steps,
                            "policy_source": "provided_policy"
                            if self._policy_runner is not None
                            else "zero_policy_wiring_check",
                            "official_runtime_executed": True,
                            "normalizer_only": False,
                            "integration_evidence": True,
                            "spec_warnings": list(self._spec_warnings),
                            "task_spec": {key: value for key, value in task_spec.items() if key not in {"task_obj"}},
                        },
                    )
                )
                logger.info("Rollout finished. success=%s, steps=%d", episode_result.get("success", False), steps)

            except Exception as exc:
                logger.exception(
                    "Closed-loop rollout execution crashed on task '%s' seed '%s'", task_name, request.sample_id
                )
                results.append(
                    GenerationResult(
                        sample_id=request.sample_id,
                        request_id=request.request_id,
                        model_id=self._model_id,
                        status="failed",
                        error=f"NativeRolloutError: {exc}",
                    )
                )

        return results

    def cleanup(self) -> None:
        """Release simulator and policy resources."""
        if self._benchmark_instance is not None:
            try:
                self._benchmark_instance.cleanup()
            except Exception as exc:
                logger.warning("Failed to safely dispose simulator instance: %s", exc)
            self._benchmark_instance = None
            self._benchmark_instance_key = None
        cleanup_policy = getattr(self._policy_runner, "cleanup", None)
        if callable(cleanup_policy):
            try:
                cleanup_policy()
            except Exception as exc:
                logger.warning("Failed to safely dispose policy runner: %s", exc)


def build_embodied_closed_loop_runner(
    model_id: str,
    benchmark_id: str,
    model_parameters: Mapping[str, Any] | None = None,
    *,
    server_url: str | None = None,
    zero_policy: bool = False,
) -> EmbodiedClosedLoopRunner:
    """Build a native closed-loop runner with an in-process or WebSocket policy."""
    params = dict(model_parameters or {})
    benchmark_kwargs = dict(params.get("benchmark_kwargs") or {})
    if "suite" in params and "suite" not in benchmark_kwargs:
        benchmark_kwargs["suite"] = params["suite"]
    if "seed" in params and "seed" not in benchmark_kwargs:
        benchmark_kwargs["seed"] = params["seed"]

    policy_runner = params.get("policy_runner")
    if policy_runner is None and not zero_policy:
        policy_runner = build_policy_adapter(model_id, params, server_url=server_url or params.get("server_url"))

    return EmbodiedClosedLoopRunner(
        model_id=model_id,
        benchmark_id=benchmark_id,
        policy_runner=policy_runner,
        benchmark_kwargs=benchmark_kwargs,
        capabilities=params.get("capabilities"),
    )


def build_ai2thor_closed_loop_runner(
    model_id: str = "zero",
    model_parameters: Mapping[str, Any] | None = None,
    *,
    server_url: str | None = None,
    zero_policy: bool = False,
) -> EmbodiedClosedLoopRunner:
    """Convenience factory for AI2-THOR closed-loop evaluation."""
    return build_embodied_closed_loop_runner(
        model_id=model_id,
        benchmark_id="ai2thor",
        model_parameters=model_parameters,
        server_url=server_url,
        zero_policy=zero_policy or str(model_id).strip().lower() in {"", "zero", "zero-policy"},
    )


def build_libero_closed_loop_runner(
    model_id: str = "openvla",
    model_parameters: Mapping[str, Any] | None = None,
    *,
    server_url: str | None = None,
    zero_policy: bool = False,
) -> EmbodiedClosedLoopRunner:
    """Convenience factory for LIBERO closed-loop evaluation."""
    return build_embodied_closed_loop_runner(
        model_id=model_id,
        benchmark_id="libero",
        model_parameters=model_parameters,
        server_url=server_url,
        zero_policy=zero_policy,
    )


__all__ = [
    "EmbodiedClosedLoopRunner",
    "build_ai2thor_closed_loop_runner",
    "build_embodied_closed_loop_runner",
    "build_libero_closed_loop_runner",
    "validate_policy_simulator_specs",
]
