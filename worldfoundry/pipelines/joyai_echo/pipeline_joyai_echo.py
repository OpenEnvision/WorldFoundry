"""WorldFoundry pipeline contracts for JoyAI-Echo 1.5 and Echo-WM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.joyai_echo import (
    JoyAIEchoLongVideoRuntime,
    JoyAIEchoWMRuntime,
)


def _runtime_options(
    model_path: Any,
    required_components: Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if isinstance(model_path, Mapping):
        options.update(model_path)
    elif model_path is not None:
        options["checkpoint_dir"] = model_path
    options.update(required_components or {})
    options.update(kwargs)
    return options


def _python_executable(options: dict[str, Any]) -> Any:
    executable = options.pop("python_executable", None)
    env_dir = options.pop("python_env_dir", options.pop("conda_dir", None))
    if executable is None and env_dir is not None:
        executable = Path(env_dir).expanduser() / "bin" / "python"
    return executable


def _pop_runner_metadata(options: dict[str, Any]) -> None:
    for key in (
        "profile_id",
        "runtime_profile",
        "pipeline_binding",
        "profile_path",
        "manifest_path",
        "variant_id",
        "repo_id",
        "runner",
        "runner_target",
        "install_profile",
    ):
        options.pop(key, None)


def _external_roots(options: dict[str, Any], *, checkpoint_repo_dir: str) -> tuple[Any, Any, Any]:
    acquisition_root = options.pop("acquisition_root", None)
    hf_models_root = options.pop("hf_models_root", None)
    source_root = options.pop("source_root", options.pop("repo_root", options.pop("runtime_root", None)))
    if source_root is None and acquisition_root is not None:
        source_root = Path(acquisition_root).expanduser() / "jd-opensource--JoyAI-Echo"
    checkpoint_dir = options.pop(
        "checkpoint_dir",
        options.pop("model_dir", options.pop("model_path", options.pop("pretrained_model_path", None))),
    )
    if checkpoint_dir is None and hf_models_root is not None:
        checkpoint_dir = Path(hf_models_root).expanduser() / checkpoint_repo_dir
    return source_root, checkpoint_dir, hf_models_root


def _merge_request_kwargs(
    kwargs: Mapping[str, Any],
    operator_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    request = dict(operator_kwargs or {})
    request.update(kwargs)
    request.pop("task_name", None)
    return request


def _output_paths(
    output_path: str | Path | None,
    *,
    fallback_name: str,
) -> tuple[Path, Path]:
    requested = Path(output_path).expanduser().resolve() if output_path is not None else Path.cwd() / fallback_name
    if requested.suffix.lower() in {".mp4", ".mov", ".webm"}:
        artifact = requested
        report = requested.with_suffix(".json")
    elif requested.suffix.lower() == ".json":
        report = requested
        artifact = requested.with_suffix(".mp4")
    else:
        artifact = requested.with_suffix(".mp4")
        report = requested.with_suffix(".json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return artifact, report


def _preflight_error(preflight: Mapping[str, Any]) -> str:
    missing = [
        *preflight.get("missing_checkpoint_files", []),
        *preflight.get("missing_runtime_files", []),
        *preflight.get("missing_system_tools", []),
        *preflight.get("blocked_reasons", []),
    ]
    details = ", ".join(str(item) for item in missing) or "unknown runtime requirement"
    return f"JoyAI-Echo preflight failed; missing: {details}"


class JoyAIEchoLongVideoPipeline(PipelineABC):
    """Reference-to-video adapter for JoyAI-Echo 1.5 / Echo-LongVideo."""

    MODEL_ID = "joyai-echo-longvideo"

    def __init__(
        self,
        runtime: JoyAIEchoLongVideoRuntime,
        *,
        model_id: str = MODEL_ID,
    ) -> None:
        self.runtime = runtime
        self.model_id = model_id
        self.history: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "JoyAIEchoLongVideoPipeline":
        options = _runtime_options(model_path, required_components, kwargs)
        resolved_model_id = str(options.pop("model_id", model_id or cls.MODEL_ID))
        _pop_runner_metadata(options)
        python_executable = _python_executable(options)
        source_root, checkpoint_dir, hf_models_root = _external_roots(
            options,
            checkpoint_repo_dir="jdopensource--JoyAI-Echo",
        )
        gemma_path = options.pop("gemma_path", options.pop("text_encoder_dir", None))
        if gemma_path is None and hf_models_root is not None:
            gemma_path = Path(hf_models_root).expanduser() / "google--gemma-3-12b-it"
        runtime = JoyAIEchoLongVideoRuntime(
            checkpoint_dir=checkpoint_dir,
            gemma_path=gemma_path,
            source_root=source_root,
            python_executable=python_executable,
            config_path=options.pop("config_path", None),
            precision=str(options.pop("precision", "bf16")),
            consumer=bool(options.pop("consumer", False)),
            low_vram=bool(options.pop("low_vram", False)),
            device=device,
        )
        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"Unknown JoyAI-Echo LongVideo load options: {unknown}")
        return cls(runtime, model_id=resolved_model_id)

    def preflight(self) -> dict[str, Any]:
        return self.runtime.preflight()

    def process(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del video
        request = _merge_request_kwargs(kwargs, operator_kwargs)
        request.setdefault("prompt", prompt)
        request.setdefault("images", images)
        if interactions and "memory_slots" not in request:
            request["memory_slots"] = list(interactions)
        return request

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        execute: bool = False,
        timeout_seconds: int = 7200,
        return_dict: bool = False,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        if not execute:
            raise RuntimeError("JoyAI-Echo LongVideo requires execute=True; dry-run artifacts are not emitted.")
        request = self.process(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            operator_kwargs=operator_kwargs,
            **kwargs,
        )
        artifact_path, report_path = _output_paths(output_path, fallback_name="joyai_echo_longvideo.mp4")
        preflight = self.preflight()
        if preflight["status"] != "ready":
            raise RuntimeError(_preflight_error(preflight))
        plan = self.runtime.build_plan(request=request, output_path=artifact_path, output_root=output_dir)
        run_result = self.runtime.run_plan(plan, timeout_seconds=timeout_seconds, log_dir=report_path.parent)
        generated_files = list(run_result.get("generated_files") or [])
        resolved_artifact = generated_files[0] if generated_files else str(artifact_path)
        result = {
            "status": run_result.get("status", "failed"),
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": resolved_artifact,
            "artifact_files": generated_files,
            "runtime": "joyai_echo.official_external_runtime",
            "backend_quality": "official_external_runtime",
            "request": request,
            "preflight": preflight,
            "runtime_result": run_result,
            "metadata_path": str(report_path),
        }
        if result["status"] != "success":
            result["error"] = run_result.get("error") or "JoyAI-Echo LongVideo execution failed."
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        self.history.append(result)
        if return_dict or result["status"] != "success":
            return result
        return resolved_artifact

    def stream(self, *args: Any, **kwargs: Any) -> dict[str, Any] | str:
        """Use the batch official entrypoint; streaming is not exposed upstream."""
        return self(*args, **kwargs)


class JoyAIEchoWMPipeline(PipelineABC):
    """Image, prompt, and Action-DSL adapter for Echo-WM Base and Flash."""

    MODEL_ID = "joyai-echo-wm"

    def __init__(
        self,
        runtime: JoyAIEchoWMRuntime,
        *,
        model_id: str = MODEL_ID,
    ) -> None:
        self.runtime = runtime
        self.model_id = model_id
        self.history: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "JoyAIEchoWMPipeline":
        options = _runtime_options(model_path, required_components, kwargs)
        resolved_model_id = str(options.pop("model_id", model_id or cls.MODEL_ID))
        _pop_runner_metadata(options)
        python_executable = _python_executable(options)
        source_root, checkpoint_dir, hf_models_root = _external_roots(
            options,
            checkpoint_repo_dir="Echo-Team--Echo-WM",
        )
        gemma_path = options.pop("gemma_path", options.pop("text_encoder_dir", None))
        if gemma_path is None and hf_models_root is not None:
            gemma_path = Path(hf_models_root).expanduser() / "google--gemma-3-12b-it-qat-q4_0-unquantized"
        runtime = JoyAIEchoWMRuntime(
            checkpoint_dir=checkpoint_dir,
            gemma_path=gemma_path,
            source_root=source_root,
            python_executable=python_executable,
            config_path=options.pop("config_path", None),
            variant=str(options.pop("variant", "base")),
            device=device,
        )
        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"Unknown Echo-WM load options: {unknown}")
        return cls(runtime, model_id=resolved_model_id)

    def preflight(self) -> dict[str, Any]:
        return self.runtime.preflight()

    def process(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del video
        request = _merge_request_kwargs(kwargs, operator_kwargs)
        request.setdefault("prompt", prompt)
        request.setdefault("images", images)
        request.setdefault("interactions", interactions)
        return request

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        execute: bool = False,
        timeout_seconds: int = 7200,
        return_dict: bool = False,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        del output_dir
        if not execute:
            raise RuntimeError("Echo-WM requires execute=True; dry-run artifacts are not emitted.")
        request = self.process(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            operator_kwargs=operator_kwargs,
            **kwargs,
        )
        artifact_path, report_path = _output_paths(output_path, fallback_name="joyai_echo_wm.mp4")
        preflight = self.preflight()
        if preflight["status"] != "ready":
            raise RuntimeError(_preflight_error(preflight))
        plan = self.runtime.build_plan(request=request, output_path=artifact_path)
        run_result = self.runtime.run_plan(plan, timeout_seconds=timeout_seconds, log_dir=report_path.parent)
        generated_files = list(run_result.get("generated_files") or [])
        resolved_artifact = generated_files[0] if generated_files else str(artifact_path)
        result = {
            "status": run_result.get("status", "failed"),
            "model_id": self.model_id,
            "artifact_kind": "generated_world",
            "artifact_path": resolved_artifact,
            "artifact_files": generated_files,
            "runtime": "joyai_echo.official_external_runtime",
            "backend_quality": "official_external_runtime",
            "request": request,
            "preflight": preflight,
            "runtime_result": run_result,
            "metadata_path": str(report_path),
        }
        if result["status"] != "success":
            result["error"] = run_result.get("error") or "Echo-WM execution failed."
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        self.history.append(result)
        if return_dict or result["status"] != "success":
            return result
        return resolved_artifact

    def stream(self, *args: Any, **kwargs: Any) -> dict[str, Any] | str:
        """Execute one upstream rollout; online streaming is not released yet."""
        return self(*args, **kwargs)


__all__ = ["JoyAIEchoLongVideoPipeline", "JoyAIEchoWMPipeline"]
