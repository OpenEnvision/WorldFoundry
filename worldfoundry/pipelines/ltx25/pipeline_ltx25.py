"""WorldFoundry contract for the in-process native LTX-2.5 recipe."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from worldfoundry.core.execution.deadline import check_deadline
from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.ltx25.runtime import LTX25DistilledRuntime

if TYPE_CHECKING:
    from worldfoundry.synthesis.visual_generation.ltx25.acceleration import LTX25AccelerationConfig


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
        "acquisition_root",
    ):
        options.pop(key, None)


def _output_paths(output_path: str | Path | None) -> tuple[Path, Path]:
    requested = Path(output_path or "ltx-2.5.mp4").expanduser().resolve()
    if requested.suffix.lower() == ".mp4":
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
        *preflight.get("missing_system_tools", []),
        *preflight.get("blocked_reasons", []),
    ]
    details = ", ".join(str(item) for item in missing) or "unknown runtime requirement"
    return f"LTX-2.5 preflight failed; missing: {details}"


class LTX25Pipeline(PipelineABC):
    """Text/image-to-audio-video adapter for the native distilled LTX-2.5 recipe.

    Execution stays in-process in the active ``ltx-2.5`` environment.
    ``timeout_seconds`` is a cooperative deadline checked between native
    stages/steps and before artifact writes; it cannot preempt a running
    CUDA kernel.
    """

    MODEL_ID = "ltx-2.5"

    def __init__(self, runtime: LTX25DistilledRuntime, *, model_id: str = MODEL_ID) -> None:
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
    ) -> LTX25Pipeline:
        options = _runtime_options(model_path, required_components, kwargs)
        resolved_model_id = str(options.pop("model_id", model_id or cls.MODEL_ID))
        _pop_runner_metadata(options)
        hf_models_root = options.pop("hf_models_root", None)
        checkpoint_dir = options.pop(
            "checkpoint_dir",
            options.pop(
                "model_dir",
                options.pop("model_path", options.pop("pretrained_model_path", None)),
            ),
        )
        if checkpoint_dir is None and hf_models_root is not None:
            checkpoint_dir = Path(hf_models_root).expanduser() / "Lightricks--LTX-2.5"
        runtime = LTX25DistilledRuntime(
            checkpoint_dir=checkpoint_dir,
            transformer_path=options.pop("transformer_path", None),
            text_encoder_path=options.pop("text_encoder_path", None),
            video_vae_path=options.pop("video_vae_path", None),
            audio_vae_path=options.pop("audio_vae_path", None),
            spatial_upsampler_path=options.pop("spatial_upsampler_path", None),
            duration_head_path=options.pop("duration_head_path", None),
            prompt_enhancer_gemma_root=options.pop("prompt_enhancer_gemma_root", None),
            offload_mode=str(options.pop("offload_mode", "cpu")),
            quantization=options.pop("quantization", None),
            diffvae_optimization=str(options.pop("diffvae_optimization", "chunked_eager")),
            device=device,
            model_variant=options.pop("model_variant", "distilled"),
            video_vae_variant=options.pop("video_vae_variant", "auto"),
            acceleration=options.pop("acceleration", None),
        )
        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"Unknown LTX-2.5 load options: {unknown}")
        return cls(runtime, model_id=resolved_model_id)

    def configure_acceleration(
        self,
        acceleration: LTX25AccelerationConfig | Mapping[str, Any] | str | None,
    ) -> LTX25AccelerationConfig | None:
        """Configure SOL-Engine acceleration on the underlying LTX-2.5 runtime."""
        return self.runtime.configure_acceleration(acceleration)

    def acceleration_report(self) -> dict[str, Any]:
        """Return auditable SOL-Engine receipts for the transformer and VideoVAE."""
        return self.runtime.acceleration_report()

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
        del video, interactions
        request = dict(operator_kwargs or {})
        request.update(kwargs)
        request.pop("task_name", None)
        request.setdefault("prompt", prompt)
        request.setdefault("images", images)
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
            raise RuntimeError("LTX-2.5 requires execute=True; dry-run artifacts are not emitted.")
        deadline = time.monotonic() + timeout_seconds
        request = self.process(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            operator_kwargs=operator_kwargs,
            **kwargs,
        )
        artifact_path, report_path = _output_paths(output_path)
        preflight = self.preflight()
        if preflight["status"] != "ready":
            raise RuntimeError(_preflight_error(preflight))
        plan = self.runtime.build_plan(request=request, output_path=artifact_path)
        run_result = self.runtime.run_plan(plan, timeout_seconds=deadline - time.monotonic())
        generated_files = list(run_result.get("generated_files") or [])
        resolved_artifact = generated_files[0] if generated_files else str(artifact_path)
        result = {
            "status": run_result.get("status", "failed"),
            "model_id": self.model_id,
            "task": plan.task,
            "artifact_kind": "generated_video",
            "artifact_path": resolved_artifact,
            "artifact_files": generated_files,
            "runtime": "worldfoundry.native_diffusion",
            "backend_quality": "native",
            "request": request,
            "preflight": preflight,
            "runtime_result": run_result,
            "metadata_path": str(report_path),
        }
        if result["status"] != "success":
            result["error"] = run_result.get("error") or "LTX-2.5 execution failed."
        check_deadline(deadline, stage="LTX-2.5 metadata artifact write")
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        self.history.append(result)
        if return_dict or result["status"] != "success":
            return result
        return resolved_artifact

    def stream(self, *args: Any, **kwargs: Any) -> dict[str, Any] | str:
        """Return one completed batch artifact through the native recipe."""
        return self(*args, **kwargs)


__all__ = ["LTX25Pipeline"]
