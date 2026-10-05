"""Longcat Video visual generation pipeline module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.longcat_video.worldfoundry_runtime import (
    LongCatVideoRuntime,
)


_DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


def _coerce_path(target: Path | str | None, *, fallback_name: str) -> Path:
    """Coerce path helper function."""
    if target is None:
        return Path.cwd() / fallback_name
    return Path(target).expanduser().resolve()


def _coerce_report_path(output_path: Path | str | None) -> Path:
    """Coerce report path helper function."""
    report_path = _coerce_path(output_path, fallback_name="longcat_video_result.json")
    if report_path.suffix.lower() in {".mp4", ".mov", ".webm", ".gif"}:
        return report_path.with_suffix(".json")
    return report_path


def _coerce_report_artifact_path(
    output_path: Path | str | None,
    *,
    artifact_suffix: str,
) -> Path:
    """Coerce report artifact path helper function."""
    artifact_path = _coerce_path(output_path, fallback_name=f"longcat_video.{artifact_suffix}")
    if artifact_path.suffix.lower() in {".mp4", ".mov", ".webm", ".gif"}:
        return artifact_path
    return artifact_path.with_suffix(f".{artifact_suffix}")


class LongCatVideoPipeline(PipelineABC):
    """WorldFoundry adapter for LongCat-Video."""

    MODEL_ID = "longcat-video"

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        task_type: str = "t2v",
        device: str = "cuda",
        runtime: LongCatVideoRuntime | None = None,
        loaded: bool = False,
        load_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir is not None else None
        self.task_type = task_type
        self.device = device
        self.runtime = runtime or LongCatVideoRuntime(
            checkpoint_dir=self.checkpoint_dir,
            task_type=task_type,
            device=device,
        )
        self.loaded = loaded
        self.load_options = dict(load_options or {})
        self.history: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "LongCatVideoPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        del model_id
        options = cls._normalize_options(model_path, required_components, kwargs)
        checkpoint_dir = options.get("checkpoint_dir") or options.get("ckpt_dir") or options.get("model_path")
        if checkpoint_dir is None and model_path is not None and not isinstance(model_path, dict):
            checkpoint_dir = model_path
        task_type = str(options.get("task_type") or "t2v")
        runtime_options = dict(options)
        for key in ("checkpoint_dir", "ckpt_dir", "task_type"):
            runtime_options.pop(key, None)
        runtime = LongCatVideoRuntime.from_pretrained(
            checkpoint_dir=checkpoint_dir,
            task_type=task_type,
            device=device,
            **runtime_options,
        )
        return cls(
            checkpoint_dir=checkpoint_dir,
            task_type=task_type,
            device=device,
            runtime=runtime,
            loaded=runtime.loaded,
            load_options=options,
        )

    @staticmethod
    def _normalize_options(
        model_path: Any,
        required_components: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize options for LongCatVideoPipeline."""
        options: dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            options["checkpoint_dir"] = model_path
        options.update(required_components or {})
        options.update(kwargs)
        return options

    def preflight(self) -> dict[str, Any]:
        """Preflight for LongCatVideoPipeline."""
        return self.runtime.preflight()

    def process(
        self,
        prompt: str,
        images: Any = None,
        video: Any = None,
        task_type: str = "t2v",
        height: int = 480,
        width: int = 832,
        resolution: str = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int | None = None,
        use_distill: bool = False,
        negative_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        request = {
            "prompt": prompt,
            "images": images,
            "video": video,
            "task_type": task_type,
            "height": height,
            "width": width,
            "resolution": resolution,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "use_distill": use_distill,
            "negative_prompt": negative_prompt or _DEFAULT_NEGATIVE_PROMPT,
            "extra": kwargs,
        }
        if task_type == "t2v":
            # The upstream demo runs every stage by default. A pipeline request
            # selects base or distilled inference; refinement is explicitly opt-in.
            request.update(
                run_base=not use_distill,
                run_distill=use_distill,
                run_refiner=False,
            )
            for key in (
                "run_base", "run_distill", "run_refiner", "fps", "cpu_offload",
                "max_sequence_length", "distill_num_inference_steps", "spatial_refine_only",
            ):
                if key in kwargs:
                    request[key] = kwargs[key]
        return request

    def _record(self, payload: dict[str, Any]) -> None:
        """Record for LongCatVideoPipeline."""
        self.history.append(payload)

    def _build_success_payload(
        self,
        request: dict[str, Any],
        *,
        artifact_path: str,
        run_result: dict[str, Any],
        output_path: Path,
        output_files: list[str],
    ) -> dict[str, Any]:
        """Build success payload for LongCatVideoPipeline."""
        return {
            "status": run_result.get("status", "failed"),
            "model_id": self.MODEL_ID,
            "artifact_kind": "generated_video",
            "runtime": "worldfoundry.synthesis.visual_generation.longcat_video.worldfoundry_runtime",
            "backend_quality": "in_tree_longcat_runtime",
            "artifact_path": artifact_path,
            "artifact_files": output_files,
            "request": request,
            "preflight": self.runtime.preflight(),
            "runtime_plan": run_result.get("runtime_plan"),
            "runtime_result": run_result,
            "metadata_path": str(output_path),
        }

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        video: Any = None,
        task_type: str | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        height: int = 480,
        width: int = 832,
        resolution: str = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        use_distill: bool = False,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        execute: bool = False,
        timeout_seconds: int = 3600,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute the complete pipeline generation flow."""
        current_task = task_type or self.task_type
        preflight = self.runtime.preflight()
        request = self.process(
            prompt=prompt,
            images=images,
            video=video,
            task_type=current_task,
            height=height,
            width=width,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            use_distill=use_distill,
            negative_prompt=negative_prompt,
            **kwargs,
        )
        request["model_id"] = self.MODEL_ID
        if execute:
            report_path = _coerce_report_path(output_path)
            artifact_output_dir = _coerce_path(output_dir, fallback_name="longcat_outputs")
            artifact_output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = _coerce_report_artifact_path(
                output_path=output_path,
                artifact_suffix="mp4",
            )
            runtime_plan = self.runtime.build_plan(
                request=request,
                task_type=current_task,
                output_dir=artifact_output_dir,
                output_path=artifact_path,
                context_parallel_size=int(kwargs.pop("context_parallel_size", 1)),
                enable_compile=bool(kwargs.pop("enable_compile", False)),
            )
            run_result = self.runtime.run_plan(
                runtime_plan,
                timeout_seconds=timeout_seconds,
                log_dir=report_path.parent,
            )
            generated_files = [item for item in run_result.get("generated_files", []) if isinstance(item, str)]
            resolved_artifact = artifact_path
            if generated_files:
                resolved_artifact = Path(generated_files[0])
            result = self._build_success_payload(
                request=request,
                artifact_path=str(resolved_artifact),
                run_result=run_result,
                output_path=report_path,
                output_files=generated_files,
            )
            if run_result.get("status") == "failed":
                result["status"] = "failed"
                result["error"] = run_result.get("error") or run_result.get("blocked_reason") or "LongCat-Video execute failed."
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self._record(result)
            if return_dict:
                return result
            if result["status"] == "success":
                return str(resolved_artifact)
            return result.get("artifact_path") or result

        del preflight
        raise RuntimeError("LongCat-Video requires execute=True; preflight artifacts are no longer emitted.")

    def stream(
        self,
        prompt: str,
        images: Any = None,
        video: Any = None,
        task_type: str | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        height: int = 480,
        width: int = 832,
        resolution: str = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        use_distill: bool = False,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        execute: bool = False,
        timeout_seconds: int = 3600,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Stream visual generation outputs chunk by chunk."""
        return self(
            prompt=prompt,
            images=images,
            video=video,
            task_type=task_type,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_distill=use_distill,
            output_path=output_path,
            output_dir=output_dir,
            execute=execute,
            timeout_seconds=timeout_seconds,
            return_dict=return_dict,
            **kwargs,
        )


__all__ = ["LongCatVideoPipeline"]
