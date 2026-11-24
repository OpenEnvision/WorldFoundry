"""Control-video generation through the canonical in-process LTX IC-LoRA recipes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from worldfoundry.core.io import artifact_root_path
from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.runtime.assets import expand_worldfoundry_path

if TYPE_CHECKING:
    from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
    from worldfoundry.core.contracts import PipelineInvocation


def _required_file(value: str | Path, label: str) -> Path:
    path = expand_worldfoundry_path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LTX IC-LoRA {label} does not exist: {path}")
    return path


def _required_dir(value: str | Path, label: str) -> Path:
    path = expand_worldfoundry_path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"LTX IC-LoRA {label} does not exist: {path}")
    if not (path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"LTX IC-LoRA {label} requires tokenizer.json: {path}")
    return path


class _LTXICLoRAV2VPipeline(PipelineABC):
    """Encode typed control video and run the native adapted/base two-stage stack."""

    CHECKPOINT_DIR = ""
    CHECKPOINT_FILE = ""
    UPSAMPLER_FILE = ""
    IC_LORA_DIR = ""
    IC_LORA_FILE = ""
    CONTROL_TYPES: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        spatial_upsampler_path: str | Path,
        gemma_root: str | Path,
        ic_lora_path: str | Path,
        device: str = "cuda",
    ) -> None:
        super().__init__(model_id=self.MODEL_ID, device=device)
        self.checkpoint_path = _required_file(checkpoint_path, "distilled checkpoint")
        self.spatial_upsampler_path = _required_file(spatial_upsampler_path, "spatial upsampler")
        self.gemma_root = _required_dir(gemma_root, "Gemma root")
        self.ic_lora_path = _required_file(ic_lora_path, "control IC-LoRA")

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]  # model-specific loader arguments, passed by keyword
        cls,
        model_path: str | Path | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> _LTXICLoRAV2VPipeline:
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(dict(required_components or {}))
        options.update(kwargs)
        cls._strip_framework_loading_options(options)
        options.pop("lazy", None)
        root = "$WORLDFOUNDRY_CKPT_DIR"
        if model_path is not None and not isinstance(model_path, Mapping):
            supplied = expand_worldfoundry_path(model_path)
            options.setdefault("checkpoint_path", supplied / cls.CHECKPOINT_FILE if supplied.is_dir() else supplied)
        options.setdefault("checkpoint_path", f"{root}/{cls.CHECKPOINT_DIR}/{cls.CHECKPOINT_FILE}")
        options.setdefault("spatial_upsampler_path", f"{root}/{cls.CHECKPOINT_DIR}/{cls.UPSAMPLER_FILE}")
        options.setdefault("ic_lora_path", f"{root}/{cls.IC_LORA_DIR}/{cls.IC_LORA_FILE}")
        options.setdefault("gemma_root", f"{root}/gemma-3-12b-it-qat-q4_0-unquantized")
        accepted = {"checkpoint_path", "spatial_upsampler_path", "ic_lora_path", "gemma_root"}
        unknown = set(options) - accepted
        if unknown:
            raise ValueError(f"unsupported LTX IC-LoRA loading options: {sorted(unknown)}")
        return cls(device=device, **options)

    def __call__(
        self,
        prompt: str,
        video: str | Path | None = None,
        *,
        video_path: str | Path | None = None,
        control_type: str | None = None,
        images: Any = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        fps: int = 24,
        seed: int = 171198,
        control_strength: float = 1.0,
        lora_strength: float = 1.0,
        offload: str = "cpu",
        skip_stage_2: bool = False,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        if images is not None:
            raise ValueError("LTX IC-LoRA V2V does not accept images")
        if kwargs:
            raise ValueError(f"unsupported LTX IC-LoRA generation options: {sorted(kwargs)}")
        if control_type not in self.CONTROL_TYPES:
            raise ValueError(f"control_type must be one of {sorted(self.CONTROL_TYPES)}")
        if video is not None and video_path is not None:
            raise ValueError("pass either video or video_path")
        control = video if video is not None else video_path
        reference = _required_file(control, "control video") if control is not None else None
        if reference is None:
            raise ValueError("LTX IC-LoRA V2V requires a control video")
        if height % 64 or width % 64 or height < 128 or width < 128:
            raise ValueError("height and width must be at least 128 and divisible by 64")
        if num_frames < 9 or (num_frames - 1) % 8:
            raise ValueError("num_frames must be 8n+1 and at least 9")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 0.0 <= control_strength <= 1.0:
            raise ValueError("control_strength must be in [0, 1]")
        if lora_strength <= 0:
            raise ValueError("lora_strength must be positive")
        if offload not in {"none", "cpu", "disk"}:
            raise ValueError("offload must be none, cpu, or disk")

        target = Path(output_path) if output_path else artifact_root_path() / "pipeline_eval" / f"{self.MODEL_ID}.mp4"
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        log_path = target.with_suffix(".ltx.log")
        from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
        from worldfoundry.core.media.codecs.pyav import save_audio_video_h264

        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"{self.MODEL_ID}: worldfoundry.native_diffusion IC-LoRA\n")
            try:
                pipeline = self._build_native_pipeline(
                    lora_strength=lora_strength, offload=offload, skip_stage_2=skip_stage_2
                )
                output = pipeline(
                    DiffusionRequest(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        sampling=SamplingConfig(
                            num_inference_steps=8 if skip_stage_2 else 11, seed=seed, guidance_scale=1.0
                        ),
                        inputs={"video": str(reference), "control_strength": control_strength, "frame_rate": fps},
                    )
                )
                save_audio_video_h264(
                    output.sample,
                    target,
                    fps=fps,
                    waveform=output.artifacts["audio"],
                    sample_rate=int(output.artifacts["audio_sampling_rate"]),
                )
            except Exception as error:
                log.write(f"failed: {type(error).__name__}: {error}\n")
                raise RuntimeError(f"{self.MODEL_ID} IC-LoRA generation failed: {error}") from error
            log.write(f"success: {target}\n")
        record = {"status": "success", "artifact_path": str(target), "video": str(target), "log_path": str(log_path)}
        return record if return_dict else str(target)

    def _build_native_pipeline(
        self, *, lora_strength: float, offload: str, skip_stage_2: bool
    ) -> NativeDiffusionPipeline:
        import torch

        from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
        from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
        from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
        from worldfoundry.base_models.diffusion_model.optimizations import (
            OffloadMode,
            OffloadPolicy,
            RuntimePolicy,
            parse_offload_policy,
        )
        from worldfoundry.base_models.diffusion_model.recipes.ltx import ltx2_v2v_recipe, ltx23_v2v_recipe

        factory = ltx2_v2v_recipe if self.MODEL_ID == "ltx-2-v2v" else ltx23_v2v_recipe
        policy = parse_offload_policy(offload, owner="LTX IC-LoRA")
        if policy.mode is OffloadMode.BLOCK and torch.device(self.device).type == "mps":
            policy = OffloadPolicy(mode=OffloadMode.COMPONENT, target="cpu")
        checkpoints = {
            "model": self.checkpoint_path,
            "gemma": self.gemma_root,
            "ic_lora": self.ic_lora_path,
        }
        if not skip_stage_2:
            checkpoints["upsampler"] = self.spatial_upsampler_path
        runner = NativeDiffusionAssembler().build(
            factory(skip_stage_2=skip_stage_2),
            policy=RuntimePolicy(device=self.device, dtype=torch.bfloat16, offload=policy),
            checkpoint_overrides={name: CheckpointSpec(source=str(path)) for name, path in checkpoints.items()},
            component_options={"denoiser:main": {"lora_strength": lora_strength}},
        )
        return NativeDiffusionPipeline(runner)

    def run_pipeline_invocation(self, invocation: PipelineInvocation) -> Mapping[str, Any]:
        if invocation.image is not None:
            raise ValueError("LTX IC-LoRA V2V does not accept images")
        options = dict(invocation.pipeline_kwargs)
        options.pop("operator_kwargs", None)
        return self(
            prompt=invocation.prompt,
            video=invocation.video,
            output_path=invocation.output_path,
            return_dict=True,
            **options,
        )


class LTX2V2VPipeline(_LTXICLoRAV2VPipeline):
    MODEL_ID = "ltx-2-v2v"
    CHECKPOINT_DIR = "Lightricks--LTX-2"
    CHECKPOINT_FILE = "ltx-2-19b-distilled.safetensors"
    UPSAMPLER_FILE = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
    IC_LORA_DIR = "Lightricks--LTX-2-19b-IC-LoRA-Pose-Control"
    IC_LORA_FILE = "ltx-2-19b-ic-lora-pose-control.safetensors"
    CONTROL_TYPES = frozenset({"pose"})


class LTX23V2VPipeline(_LTXICLoRAV2VPipeline):
    MODEL_ID = "ltx-2.3-v2v"
    CHECKPOINT_DIR = "Lightricks--LTX-2.3"
    CHECKPOINT_FILE = "ltx-2.3-22b-distilled-1.1.safetensors"
    UPSAMPLER_FILE = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    IC_LORA_DIR = "Lightricks--LTX-2.3-22b-IC-LoRA-Union-Control"
    IC_LORA_FILE = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
    CONTROL_TYPES = frozenset({"canny", "depth", "pose"})


__all__ = ["LTX2V2VPipeline", "LTX23V2VPipeline"]
