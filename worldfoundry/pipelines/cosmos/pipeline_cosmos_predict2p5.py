"""Public Cosmos Predict 2.5 pipeline backed by native diffusion recipes."""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.optimizations import (
    RuntimePolicy,
    parse_offload_policy,
    parse_torch_dtype,
)
from worldfoundry.core.media.codecs.video import save_image_or_video_tensor

from ..pipeline_utils import PipelineABC
from .parallel_options import agree_cosmos_request, build_cosmos_parallel_context
from .reason1_options import reason1_component_options

COSMOS_PREDICT2P5_DEFAULT_FPS = 16
COSMOS_PREDICT2P5_DEFAULT_NUM_FRAMES = 93
COSMOS_PREDICT2P5_DEFAULT_NUM_INFERENCE_STEPS = 35
DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
    "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)


def _model_id(requested: str | None, source: object) -> str:
    value = str(requested or "").lower()
    if "14b" in value or "14b" in str(source).lower():
        return "cosmos-predict2.5-14b"
    if value and value not in {
        "cosmos-predict-2.5",
        "cosmos-predict-2.5-2b",
        "cosmos-predict-2p5",
        "cosmos-predict2.5",
        "cosmos-predict2.5-2b",
        "cosmos-predict2p5",
        "nvidia/cosmos-predict2.5-2b",
        "2b",
    }:
        raise ValueError(f"unsupported Cosmos Predict 2.5 selector: {requested!r}")
    return "cosmos-predict2.5-2b"


class CosmosPredict2p5Pipeline(PipelineABC):
    """Studio-facing wrapper around the canonical 2B/14B native recipes."""

    MODEL_ID = "cosmos-predict2.5"

    def __init__(
        self, *, native_pipeline: NativeDiffusionPipeline, device: str, model_id: str, parallel_context=None,
    ) -> None:
        super().__init__(model_id=model_id, synthesis_model=None, device=device)
        self.native_pipeline = native_pipeline
        self.parallel_context = parallel_context

    def close(self) -> None:
        """Release owned inference subgroups; keep the caller-owned world live."""

        if self.parallel_context is not None:
            self.parallel_context.close()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "CosmosPredict2p5Pipeline":
        """Load native components with optional Reason1 placement and reuse.

        ``text_embedding_cache_size`` counts complete prompt batches (default
        zero); ``text_embedding_cache_max_bytes`` bounds host tensor storage
        (default 512 MiB). ``text_encoder_run_on_cpu=True`` keeps Reason1
        weights and computation on CPU while returning GPU conditioning.
        ``tensor_parallel`` and ``context_parallel`` default to one. Multi-rank
        inference requires an initialized world, explicit per-rank CUDA devices
        and ``offload_mode=none``, ``torch_dtype=float32``. Low-precision model
        parallelism is rejected because its generation drift failed validation.
        All ranks load and call together; each returns
        full tensors, and only rank zero writes video artifacts. Call ``close``
        together before destroying the caller-owned world. Parallel kernels can
        differ numerically from single-rank inference.
        """

        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        encoder_options = reason1_component_options(options)
        source = (
            options.get("checkpoint_path")
            or options.get("transformer_model_path")
            or options.get("model_path")
            or model_path
        )
        resolved_model_id = _model_id(str(options.get("variant_id") or model_id or ""), source)
        overrides: dict[str, str] = {}
        if isinstance(source, (str, Path)):
            checkpoint = Path(source).expanduser()
            if checkpoint.is_dir():
                root = checkpoint.resolve()
                overrides["transformer"] = str(root)
                if (root / "tokenizer.pth").is_file():
                    overrides["vae"] = str(root)
            elif checkpoint.is_file():
                overrides["transformer"] = str(checkpoint.resolve())
        vae_source = options.get("vae_model_path")
        if isinstance(vae_source, (str, Path)) and Path(vae_source).expanduser().exists():
            overrides["vae"] = str(Path(vae_source).expanduser().resolve())
        text_source = options.get("text_encoder_model_path")
        if isinstance(text_source, (str, Path)) and Path(text_source).expanduser().is_dir():
            root = str(Path(text_source).expanduser().resolve())
            overrides.update({"text-encoder": root, "tokenizer": root})

        policy = RuntimePolicy(
            device=torch.device(device),
            dtype=parse_torch_dtype(
                options.get("torch_dtype", options.get("weight_dtype", options.get("dtype"))),
                owner="Cosmos2.5",
            ),
            offload=parse_offload_policy(options.get("offload_mode", "block"), owner="Cosmos2.5"),
        )
        parallel = build_cosmos_parallel_context(
            options, policy, attention_heads=40 if resolved_model_id.endswith("14b") else 16,
        )
        component_options = {
            "conditioner:main": encoder_options,
            "latent_initializer:main": {
                "tiled": bool(options.get("vae_tiling", False)),
                "tile_size": tuple(options.get("vae_tile_size", (34, 34))),
                "tile_stride": tuple(options.get("vae_tile_stride", (18, 16))),
            }
        }
        if parallel is not None:
            component_options["denoiser:main"] = {
                "tensor_parallel": parallel.tp_size, "context_parallel": parallel.cp_size,
                "parallel_context": parallel,
            }
        try:
            if parallel is not None:
                parallel.agree(("cosmos25-pipeline", resolved_model_id, overrides, str(policy.dtype),
                                encoder_options, component_options["latent_initializer:main"]))
            native = NativeDiffusionPipeline.from_pretrained(
                resolved_model_id, policy=policy, checkpoint_overrides=overrides or None,
                extensions=options.get("extensions", ()), component_options=component_options,
            )
        except BaseException:
            if parallel is not None:
                try:
                    parallel.close()
                except Exception:
                    pass
            raise
        return cls(native_pipeline=native, device=device, model_id=resolved_model_id, parallel_context=parallel)

    @classmethod
    def plan(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        source = (
            options.get("checkpoint_path")
            or options.get("transformer_model_path")
            or options.get("model_path")
            or model_path
        )
        return {
            "model_id": _model_id(str(options.get("variant_id") or options.get("model_id") or ""), source),
            "checkpoint": str(source) if source is not None else None,
            "backend": "worldfoundry-native-diffusion",
            "native_inference": True,
            "blocked": False,
        }

    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        images: Any = None,
        image: Any = None,
        image_path: str | None = None,
        input_path: str | None = None,
        video: Any = None,
        video_path: str | None = None,
        output_path: str | None = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = COSMOS_PREDICT2P5_DEFAULT_NUM_INFERENCE_STEPS,
        fps: int = COSMOS_PREDICT2P5_DEFAULT_FPS,
        num_frames: int = COSMOS_PREDICT2P5_DEFAULT_NUM_FRAMES,
        height: int = 704,
        width: int = 1280,
        seed: int = 0,
        num_latent_conditional_frames: int = 1,
        conditional_frame_timestep: float | None = None,
        output_type: str = "video",
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        image_value = image if image is not None else images
        image_value = image_value if image_value is not None else image_path
        video_value = video if video is not None else video_path
        if image_value is None and video_value is None and input_path is not None:
            if Path(input_path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                video_value = input_path
            else:
                image_value = input_path
        inputs: dict[str, object] = {
            "fps": int(fps),
            "return_latent": output_type == "latent",
            "num_latent_conditional_frames": int(num_latent_conditional_frames),
        }
        if conditional_frame_timestep is not None:
            inputs["conditional_frame_timestep"] = float(conditional_frame_timestep)
        if image_value is not None:
            inputs["image"] = image_value
        if video_value is not None:
            inputs["video"] = video_value
        if self.parallel_context is not None:
            agree_cosmos_request(
                self.parallel_context, prompt=prompt, negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
                inputs=inputs, seed=int(seed), height=int(height), width=int(width), num_frames=int(num_frames),
                steps=int(num_inference_steps), guidance_scale=float(guidance_scale), output_type=output_type,
            )
            actual_seed = self.parallel_context.broadcast_seed(int(seed))
        else:
            actual_seed = secrets.randbits(63) if int(seed) < 0 else int(seed)
        request = DiffusionRequest(
            prompt=prompt,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            sampling=SamplingConfig(
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                seed=actual_seed,
            ),
            inputs=inputs,
        )
        output = self.native_pipeline(request)
        artifact_path = None
        if (output_path is not None and output_type != "latent"
                and (self.parallel_context is None or self.parallel_context.is_main)):
            artifact_path = save_image_or_video_tensor(output.sample, output_path, fps=int(fps))
        result = {
            "video": output.sample,
            "latents": output.latents,
            "artifact_path": artifact_path,
            "metadata": dict(output.metadata),
        }
        return result if return_dict else (artifact_path or output.sample)


__all__ = ["CosmosPredict2p5Pipeline"]
