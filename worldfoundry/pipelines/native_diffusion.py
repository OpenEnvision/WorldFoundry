"""Shared public adapter for native image and video diffusion recipes."""

from __future__ import annotations

import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.optimizations import (
    RuntimePolicy,
    parse_attention_backend,
    parse_offload_policy,
    parse_quantization_policy,
    parse_torch_dtype,
)
from worldfoundry.core.media.codecs.video import save_image_or_video_tensor
from worldfoundry.operators.runtime_video_operator import RuntimeVideoOperator
from worldfoundry.synthesis.visual_generation.memory.video import VideoArtifactMemory

from .pipeline_utils import PipelineABC


class NativeVisualDiffusionPipeline(PipelineABC):
    """Thin product boundary shared by every native visual diffusion recipe.

    Model packages own checkpoint-compatible networks.  This adapter owns the
    common product-facing request, artifact, and memory behavior, so image and
    video families do not grow parallel pipeline implementations.
    """

    MODEL_ID: ClassVar[str] = ""
    OWNER: ClassVar[str] = "native diffusion"
    CHECKPOINT_ROLES: ClassVar[tuple[str, ...]] = ()
    PRIMARY_CHECKPOINT_ROLE: ClassVar[str | None] = None
    ALLOW_MODEL_ID_OVERRIDE: ClassVar[bool] = False
    PEFT_ADAPTER_COMPONENT: ClassVar[str | None] = None
    GENERATION_TYPE: ClassVar[str] = "t2v"
    ACCEPTS_IMAGES: ClassVar[bool] = False
    REQUIRES_IMAGES: ClassVar[bool] = False
    ACCEPTS_VIDEO: ClassVar[bool] = False
    ACCEPTS_INTERACTIONS: ClassVar[bool] = False
    DEFAULT_HEIGHT: ClassVar[int] = 512
    DEFAULT_WIDTH: ClassVar[int] = 512
    DEFAULT_NUM_FRAMES: ClassVar[int] = 1
    DEFAULT_NUM_INFERENCE_STEPS: ClassVar[int] = 50
    DEFAULT_GUIDANCE_SCALE: ClassVar[float] = 7.5
    DEFAULT_NEGATIVE_PROMPT: ClassVar[str | None] = None
    DEFAULT_FPS: ClassVar[int] = 24
    OUTPUT_VALUE_RANGE: ClassVar[str | tuple[float, float]] = "auto"
    DEFAULT_SCHEDULER_OPTIONS: ClassVar[Mapping[str, object]] = {}
    SCHEDULER_OPTION_ALIASES: ClassVar[Mapping[str, str]] = {}
    REQUEST_INPUT_DEFAULTS: ClassVar[Mapping[str, object]] = {}
    REQUEST_INPUT_ALIASES: ClassVar[Mapping[str, str]] = {}
    NUM_FRAMES_ALIASES: ClassVar[tuple[str, ...]] = ("frame_num", "frames")
    NUM_INFERENCE_STEP_ALIASES: ClassVar[tuple[str, ...]] = ("infer_steps",)
    GUIDANCE_SCALE_ALIASES: ClassVar[tuple[str, ...]] = ("cfg_scale",)
    RUNTIME_POLICY_OPTION_KEYS: ClassVar[tuple[str, ...]] = (
        "adacache",
        "approximate_attention",
        "blocktaylorseer",
        "compile_backend",
        "compile_dynamic",
        "compile_fullgraph",
        "compile_mode",
        "compile_options",
        "cfg_gate_fraction",
        "cfg_gate_step",
        "cfg_parallel",
        "cfg_parallel_degree",
        "cuda_graph",
        "custom",
        "device_map",
        "dit_weight_dtype",
        "feature_cache",
        "fused_rope",
        "fuse_qkv",
        "qkv_strategy",
        "qkv_split_threshold",
        "inplace_residual",
        "magcache",
        "rope_precision",
        "rms_norm_precision",
        "sequence_parallel",
        "sp_degree",
        "static_cross_kv",
        "taylorseer",
        "teacache",
        "teacache_thresh",
        "vae_decode_autocast",
        "vae_channels_last_3d",
        "vae_parallel",
        "vae_parallel_degree",
        "vae_preview_decoder_path",
        "vae_spatial_tiling",
        "vae_temporal_chunk_size",
        "vae_tile_size",
        "vae_tile_stride",
        "vae_tiled_decode",
        "vae_weight_dtype",
    )

    def __init__(
        self,
        *,
        native_pipeline: NativeDiffusionPipeline,
        device: str,
        model_id: str | None = None,
    ) -> None:
        requested_model_id = str(model_id or self.MODEL_ID).strip()
        if not requested_model_id:
            raise ValueError("native visual pipeline subclasses must declare a model ID")
        super().__init__(
            model_id=requested_model_id,
            operator=RuntimeVideoOperator(generation_type=self.GENERATION_TYPE),
            memory_module=VideoArtifactMemory(model_id=requested_model_id),
            device=device,
        )
        self.native_pipeline = native_pipeline
        self.synthesis_model = native_pipeline
        self.generation_type = self.GENERATION_TYPE
        self.model_id = native_pipeline.model_id
        self.model_name = native_pipeline.model_id

    @classmethod
    def _options(
        cls,
        model_path: str | Mapping[str, Any] | None,
        required_components: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        return options

    @classmethod
    def _checkpoint_overrides(
        cls,
        model_path: str | Mapping[str, Any] | None,
        options: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        explicit = options.get("checkpoint_overrides")
        if isinstance(explicit, Mapping):
            return {str(name): value for name, value in explicit.items()}
        source = options.get(
            "checkpoint_path",
            options.get(
                "checkpoint_dir",
                options.get("pretrained_model_path", options.get("model_path", model_path)),
            ),
        )
        if not isinstance(source, (str, Path)):
            return None
        value = str(source)
        root = value if "://" in value else str(Path(value).expanduser())
        if cls.PRIMARY_CHECKPOINT_ROLE is not None:
            if cls.PRIMARY_CHECKPOINT_ROLE not in cls.CHECKPOINT_ROLES:
                raise ValueError(
                    f"{cls.__name__}.PRIMARY_CHECKPOINT_ROLE must be one of "
                    f"{cls.CHECKPOINT_ROLES}, got {cls.PRIMARY_CHECKPOINT_ROLE!r}"
                )
            return {cls.PRIMARY_CHECKPOINT_ROLE: root}
        return {name: root for name in cls.CHECKPOINT_ROLES} or None

    @classmethod
    def _requested_model_id(cls, options: Mapping[str, Any]) -> str:
        if cls.ALLOW_MODEL_ID_OVERRIDE:
            value = options.get("variant", options.get("variant_id"))
            if value is None:
                value = options.get("model_id", options.get("profile_id"))
            if value is not None:
                return str(value)
        return cls.MODEL_ID

    @classmethod
    def _component_options(
        cls,
        options: Mapping[str, Any],
    ) -> dict[str, dict[str, object]] | None:
        configured = options.get("component_options")
        if configured is None:
            resolved: dict[str, dict[str, object]] = {}
        elif isinstance(configured, Mapping):
            resolved = {}
            for raw_name, raw_options in configured.items():
                name = str(raw_name)
                if not isinstance(raw_options, Mapping):
                    raise TypeError(f"component options for {name!r} must be a mapping")
                resolved[name] = dict(raw_options)
        else:
            raise TypeError("component_options must be a mapping when provided")

        adapter_path = options.get("peft_adapter_path")
        if adapter_path is None:
            return resolved or None
        if cls.PEFT_ADAPTER_COMPONENT is None:
            raise ValueError(f"{cls.__name__} does not support PEFT adapter loading")
        if not isinstance(adapter_path, (str, Path)):
            raise TypeError("peft_adapter_path must be a local filesystem path")
        if isinstance(adapter_path, str) and not adapter_path.strip():
            raise ValueError("peft_adapter_path must not be empty")
        normalized_path = str(Path(adapter_path).expanduser())
        component_options = dict(resolved.get(cls.PEFT_ADAPTER_COMPONENT, {}))
        if "peft_adapter_path" in component_options:
            raise ValueError(
                "peft_adapter_path must be configured once at the native pipeline boundary"
            )
        component_options["peft_adapter_path"] = normalized_path
        resolved[cls.PEFT_ADAPTER_COMPONENT] = component_options
        return resolved

    @classmethod
    def _runtime_policy_options(cls, options: Mapping[str, Any]) -> dict[str, object]:
        """Forward public optimization flags to the native module loader.

        These flags used to be accepted by ``from_pretrained`` but dropped
        while constructing :class:`RuntimePolicy`, so a Workspace manifest
        could claim an optimization that was never applied.  Keep the public
        option names explicit: unknown loading kwargs must not silently become
        runtime policy settings.
        """

        explicit = options.get("runtime_options")
        if explicit is None:
            resolved: dict[str, object] = {}
        elif isinstance(explicit, Mapping):
            resolved = {str(name): value for name, value in explicit.items()}
        else:
            raise TypeError("runtime_options must be a mapping when provided")
        # No native denoiser installs a combined residual/AdaLN runtime.
        # This old opt-in must fail before loading instead of being ignored.
        for value in (options.get("fused_residual_adaln"), resolved.get("fused_residual_adaln")):
            if value is None:
                continue
            if not isinstance(value, bool):
                raise TypeError("fused_residual_adaln must be a bool")
            if value:
                raise ValueError("fused_residual_adaln is unavailable: the combined kernel is not implemented")
        for name in cls.RUNTIME_POLICY_OPTION_KEYS:
            if name in options and options[name] is not None:
                resolved[name] = options[name]
        return resolved

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "NativeVisualDiffusionPipeline":
        options = cls._options(model_path, required_components, kwargs)
        if model_id is not None:
            configured_model_id = options.get("model_id")
            if configured_model_id is not None and str(configured_model_id) != str(model_id):
                raise ValueError(
                    "native pipeline model_id differs between the runner and loading options"
                )
            options["model_id"] = model_id
        requested_model_id = cls._requested_model_id(options)
        runtime_policy_options = cls._runtime_policy_options(options)
        sequence_parallel = runtime_policy_options.get(
            "sequence_parallel",
            runtime_policy_options.get("sp_degree", False),
        )
        cfg_parallel = runtime_policy_options.get(
            "cfg_parallel",
            runtime_policy_options.get("cfg_parallel_degree", False),
        )
        raw_vae_parallel = runtime_policy_options.get(
            "vae_parallel",
            runtime_policy_options.get("vae_parallel_degree", False),
        )
        if isinstance(raw_vae_parallel, bool):
            vae_parallel_degree = (
                int(os.getenv("WORLD_SIZE", "1")) if raw_vae_parallel else 1
            )
            if raw_vae_parallel and vae_parallel_degree <= 1:
                raise RuntimeError(
                    "vae_parallel=True requires torchrun WORLD_SIZE>1 or an explicit degree"
                )
        else:
            vae_parallel_degree = int(raw_vae_parallel or 1)
        if vae_parallel_degree <= 0:
            raise ValueError("vae_parallel degree must be positive")
        if (
            "vae_parallel" in runtime_policy_options
            and "vae_parallel_degree" in runtime_policy_options
            and int(runtime_policy_options["vae_parallel_degree"])
            != vae_parallel_degree
        ):
            raise ValueError("vae_parallel and vae_parallel_degree must match")

        if sequence_parallel or cfg_parallel or vae_parallel_degree > 1:
            degree = int(sequence_parallel or 1)
            cfg_degree = int(cfg_parallel or 1)
            if "sequence_parallel" in runtime_policy_options and "sp_degree" in runtime_policy_options:
                if int(runtime_policy_options["sp_degree"]) != degree:
                    raise ValueError("sequence_parallel and sp_degree must match")
            if "cfg_parallel" in runtime_policy_options and "cfg_parallel_degree" in runtime_policy_options:
                if int(runtime_policy_options["cfg_parallel_degree"]) != cfg_degree:
                    raise ValueError("cfg_parallel and cfg_parallel_degree must match")
            if torch.device(device).type != "cuda":
                raise ValueError("sequence/CFG/VAE parallelism requires a CUDA pipeline device")
            from worldfoundry.core.distributed.sequence_parallel.runtime import (
                ensure_parallel_runtime,
            )

            mesh_world_size = degree * cfg_degree
            if mesh_world_size > 1 and vae_parallel_degree not in {
                1,
                mesh_world_size,
            }:
                raise ValueError(
                    "vae_parallel degree must equal the complete SP×CFG world size "
                    f"({mesh_world_size}) when combined with sequence/CFG parallelism"
                )
            if mesh_world_size == 1:
                degree = vae_parallel_degree
            device = str(ensure_parallel_runtime(degree, cfg_degree))
            if sequence_parallel:
                runtime_policy_options["sequence_parallel"] = degree
            if cfg_parallel:
                runtime_policy_options["cfg_parallel"] = cfg_degree
            runtime_policy_options.pop("sp_degree", None)
            runtime_policy_options.pop("cfg_parallel_degree", None)
            if vae_parallel_degree > 1:
                runtime_policy_options["vae_parallel"] = vae_parallel_degree
            runtime_policy_options.pop("vae_parallel_degree", None)
        native = NativeDiffusionPipeline.from_pretrained(
            requested_model_id,
            policy=RuntimePolicy(
                device=torch.device(device),
                dtype=parse_torch_dtype(
                    options.get("torch_dtype", options.get("weight_dtype", options.get("dtype"))),
                    owner=cls.OWNER,
                ),
                attention=parse_attention_backend(
                    options.get(
                        "attention_backend",
                        options.get("attention_implementation", options.get("attention", "auto")),
                    ),
                    owner=cls.OWNER,
                ),
                offload=parse_offload_policy(
                    options.get("offload_mode", "block"),
                    allow_disk=False,
                    owner=cls.OWNER,
                ),
                quantization=parse_quantization_policy(
                    options.get("quantization", options.get("quantization_mode")),
                    owner=cls.OWNER,
                ),
                compile=bool(options.get("compile", options.get("torch_compile", False))),
                options=runtime_policy_options,
            ),
            checkpoint_overrides=cls._checkpoint_overrides(model_path, options),
            component_options=cls._component_options(options),
        )
        return cls(native_pipeline=native, device=device, model_id=requested_model_id)

    @classmethod
    def plan(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        options = cls._options(model_path, required_components, kwargs)
        cls._runtime_policy_options(options)
        model_id = cls._requested_model_id(options)
        return {
            "model_id": model_id,
            "backend": "worldfoundry-native-diffusion",
            "runtime": "worldfoundry-native-diffusion",
            "native_inference": True,
            "checkpoints": cls._checkpoint_overrides(model_path, options),
            "blocked": False,
        }

    def process(
        self,
        prompt: str | list[str] = "",
        images: Any = None,
        video: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if images is not None and not self.ACCEPTS_IMAGES:
            raise ValueError(f"{self.model_id} does not accept image inputs")
        if images is None and self.REQUIRES_IMAGES:
            raise ValueError(f"{self.model_id} requires one or more reference images")
        if video is not None and not self.ACCEPTS_VIDEO:
            raise ValueError(f"{self.model_id} does not accept video inputs")
        values = prompt if isinstance(prompt, list) else [prompt]
        processed = []
        for value in values:
            self.operator.get_interaction(value)
            try:
                interaction = self.operator.process_interaction()
            finally:
                self.operator.delete_last_interaction()
            processed.append(str(interaction["processed_prompt"]))
        return {
            "prompt": processed if isinstance(prompt, list) else processed[0],
            "images": images,
            "video": video,
        }

    @staticmethod
    def _pop_first(options: dict[str, Any], names: tuple[str, ...]) -> Any:
        selected = None
        for name in names:
            if name in options:
                value = options.pop(name)
                if selected is None:
                    selected = value
        return selected

    def __call__(
        self,
        prompt: str | list[str] = "",
        images: Any = None,
        image_path: str | Path | None = None,
        video: Any = None,
        video_path: str | Path | None = None,
        input_path: str | Path | None = None,
        ref_image_path: str | Path | None = None,
        interactions: Any = None,
        negative_prompt: str | list[str] | None = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int = 0,
        output_type: str = "auto",
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        # Evaluation attaches sample/task metadata as operator kwargs for
        # action-policy pipelines.  Native diffusion recipes do not consume
        # that routing metadata, so keep it outside the strict model option
        # boundary.  A reference-image path is a standard image alias.
        del operator_kwargs
        if images is None and self.ACCEPTS_IMAGES and ref_image_path is not None:
            images = ref_image_path
        # Studio exposes one generic input picker.  Bind it only for model
        # families that consume the corresponding modality; prompt-only
        # pipelines may safely share the same Workspace task profile.
        if images is None and self.ACCEPTS_IMAGES:
            images = image_path if image_path is not None else input_path
        if video is None and self.ACCEPTS_VIDEO:
            video = video_path if video_path is not None else input_path
        if output_path is None and output_dir is not None:
            # Studio owns a per-run directory and passes it to every pipeline.
            # Leave the filename extension open so the shared saver can choose
            # an image or video suffix from the actual decoded sample.
            output_path = Path(output_dir).expanduser() / self.model_id
        if interactions not in (None, (), []) and not self.ACCEPTS_INTERACTIONS:
            raise ValueError(f"{self.model_id} does not accept action interactions")
        options = dict(kwargs)
        # ``task_type`` is Workspace routing metadata rather than a model
        # parameter.  The selected public pipeline class already fixes the
        # task, so consuming it here keeps Studio and direct calls on the same
        # strict inference boundary.
        options.pop("task_type", None)
        studio_videos = options.pop("videos", None)
        if video is None and studio_videos is not None:
            video = studio_videos
        aliased_num_frames = self._pop_first(options, self.NUM_FRAMES_ALIASES)
        aliased_num_inference_steps = self._pop_first(options, self.NUM_INFERENCE_STEP_ALIASES)
        aliased_guidance_scale = self._pop_first(options, self.GUIDANCE_SCALE_ALIASES)
        if num_frames is None:
            num_frames = aliased_num_frames
        if num_inference_steps is None:
            num_inference_steps = aliased_num_inference_steps
        if guidance_scale is None:
            guidance_scale = aliased_guidance_scale

        scheduler_options = dict(self.DEFAULT_SCHEDULER_OPTIONS)
        for alias, canonical in self.SCHEDULER_OPTION_ALIASES.items():
            if alias in options:
                value = options.pop(alias)
                if canonical not in options and value is not None:
                    options[canonical] = value
        for key in tuple(scheduler_options):
            if key in options:
                value = options.pop(key)
                if value is not None:
                    scheduler_options[key] = value

        request_inputs = dict(self.REQUEST_INPUT_DEFAULTS)
        for alias, canonical in self.REQUEST_INPUT_ALIASES.items():
            if alias in options and canonical not in options:
                options[canonical] = options.pop(alias)
        for key in tuple(request_inputs):
            if key in options:
                value = options.pop(key)
                if value is not None:
                    request_inputs[key] = value
        if options:
            raise TypeError(f"unsupported {self.model_id} inference options: {sorted(options)}")

        processed = self.process(prompt=prompt, images=images, video=video)
        actual_fps = int(fps or self.DEFAULT_FPS)
        request_inputs.update({"fps": actual_fps, "return_latent": output_type == "latent"})
        if self.ACCEPTS_IMAGES:
            request_inputs["images"] = processed["images"]
        if self.ACCEPTS_VIDEO:
            request_inputs["video"] = processed["video"]
        if self.ACCEPTS_INTERACTIONS and interactions is not None:
            request_inputs["interactions"] = interactions
        output = self.native_pipeline(
            DiffusionRequest(
                prompt=processed["prompt"],
                negative_prompt=(
                    self.DEFAULT_NEGATIVE_PROMPT if negative_prompt is None else negative_prompt
                ),
                height=int(height or self.DEFAULT_HEIGHT),
                width=int(width or self.DEFAULT_WIDTH),
                num_frames=int(num_frames or self.DEFAULT_NUM_FRAMES),
                sampling=SamplingConfig(
                    num_inference_steps=int(num_inference_steps or self.DEFAULT_NUM_INFERENCE_STEPS),
                    guidance_scale=float(
                        guidance_scale if guidance_scale is not None else self.DEFAULT_GUIDANCE_SCALE
                    ),
                    seed=secrets.randbits(63) if int(seed) < 0 else int(seed),
                    scheduler_options=scheduler_options,
                ),
                inputs=request_inputs,
            )
        )
        artifact_path = None
        if output_path is not None and output_type != "latent":
            sample_to_save = output.sample.unsqueeze(2) if output.sample.ndim == 4 else output.sample
            artifact_path = save_image_or_video_tensor(
                sample_to_save,
                output_path,
                fps=actual_fps,
                value_range=self.OUTPUT_VALUE_RANGE,
            )
        is_image = output.sample.ndim == 4
        result = {
            "sample": output.sample,
            "generated": output.sample,
            "latents": output.latents,
            "artifact_path": artifact_path,
            "generated_image_path": artifact_path if is_image else None,
            "generated_video_path": None if is_image else artifact_path,
            "model_name": self.model_id,
            "generation_type": self.GENERATION_TYPE,
            "metadata": dict(output.metadata),
        }
        if is_image:
            result.update(image=output.sample, generated_image=output.sample)
        else:
            result.update(video=output.sample, generated_video=output.sample)
        return result if return_dict else (artifact_path or output.sample)

    def stream(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        **kwargs: Any,
    ) -> Any:
        result = self(prompt=prompt, images=images, video=video, return_dict=True, **kwargs)
        value = result.get("artifact_path") or result["sample"]
        self.memory_module.record(value, metadata={"prompt": prompt, "model_name": self.model_id})
        return value

    def get_synthesis_model(self) -> NativeDiffusionPipeline:
        return self.native_pipeline


__all__ = ["NativeVisualDiffusionPipeline"]
