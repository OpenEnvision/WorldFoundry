"""WoW synthesis adapters on the canonical Wan and Cosmos2 runtimes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image

from worldfoundry.core.media.types import VIDEO_EXTENSIONS
from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.core.media.resolutions import VIDEO_RES_SIZE_INFO

from ...base_synthesis import BaseSynthesis

WAN_DEFAULT_NEGATIVE_PROMPT = "low quality, distorted, ugly, bad anatomy"
DIT_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
    "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)


def _as_local_dir(path: str | os.PathLike[str], label: str) -> Path:
    model_root = Path(path).expanduser()
    if model_root.is_dir():
        return model_root

    path_text = str(path)
    is_hf_repo_id = "/" in path_text and not path_text.startswith(("/", ".", "~")) and "://" not in path_text
    if is_hf_repo_id:
        try:
            return resolve_local_hf_model_path(path_text)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{label} repository {path_text!r} is not staged locally. "
                "Pre-download the pinned checkpoint before WoW inference."
            ) from exc

    raise FileNotFoundError(
        f"{label} must resolve to a local checkpoint directory for WoW in-tree inference: {model_root}"
    )


def _custom_checkpoint_path(model_root: Path, custom_checkpoint_name: str | os.PathLike[str] | None) -> Path | None:
    if not custom_checkpoint_name:
        return None
    candidate = Path(custom_checkpoint_name).expanduser()
    return candidate if candidate.is_absolute() else model_root / candidate


def _resolve_dit_path(
    pretrained_model_path: str | os.PathLike[str] | None,
    *,
    model_size: str,
    resolution: str,
    fps: int,
    explicit_dit_path: str | os.PathLike[str] | None,
) -> str:
    if explicit_dit_path:
        candidate = Path(explicit_dit_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"WoW DiT checkpoint not found: {candidate}")
        return str(candidate)

    if pretrained_model_path:
        path_text = str(pretrained_model_path)
        candidate = Path(path_text).expanduser()
        if not candidate.exists():
            is_hf_repo_id = "/" in path_text and not path_text.startswith(("/", ".", "~")) and "://" not in path_text
            if is_hf_repo_id:
                try:
                    candidate = resolve_local_hf_model_path(path_text)
                except FileNotFoundError:
                    pass
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            relative_names = (
                Path("wow_dit_2b.pt"),
                Path(f"model-{resolution}p-{fps}fps.pt"),
                Path("model.pt"),
                Path("dit_models/wow-dit-2b/checkpoints/wow_dit_2b.pt"),
            )
            for relative_name in relative_names:
                nested = candidate / relative_name
                if nested.is_file():
                    return str(nested)

    raise FileNotFoundError(
        "WoW DiT backend requires a local wow_dit_2b.pt checkpoint file. "
        "Pass dit_path or a model_path directory containing that file; "
        f"model_size={model_size}, resolution={resolution}, fps={fps}."
    )


def _option(options: Mapping[str, Any], args: Any, name: str, default: Any) -> Any:
    value = options.get(name)
    if value is not None:
        return value
    if args is not None and hasattr(args, name):
        value = getattr(args, name)
        if value is not None:
            return value
    return default


class WoWSynthesis(BaseSynthesis):
    """WoW synthesis wrapper matching the official Wan and DiT demo parameters."""

    def __init__(
        self,
        pipeline: Any,
        *,
        backend: str,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.backend = backend
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | Mapping[str, Any] | None = "X-Humanoid/WoW-1-Wan-14B-600k",
        synthesis_args: Any = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "WoWSynthesis":
        """Load the native WoW Wan checkpoint route by default.

        The official repository marks the Wan demo as recommended; its checkpoint
        is assembled on the shared Wan runtime. The DiT 2B checkpoint uses the
        canonical Cosmos Predict2 Video2World recipe and
        remains available through ``runtime_backend="dit2b"`` plus a local
        ``dit_path`` or checkpoint file as ``model_path``.
        """

        options: dict[str, Any] = dict(kwargs)
        if isinstance(pretrained_model_path, Mapping):
            options.update(pretrained_model_path)
            pretrained_model_path = (
                options.pop("model_path", None)
                or options.pop("pretrained_model_path", None)
                or options.pop("checkpoint_folder", None)
                or options.pop("repo_root", None)
            )

        backend = str(
            options.pop("runtime_backend", None) or getattr(synthesis_args, "runtime_backend", "wan") or "wan"
        ).lower()
        if backend in {"dit", "dit-2b", "dit2b", "cosmos", "cosmos2"}:
            return cls._from_dit2b(
                pretrained_model_path=pretrained_model_path,
                synthesis_args=synthesis_args,
                device=device,
                **options,
            )
        return cls._from_wan(
            pretrained_model_path=pretrained_model_path,
            synthesis_args=synthesis_args,
            device=device,
            **options,
        )

    @classmethod
    def _from_wan(
        cls,
        *,
        pretrained_model_path: str | os.PathLike[str] | None,
        synthesis_args: Any,
        device: str,
        **kwargs: Any,
    ) -> "WoWSynthesis":
        if pretrained_model_path is None:
            raise FileNotFoundError("WoW Wan backend requires a local checkpoint folder.")
        model_root = _as_local_dir(pretrained_model_path, "WoW Wan checkpoint")
        custom_checkpoint_name = kwargs.pop(
            "custom_checkpoint",
            kwargs.pop("custom_checkpoint_name", getattr(synthesis_args, "custom_checkpoint", "WoW_video_dit.pt")),
        )

        custom_checkpoint = _custom_checkpoint_path(model_root, custom_checkpoint_name)
        if custom_checkpoint is None or not custom_checkpoint.is_file():
            raise FileNotFoundError(
                "WoW Wan inference requires its trained transformer checkpoint; "
                f"expected {custom_checkpoint or model_root / 'WoW_video_dit.pt'}"
            )

        from .native_wan_pipeline import load_wow_wan_pipeline

        pipeline = load_wow_wan_pipeline(
            model_root=model_root,
            transformer_checkpoint=custom_checkpoint,
            device=device,
            torch_dtype=torch.bfloat16,
        )

        enable_vram = getattr(synthesis_args, "enable_vram_management", True) and not getattr(
            synthesis_args, "no_vram_management", False
        )
        if enable_vram:
            persistent_param_gb = getattr(synthesis_args, "persistent_param_gb", 70)
            pipeline.enable_vram_management(num_persistent_param_in_dit=int(persistent_param_gb * 10**9))
        else:
            # The native loader builds components on CPU. Without VRAM management,
            # the staged runner does not move them before prompt encoding.
            pipeline.to(device)

        return cls(pipeline=pipeline, backend="wan", device=device)

    @classmethod
    def _from_dit2b(
        cls,
        *,
        pretrained_model_path: str | os.PathLike[str] | None,
        synthesis_args: Any,
        device: str,
        **kwargs: Any,
    ) -> "WoWSynthesis":
        model_size = str(_option(kwargs, synthesis_args, "model_size", "2B")).upper()
        if model_size != "2B":
            raise ValueError("WoW DiT exposes only the released 2B Cosmos Predict2 checkpoint")
        resolution = str(_option(kwargs, synthesis_args, "resolution", "720"))
        fps = int(_option(kwargs, synthesis_args, "fps", 16))
        if resolution not in {"480", "720"}:
            raise ValueError("WoW DiT resolution must be '480' or '720'")
        if fps not in {10, 16}:
            raise ValueError("WoW DiT fps must be 10 or 16")
        num_gpus = int(_option(kwargs, synthesis_args, "num_gpus", 1))
        if num_gpus != 1:
            raise NotImplementedError(
                "WoW DiT context parallelism is not yet implemented by the canonical Cosmos2 runner"
            )
        if not bool(_option(kwargs, synthesis_args, "disable_guardrail", True)):
            raise NotImplementedError("WoW DiT native inference does not yet bind a guardrail component")
        if not bool(_option(kwargs, synthesis_args, "disable_prompt_refiner", True)):
            raise NotImplementedError("WoW DiT native inference does not yet bind a prompt-refiner component")

        dit_path = _resolve_dit_path(
            pretrained_model_path,
            model_size=model_size,
            resolution=resolution,
            fps=fps,
            explicit_dit_path=_option(kwargs, synthesis_args, "dit_path", None),
        )
        component_paths: dict[str, Any] = {"transformer_model_path": dit_path}
        for name in (
            "vae_model_path",
            "tokenizer_model_path",
            "text_encoder_model_path",
            "text_tokenizer_path",
            "offload_mode",
            "torch_dtype",
            "weight_dtype",
            "dtype",
            "vae_tiling",
            "vae_tile_size",
            "vae_tile_stride",
        ):
            value = kwargs.get(name)
            if value is not None:
                component_paths[name] = value

        from worldfoundry.pipelines.cosmos.pipeline_cosmos_predict2 import CosmosPredict2Pipeline

        pipeline = CosmosPredict2Pipeline.from_pretrained(
            model_path=component_paths,
            device=device,
            model_id="cosmos-predict2-2b-video2world",
        )
        return cls(pipeline=pipeline, backend="dit2b", device=device)

    @torch.no_grad()
    def predict(
        self,
        input_image: Image.Image | None = None,
        text_prompt: str = "",
        synthesis_args: Any = None,
        input_path: str | os.PathLike[str] | None = None,
        output_path: str | os.PathLike[str] | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Generate a WoW video and optionally save/return a runner mapping."""

        if self.backend == "dit2b":
            result = self._predict_dit2b(
                input_image=input_image,
                input_path=input_path,
                text_prompt=text_prompt,
                synthesis_args=synthesis_args,
                output_path=output_path,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            result = self._predict_wan(
                input_image=input_image,
                text_prompt=text_prompt,
                synthesis_args=synthesis_args,
                output_path=output_path,
                fps=fps,
                return_dict=return_dict,
                **kwargs,
            )
        return result

    def _predict_wan(
        self,
        *,
        input_image: Image.Image | None,
        text_prompt: str,
        synthesis_args: Any,
        output_path: str | os.PathLike[str] | None,
        fps: int | None,
        return_dict: bool,
        **kwargs: Any,
    ) -> Any:
        if input_image is None:
            raise ValueError("WoW Wan backend requires an input image or first frame.")
        steps = int(_option(kwargs, synthesis_args, "steps", 50))
        seed = int(_option(kwargs, synthesis_args, "seed", 42))
        tiled = bool(_option(kwargs, synthesis_args, "tiled", not bool(getattr(synthesis_args, "no_tiled", False))))
        num_frames = int(_option(kwargs, synthesis_args, "num_frames", 41))
        negative_prompt = str(_option(kwargs, synthesis_args, "negative_prompt", WAN_DEFAULT_NEGATIVE_PROMPT))
        output_fps = int(fps if fps is not None else _option(kwargs, synthesis_args, "output_fps", 15))

        output_video = self.pipeline(
            prompt=text_prompt,
            negative_prompt=negative_prompt,
            input_image=input_image,
            num_inference_steps=steps,
            seed=seed,
            tiled=tiled,
            num_frames=num_frames,
        )

        artifact_path = ""
        if output_path is not None:
            from worldfoundry.core.media.codecs.video import write_video

            artifact_path = str(output_path)
            write_video(output_video, artifact_path, fps=output_fps, quality=5)

        if return_dict:
            return {
                "status": "ok",
                "runtime": "wow-wan-native",
                "backend_quality": "native_recipe",
                "artifact_kind": "generated_video",
                "artifact_path": artifact_path,
                "video": output_video,
                "metadata": {
                    "steps": steps,
                    "seed": seed,
                    "tiled": tiled,
                    "num_frames": num_frames,
                    "fps": output_fps,
                    "negative_prompt": negative_prompt,
                },
            }
        return output_video

    def _predict_dit2b(
        self,
        *,
        input_image: Image.Image | None,
        input_path: str | os.PathLike[str] | None,
        text_prompt: str,
        synthesis_args: Any,
        output_path: str | os.PathLike[str] | None,
        return_dict: bool,
        **kwargs: Any,
    ) -> Any:
        if input_path is None and input_image is None:
            raise ValueError("WoW DiT backend requires input_path or input_image.")

        negative_prompt = str(_option(kwargs, synthesis_args, "negative_prompt", DIT_DEFAULT_NEGATIVE_PROMPT))
        num_conditional_frames = int(_option(kwargs, synthesis_args, "num_conditional_frames", 1))
        if num_conditional_frames not in {1, 5}:
            raise ValueError("WoW DiT num_conditional_frames must be 1 or 5")
        guidance = float(_option(kwargs, synthesis_args, "guidance", 7.0))
        seed = int(_option(kwargs, synthesis_args, "seed", 42))
        num_sampling_step = int(_option(kwargs, synthesis_args, "num_sampling_step", 35))
        resolution = str(_option(kwargs, synthesis_args, "resolution", "720"))
        fps = int(_option(kwargs, synthesis_args, "fps", 16))
        if fps not in {10, 16}:
            raise ValueError("WoW DiT fps must be 10 or 16")
        try:
            height, width = VIDEO_RES_SIZE_INFO[resolution]["9,16"]
        except KeyError as exc:
            raise ValueError(f"unsupported WoW DiT resolution: {resolution!r}") from exc
        if num_conditional_frames == 5 and (
            input_path is None or Path(input_path).suffix.lower() not in VIDEO_EXTENSIONS
        ):
            raise ValueError("WoW DiT 5-frame conditioning requires a video input path")
        num_frames = 61 if fps == 10 else 93
        media_options = {"input_path": str(input_path)} if input_path is not None else {"image": input_image}

        native_result = self.pipeline(
            prompt=text_prompt,
            negative_prompt=negative_prompt,
            output_path=output_path,
            guidance_scale=guidance,
            num_inference_steps=num_sampling_step,
            fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            num_latent_conditional_frames=(num_conditional_frames - 1) // 4 + 1,
            return_dict=True,
            **media_options,
        )
        output_video = native_result["video"]
        artifact_path = str(native_result.get("artifact_path") or "")

        if return_dict:
            return {
                "status": "ok",
                "runtime": "wow-dit2b-native-cosmos2",
                "backend_quality": "native_recipe",
                "artifact_kind": "generated_video",
                "artifact_path": artifact_path,
                "video": output_video,
                "metadata": {
                    "input_path": str(input_path) if input_path is not None else None,
                    "num_conditional_frames": num_conditional_frames,
                    "guidance": guidance,
                    "seed": seed,
                    "num_sampling_step": num_sampling_step,
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                    "fps": fps,
                    "negative_prompt": negative_prompt,
                    "native_model_id": "cosmos-predict2-2b-video2world",
                },
            }
        return output_video
