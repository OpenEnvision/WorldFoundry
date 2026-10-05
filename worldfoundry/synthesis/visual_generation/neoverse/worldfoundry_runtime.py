from __future__ import annotations

import contextlib
import inspect
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from .runtime_env import (
    DEFAULT_NEOVERSE_LORA_NAME,
    DEFAULT_NEOVERSE_REPO,
    ensure_neoverse_runtime,
    resolve_neoverse_lora_path,
    resolve_neoverse_model_dir,
    resolve_neoverse_reconstructor_path,
    runtime_root,
)


DEFAULT_PROMPT = (
    "A smooth video with complete scene content. Inpaint any missing regions or margins naturally "
    "to match the surrounding scene."
)


def _pil_to_tensor(image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class NeoVerseOfficialRuntime:
    """Native NeoVerse runtime assembled from shared Wan components."""

    def __init__(
        self,
        pipeline,
        *,
        model_dir: str,
        reconstructor_path: str,
        lora_path: Optional[str],
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16,
        default_prompt: str = DEFAULT_PROMPT,
        default_negative_prompt: str = "",
        default_cfg_scale: float = 1.0,
        default_num_inference_steps: int = 4,
        height: int = 336,
        width: int = 560,
    ) -> None:
        self.pipeline = pipeline
        self.model_dir = str(Path(model_dir).expanduser().resolve())
        self.reconstructor_path = str(Path(reconstructor_path).expanduser().resolve())
        self.lora_path = str(Path(lora_path).expanduser().resolve()) if lora_path else None
        self.device = device
        self.weight_dtype = weight_dtype
        self.default_prompt = default_prompt
        self.default_negative_prompt = default_negative_prompt
        self.default_cfg_scale = float(default_cfg_scale)
        self.default_num_inference_steps = int(default_num_inference_steps)
        self.height = int(height)
        self.width = int(width)

    @classmethod
    def bundled_runtime_root(cls) -> str:
        return str(runtime_root().resolve())

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = DEFAULT_NEOVERSE_REPO,
        args=None,
        device: Optional[str] = None,
        weight_dtype: torch.dtype = torch.bfloat16,
        reconstructor_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        disable_lora: bool = False,
        lora_alpha: float = 1.0,
        enable_vram_management: bool = False,
        default_prompt: str = DEFAULT_PROMPT,
        default_negative_prompt: str = "",
        cfg_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        height: int = 336,
        width: int = 560,
        **kwargs,
    ) -> "NeoVerseOfficialRuntime":
        del args, kwargs
        ensure_neoverse_runtime()
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import (
            WanVideoVAE,
            WanVideoVAEStateDictConverter,
        )
        from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
            WAN21_T2V_14B_CONFIG,
            WanModelStateDictConverter,
        )
        from worldfoundry.base_models.diffusion_model.models.encoders.wan import WanTextEncoder
        from worldfoundry.base_models.diffusion_model.models.networks.wan import WanModel
        from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import (
            NeoVerseControlBranch,
            NeoVerseControlBranchDictConverter,
        )
        from worldfoundry.synthesis.visual_generation.neoverse.native.worldmirror import (
            NeoVerseWorldMirror,
        )
        from worldfoundry.core.model_loading import hash_state_dict_keys, load_model, load_state_dict
        from worldfoundry.synthesis.visual_generation.neoverse.native.depth_anything_reconstructor import (
            DepthAnything3Reconstructor,
        )
        from worldfoundry.synthesis.visual_generation.neoverse.native.pipeline import (
            WanVideoNeoVersePipeline,
        )

        resolved_device = device or "cuda"
        model_dir = resolve_neoverse_model_dir(pretrained_model_path or DEFAULT_NEOVERSE_REPO)
        resolved_reconstructor_path = resolve_neoverse_reconstructor_path(
            model_dir,
            override=reconstructor_path,
        )
        resolved_lora_path = resolve_neoverse_lora_path(
            model_dir,
            override=lora_path,
            use_lora=not disable_lora,
        )

        diffusion_files = sorted(model_dir.glob("diffusion_pytorch_model*.safetensors"))
        if not diffusion_files:
            raise FileNotFoundError(f"NeoVerse diffusion checkpoint shards not found in {model_dir}")
        tokenizer_path = model_dir / "google" / "umt5-xxl"
        if not tokenizer_path.is_dir():
            tokenizer_path = model_dir / "google"

        combined_state = load_state_dict(
            [str(path) for path in diffusion_files],
            torch_dtype=weight_dtype,
            device="cpu",
        )
        dit_state, dit_config = WanModelStateDictConverter().from_civitai(combined_state)
        dit = load_model(
            WanModel,
            path=None,
            config=dit_config or WAN21_T2V_14B_CONFIG,
            torch_dtype=weight_dtype,
            device="cpu",
            state_dict=dit_state,
        )
        control_state, control_config = NeoVerseControlBranchDictConverter().from_civitai(combined_state)
        control_branch = load_model(
            NeoVerseControlBranch,
            path=None,
            config=control_config,
            torch_dtype=weight_dtype,
            device="cpu",
            state_dict=control_state,
            strict=False,
        )
        text_encoder = load_model(
            WanTextEncoder,
            str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
            torch_dtype=weight_dtype,
            device="cpu",
        )
        vae = load_model(
            WanVideoVAE,
            str(model_dir / "Wan2.1_VAE.pth"),
            torch_dtype=weight_dtype,
            device="cpu",
            state_dict_converter=WanVideoVAEStateDictConverter().from_civitai,
        )

        reconstructor_state = load_state_dict(str(resolved_reconstructor_path), device="cpu")
        reconstructor_hash = hash_state_dict_keys(reconstructor_state)
        if reconstructor_hash == "1a1d001a35f78f3a7796a1e719ead340":
            reconstructor_class = NeoVerseWorldMirror
        elif reconstructor_hash == "252f1c3923a62665aee9b32f1b18afb5":
            reconstructor_class = DepthAnything3Reconstructor
        else:
            raise ValueError(f"Unsupported NeoVerse reconstructor checkpoint layout: {reconstructor_hash}")
        converted_reconstructor, reconstructor_config = reconstructor_class.state_dict_converter().from_civitai(
            reconstructor_state
        )
        reconstructor_config = dict(reconstructor_config)
        strict_reconstructor = bool(reconstructor_config.pop("strict_load", True))
        reconstructor_dtype = (
            torch.float32 if reconstructor_config.pop("upcast_to_float32", False) else weight_dtype
        )
        reconstructor = load_model(
            reconstructor_class,
            path=None,
            config=reconstructor_config,
            torch_dtype=reconstructor_dtype,
            device="cpu",
            state_dict=converted_reconstructor,
            strict=strict_reconstructor,
        )

        pipeline = WanVideoNeoVersePipeline.from_components(
            text_encoder=text_encoder,
            dit=dit,
            vae=vae,
            control_branch=control_branch,
            reconstructor=reconstructor,
            tokenizer_path=str(tokenizer_path),
            lora_path=str(resolved_lora_path) if resolved_lora_path else None,
            lora_alpha=lora_alpha,
            device=resolved_device,
            torch_dtype=weight_dtype,
            enable_vram_management=enable_vram_management,
        )

        return cls(
            pipeline=pipeline,
            model_dir=str(model_dir),
            reconstructor_path=str(resolved_reconstructor_path),
            lora_path=str(resolved_lora_path) if resolved_lora_path else None,
            device=resolved_device,
            weight_dtype=weight_dtype,
            default_prompt=default_prompt,
            default_negative_prompt=default_negative_prompt,
            default_cfg_scale=cfg_scale if cfg_scale is not None else (1.0 if resolved_lora_path else 5.0),
            default_num_inference_steps=num_inference_steps if num_inference_steps is not None else (4 if resolved_lora_path else 50),
            height=height,
            width=width,
        )

    def _autocast_context(self):
        if str(self.device).startswith("cuda"):
            return torch.amp.autocast("cuda", dtype=self.weight_dtype)
        return contextlib.nullcontext()

    def _build_views(self, images: Sequence[Any], *, static_scene: bool):
        tensors = [_pil_to_tensor(image)[None] for image in images]
        frames = torch.stack(tensors, dim=1).to(self.device)
        num_frames = len(images)
        if static_scene:
            timestamps = torch.zeros((1, num_frames), dtype=torch.int64, device=self.device)
            is_static = torch.ones((1, num_frames), dtype=torch.bool, device=self.device)
        else:
            timestamps = torch.arange(num_frames, dtype=torch.int64, device=self.device).unsqueeze(0)
            is_static = torch.zeros((1, num_frames), dtype=torch.bool, device=self.device)

        return {
            "img": frames,
            "is_target": torch.zeros((1, num_frames), dtype=torch.bool, device=self.device),
            "is_static": is_static,
            "timestamp": timestamps,
        }

    def _build_camera_trajectory(
        self,
        *,
        keyframes=None,
        predefined_trajectory: Optional[str] = None,
        num_frames: Optional[int] = None,
        trajectory_file: Optional[str] = None,
        trajectory_data=None,
        trajectory_mode: str = "relative",
        trajectory_name: str = "neoverse_trajectory",
        zoom_ratio: float = 1.0,
        angle: Optional[float] = None,
        distance: Optional[float] = None,
        orbit_radius: Optional[float] = None,
        use_first_frame: bool = True,
    ):
        ensure_neoverse_runtime()
        from worldfoundry.synthesis.visual_generation.neoverse.native.auxiliary import (
            CameraTrajectory,
        )

        if predefined_trajectory is not None:
            return CameraTrajectory.from_predefined(
                predefined_trajectory,
                num_frames=num_frames or 81,
                mode=trajectory_mode,
                angle=angle,
                distance=distance,
                orbit_radius=orbit_radius,
                zoom_ratio=zoom_ratio,
            )

        if trajectory_file is not None:
            return CameraTrajectory.from_json(trajectory_file)

        if trajectory_data is not None:
            if isinstance(trajectory_data, list):
                if num_frames is None:
                    raise ValueError("num_frames is required when trajectory_data is a keyframe list.")
                return CameraTrajectory.from_keyframes(
                    trajectory_data,
                    num_frames=num_frames,
                    mode=trajectory_mode,
                    name=trajectory_name,
                    zoom_ratio=zoom_ratio,
                    use_first_frame=use_first_frame,
                )

            if not isinstance(trajectory_data, dict):
                raise TypeError(
                    f"trajectory_data must be a list or dict, got {type(trajectory_data)!r}."
                )

            resolved_mode = trajectory_data.get("mode", trajectory_mode)
            resolved_name = trajectory_data.get("name", trajectory_name)
            resolved_zoom_ratio = trajectory_data.get("zoom_ratio", zoom_ratio)
            resolved_use_first_frame = trajectory_data.get("use_first_frame", use_first_frame)
            resolved_num_frames = trajectory_data.get("num_frames", num_frames)
            if resolved_num_frames is None:
                raise ValueError("Unable to determine num_frames for trajectory_data.")

            if "keyframes" in trajectory_data:
                keyframes_list = []
                for keyframe in trajectory_data["keyframes"]:
                    frame_key, operations = next(iter(keyframe.items()))
                    keyframes_list.append({int(frame_key): operations})
                return CameraTrajectory.from_keyframes(
                    keyframes_list,
                    num_frames=resolved_num_frames,
                    mode=resolved_mode,
                    name=resolved_name,
                    zoom_ratio=resolved_zoom_ratio,
                    use_first_frame=resolved_use_first_frame,
                )

            if "trajectory" in trajectory_data:
                trajectory = trajectory_data["trajectory"]
                if "frame_indices" not in trajectory or "frame_matrices" not in trajectory:
                    raise ValueError(
                        "trajectory_data['trajectory'] must contain frame_indices and frame_matrices."
                    )
                cameras = CameraTrajectory._interpolate_sparse_matrices(
                    trajectory["frame_indices"],
                    trajectory["frame_matrices"],
                    resolved_num_frames,
                )
                return CameraTrajectory(
                    cameras,
                    mode=resolved_mode,
                    name=resolved_name,
                    zoom_ratio=resolved_zoom_ratio,
                    use_first_frame=resolved_use_first_frame,
                )

            raise ValueError("trajectory_data must contain either 'keyframes' or 'trajectory'.")

        if keyframes is None:
            raise ValueError("Either keyframes, trajectory_file, or trajectory_data must be provided.")
        if num_frames is None:
            raise ValueError("num_frames is required when building a trajectory from keyframes.")

        return CameraTrajectory.from_keyframes(
            keyframes,
            num_frames=num_frames,
            mode=trajectory_mode,
            name=trajectory_name,
            zoom_ratio=zoom_ratio,
            use_first_frame=use_first_frame,
        )

    @torch.no_grad()
    def predict(
        self,
        *,
        images,
        prompt: str = "",
        keyframes=None,
        predefined_trajectory: Optional[str] = None,
        num_frames: Optional[int] = None,
        trajectory_file: Optional[str] = None,
        trajectory_data=None,
        trajectory_mode: str = "relative",
        trajectory_name: str = "neoverse_trajectory",
        zoom_ratio: float = 1.0,
        angle: Optional[float] = None,
        distance: Optional[float] = None,
        orbit_radius: Optional[float] = None,
        use_first_frame: bool = True,
        negative_prompt: Optional[str] = None,
        alpha_threshold: float = 1.0,
        seed: int = 42,
        cfg_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        static_scene: Optional[bool] = None,
        save_root: Optional[str] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        del kwargs
        ensure_neoverse_runtime()
        from worldfoundry.synthesis.visual_generation.neoverse.native.auxiliary import (
            homo_matrix_inverse,
            load_video,
        )

        requested_static = bool(static_scene) if static_scene is not None else False
        if isinstance(images, (str, os.PathLike)):
            input_frames = load_video(
                str(images),
                num_frames=num_frames or 1,
                resolution=(self.width, self.height),
                static_scene=requested_static,
            )
        elif isinstance(images, (list, tuple)) and all(isinstance(item, (str, os.PathLike)) for item in images):
            input_frames = load_video(
                [str(item) for item in images],
                num_frames=num_frames or len(images),
                resolution=(self.width, self.height),
                static_scene=requested_static,
            )
        elif hasattr(images, "convert"):
            input_frames = [images.convert("RGB")]
        else:
            input_frames = list(images)
        if len(input_frames) == 0:
            raise ValueError("NeoVerse requires at least one input frame.")

        static_flag = bool(static_scene) if static_scene is not None else len(input_frames) == 1
        cam_traj = self._build_camera_trajectory(
            keyframes=keyframes,
            predefined_trajectory=predefined_trajectory,
            num_frames=num_frames,
            trajectory_file=trajectory_file,
            trajectory_data=trajectory_data,
            trajectory_mode=trajectory_mode,
            trajectory_name=trajectory_name,
            zoom_ratio=zoom_ratio,
            angle=angle,
            distance=distance,
            orbit_radius=orbit_radius,
            use_first_frame=use_first_frame,
        )

        views = self._build_views(input_frames, static_scene=static_flag)
        height = int(input_frames[0].size[1])
        width = int(input_frames[0].size[0])

        previous_save_root = getattr(self.pipeline, "save_root", None)
        self.pipeline.save_root = save_root
        try:
            if getattr(self.pipeline, "vram_management_enabled", False):
                self.pipeline.reconstructor.to(self.device)

            with self._autocast_context():
                reconstructor_forward = getattr(self.pipeline.reconstructor, "forward", None)
                supports_use_motion = (
                    reconstructor_forward is not None
                    and "use_motion" in inspect.signature(reconstructor_forward).parameters
                )
                if supports_use_motion:
                    predictions = self.pipeline.reconstructor(views, is_inference=True, use_motion=False)
                else:
                    predictions = self.pipeline.reconstructor(views, is_inference=True)

            if getattr(self.pipeline, "vram_management_enabled", False):
                self.pipeline.reconstructor.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            gaussians = predictions["splats"]
            intrinsics = predictions.get("rendered_intrinsics", predictions["camera_intrs"])[0]
            input_cam2world = predictions.get("rendered_extrinsics", predictions["camera_poses"])[0]
            timestamps = predictions.get("rendered_timestamps", views["timestamp"])[0]

            if static_flag:
                intrinsics = intrinsics[:1].repeat(len(cam_traj), 1, 1)
                timestamps = timestamps[:1].repeat(len(cam_traj))
            elif input_cam2world.shape[0] != len(cam_traj):
                raise ValueError(
                    "For non-static NeoVerse inputs, trajectory length must match the input-frame count."
                )

            ratio = torch.linspace(1.0, cam_traj.zoom_ratio, intrinsics.shape[0], device=self.device)
            zoomed_intrinsics = intrinsics.clone()
            zoomed_intrinsics[:, 0, 0] *= ratio
            zoomed_intrinsics[:, 1, 1] *= ratio

            target_cam2world = cam_traj.c2w.to(self.device)
            if cam_traj.mode == "relative" and not static_flag:
                target_cam2world = input_cam2world @ target_cam2world

            if isinstance(gaussians, list):
                target_world2cam = homo_matrix_inverse(target_cam2world)
                target_rgb, target_depth, target_alpha = (
                    self.pipeline.reconstructor.gs_renderer.rasterizer.forward(
                        gaussians,
                        render_viewmats=[target_world2cam],
                        render_Ks=[zoomed_intrinsics],
                        render_timestamps=[timestamps],
                        sh_degree=0,
                        width=width,
                        height=height,
                    )
                )
            else:
                render_chunk_size = max(1, int(os.environ.get("NEOVERSE_RENDER_CHUNK_SIZE", "1")))
                rgb_chunks, depth_chunks, alpha_chunks = [], [], []
                raster_colors = gaussians["sh"] if "sh" in gaussians else gaussians["colors"]
                raster_kwargs = {"sh_degree": 0} if "sh" in gaussians else {}
                for start in range(0, target_cam2world.shape[0], render_chunk_size):
                    end = min(start + render_chunk_size, target_cam2world.shape[0])
                    target_world2cam = homo_matrix_inverse(target_cam2world[start:end])
                    chunk_rgb, chunk_depth, chunk_alpha = (
                        self.pipeline.reconstructor.gs_renderer.rasterizer.rasterize_batches(
                            gaussians["means"],
                            gaussians["quats"],
                            gaussians["scales"],
                            gaussians["opacities"],
                            raster_colors,
                            target_world2cam.unsqueeze(0),
                            zoomed_intrinsics[start:end].unsqueeze(0),
                            width=width,
                            height=height,
                            **raster_kwargs,
                        )
                    )
                    rgb_chunks.append(chunk_rgb)
                    depth_chunks.append(chunk_depth)
                    alpha_chunks.append(chunk_alpha)
                target_rgb = torch.cat(rgb_chunks, dim=1)
                target_depth = torch.cat(depth_chunks, dim=1)
                target_alpha = torch.cat(alpha_chunks, dim=1)
            target_mask = (target_alpha > alpha_threshold).float()
            if cam_traj.use_first_frame:
                target_rgb[0, 0] = views["img"][0, 0].permute(1, 2, 0)
                target_mask[0, 0] = 1.0

            wrapped_data = {
                "source_views": views,
                "target_rgb": target_rgb,
                "target_depth": target_depth,
                "target_mask": target_mask,
                "target_poses": target_cam2world.unsqueeze(0),
                "target_intrs": zoomed_intrinsics.unsqueeze(0),
            }

            generated_frames = self.pipeline(
                prompt=prompt or self.default_prompt,
                negative_prompt=negative_prompt if negative_prompt is not None else self.default_negative_prompt,
                seed=seed,
                rand_device=self.device,
                height=height,
                width=width,
                num_frames=len(cam_traj),
                cfg_scale=cfg_scale if cfg_scale is not None else self.default_cfg_scale,
                num_inference_steps=(
                    num_inference_steps
                    if num_inference_steps is not None
                    else self.default_num_inference_steps
                ),
                tiled=False,
                **wrapped_data,
            )
        finally:
            self.pipeline.save_root = previous_save_root

        result = {
            "video": generated_frames,
            "frames": generated_frames,
            "trajectory": cam_traj,
            "num_frames": len(cam_traj),
            "target_poses": target_cam2world,
            "target_intrs": zoomed_intrinsics,
            "target_mask": target_mask,
            "model_dir": self.model_dir,
            "reconstructor_path": self.reconstructor_path,
            "lora_path": self.lora_path,
            "default_lora_name": DEFAULT_NEOVERSE_LORA_NAME,
        }
        if return_dict:
            return result
        return result["video"]
