"""Lyra-1 synthesis over WorldFoundry's native GEN3C recipe."""

from __future__ import annotations

import json
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


class Lyra1Runtime:
    """Generate the camera-controlled videos consumed by Lyra-1 reconstruction.

    Lyra-1's synthesis stage is GEN3C.  The released Lyra checkpoints are used
    by the optional 3D Gaussian representation stage, not by a second diffusion
    runtime, so this adapter delegates directly to the native GEN3C pipeline.
    """

    MODEL_ID = "lyra-1"
    DISPLAY_NAME = "Lyra-1"
    BLOCKED_REASONS: tuple[str, ...] = ()
    MULTI_TRAJECTORY_INDEX = {
        "left": 0,
        "right": 1,
        "up": 2,
        "down": 2,
        "zoom_out": 3,
        "zoom_in": 4,
        "clockwise": 5,
        "counterclockwise": 5,
    }
    _MULTI_TRAJECTORIES = {
        "left": (0, (0.2, 0.3)),
        "right": (1, (0.2, 0.3)),
        "up": (2, (0.1, 0.2)),
        "zoom_out": (3, (0.3, 0.4)),
        "zoom_in": (4, (0.3, 0.4)),
        "clockwise": (5, (0.4, 0.6)),
    }

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        device: str = "cuda",
        defaults: dict[str, Any] | None = None,
        model_id: str = MODEL_ID,
    ) -> None:
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "image_or_video_to_world_video"
        self.checkpoint_dir = None if checkpoint_dir is None else str(Path(checkpoint_dir).expanduser())
        self.device = str(device)
        self.defaults = dict(defaults or {})
        self._pipeline: Any = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        checkpoint_dir: str | None = None,
        default_mode: str = "static",
        **kwargs: Any,
    ) -> "Lyra1Runtime":
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        options.update(kwargs)
        checkpoint_value = checkpoint_dir or options.get("checkpoint_dir")

        from worldfoundry.synthesis.visual_generation.gen3c.runtime_env import (
            prepare_gen3c_checkpoint_root,
        )

        resolved = prepare_gen3c_checkpoint_root(checkpoint_value)
        return cls(
            checkpoint_dir=resolved,
            device=str(device or options.get("device") or "cuda"),
            defaults={"default_mode": default_mode, **options},
            model_id=str(options.get("model_id") or options.get("profile_id") or cls.MODEL_ID),
        )

    def _native_pipeline(self):
        if self._pipeline is None:
            from worldfoundry.pipelines.gen3c.pipeline_gen3c import Gen3CPipeline

            self._pipeline = Gen3CPipeline.from_pretrained(
                self.checkpoint_dir,
                device=self.device,
                torch_dtype=self.defaults.get("torch_dtype", self.defaults.get("weight_dtype")),
                offload_mode=self.defaults.get("offload_mode", "block"),
            )
        return self._pipeline

    @staticmethod
    def _clip_name(visual_input: Any) -> str:
        if isinstance(visual_input, (str, Path)):
            return Path(visual_input).stem or "lyra1"
        return "lyra1"

    @staticmethod
    def _dynamic_reference(visual_input: Any) -> Any:
        from worldfoundry.core.io import coerce_video_frames

        frames = coerce_video_frames(visual_input)
        if not frames:
            raise ValueError("Lyra-1 dynamic input contains no frames")
        return frames[0]

    @staticmethod
    def _save_reconstruction_inputs(
        result: Mapping[str, Any],
        *,
        output_root: Path,
        clip_name: str,
    ) -> None:
        artifacts = dict(result.get("artifacts") or {})
        camera_to_world = artifacts.get("camera_to_world")
        camera_intrinsics = artifacts.get("camera_intrinsics")
        if camera_to_world is not None:
            poses = torch.as_tensor(camera_to_world).detach().float().cpu()
            if poses.ndim == 4:
                poses = poses[0]
            pose_dir = output_root / "pose"
            pose_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                pose_dir / f"{clip_name}.npz",
                data=poses.numpy(),
                inds=np.arange(poses.shape[0]),
            )
        if camera_intrinsics is not None:
            intrinsics = torch.as_tensor(camera_intrinsics).detach().float().cpu()
            if intrinsics.ndim == 4:
                intrinsics = intrinsics[0]
            if intrinsics.shape[-2:] == (3, 3):
                intrinsics = torch.stack(
                    (
                        intrinsics[:, 0, 0],
                        intrinsics[:, 1, 1],
                        intrinsics[:, 0, 2],
                        intrinsics[:, 1, 2],
                    ),
                    dim=-1,
                )
            intrinsics_dir = output_root / "intrinsics"
            intrinsics_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                intrinsics_dir / f"{clip_name}.npz",
                data=intrinsics.numpy(),
                inds=np.arange(intrinsics.shape[0]),
            )
        latents = result.get("latents")
        if isinstance(latents, torch.Tensor):
            latent_value = latents.detach().float().cpu()
            if latent_value.ndim == 5:
                latent_value = latent_value.permute(0, 2, 1, 3, 4).contiguous()
            latent_dir = output_root / "latent"
            latent_dir.mkdir(parents=True, exist_ok=True)
            torch.save(latent_value.numpy(), latent_dir / f"{clip_name}.pkl")

    def _write_plan(
        self,
        *,
        plan_path: str | Path | None,
        output_root: Path,
        mode: str,
        prompt: str,
        trajectory: str,
        visual_input: Any,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        target = Path(plan_path).expanduser() if plan_path is not None else output_root / "lyra1_plan.json"
        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "planned",
            "model_id": self.model_id,
            "backend": "worldfoundry-native-diffusion",
            "recipe": "gen3c-cosmos1-7b",
            "checkpoint_dir": self.checkpoint_dir,
            "mode": mode,
            "prompt": prompt,
            "trajectory": trajectory,
            "input": str(visual_input) if isinstance(visual_input, (str, Path)) else "<in-memory>",
            "output_root": str(output_root),
            "options": {key: str(value) for key, value in options.items()},
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "status": "planned",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": str(target),
            "runtime": payload["backend"],
            "backend_quality": "native_recipe",
            "blocked_reasons": [],
        }

    def predict(
        self,
        visual_input: Any = None,
        mode: str = "static",
        prompt: str = "",
        trajectory: str = "zoom_in",
        output_root: str | None = None,
        checkpoint_dir: str | None = None,
        num_video_frames: int | None = None,
        fps: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
        guidance: float | None = None,
        num_gpus: int | None = None,
        movement_distance: float | None = None,
        camera_rotation: str | None = None,
        multi_trajectory: bool = False,
        total_movement_distance_factor: float | None = None,
        vipe_path: str | None = None,
        vipe_starting_frame_idx: int | None = None,
        filter_points_threshold: float | None = None,
        foreground_masking: bool | None = None,
        center_depth_quantile: bool | None = None,
        flip_supervision: bool | None = None,
        offload_diffusion_transformer: bool | None = None,
        offload_tokenizer: bool | None = None,
        offload_text_encoder_model: bool | None = None,
        offload_prompt_upsampler: bool | None = None,
        offload_guardrail_models: bool | None = None,
        disable_prompt_encoder: bool | None = None,
        disable_guardrail: bool | None = None,
        show_progress: bool = True,
        execute: bool = False,
        plan_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del (
            checkpoint_dir,
            vipe_starting_frame_idx,
            filter_points_threshold,
            foreground_masking,
            flip_supervision,
            offload_diffusion_transformer,
            offload_tokenizer,
            offload_text_encoder_model,
            offload_prompt_upsampler,
            offload_guardrail_models,
            disable_prompt_encoder,
            disable_guardrail,
            show_progress,
        )
        mode = str(mode or self.defaults.get("default_mode", "static")).lower()
        if mode not in {"static", "dynamic"}:
            raise ValueError(f"Unsupported Lyra-1 mode: {mode}")
        frames = int(num_video_frames or self.defaults.get("num_video_frames", 121))
        if frames != 121:
            raise ValueError("The native GEN3C recipe currently generates exactly 121 frames")
        gpus = int(num_gpus or self.defaults.get("num_gpus", 1))
        if execute and gpus != 1:
            raise NotImplementedError("Lyra-1 native inference currently supports one process/device")

        output_path = Path(output_root or tempfile.mkdtemp(prefix=f"lyra1_{mode}_")).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        run_options = {
            "num_frames": frames,
            "fps": int(fps or self.defaults.get("fps", 24)),
            "height": int(height or self.defaults.get("height", 704)),
            "width": int(width or self.defaults.get("width", 1280)),
            "num_inference_steps": int(num_steps or self.defaults.get("num_steps", 35)),
            "seed": int(seed if seed is not None else self.defaults.get("seed", 1)),
            "guidance_scale": float(guidance if guidance is not None else self.defaults.get("guidance", 1.0)),
            "camera_rotation": str(camera_rotation or self.defaults.get("camera_rotation", "center_facing")),
            "movement_distance": float(
                movement_distance if movement_distance is not None else self.defaults.get("movement_distance", 0.3)
            ),
        }
        if center_depth_quantile:
            kwargs["center_depth_quantile"] = True
        if not execute:
            return self._write_plan(
                plan_path=plan_path,
                output_root=output_path,
                mode=mode,
                prompt=prompt,
                trajectory=trajectory,
                visual_input=vipe_path or visual_input,
                options={**run_options, **kwargs, "multi_trajectory": multi_trajectory},
            )

        warp_images = kwargs.pop("rendered_warp_images", None)
        warp_masks = kwargs.pop("rendered_warp_masks", None)
        if mode == "dynamic":
            if vipe_path is not None:
                raise NotImplementedError(
                    "VIPE/Cache4D preprocessing is not part of the native diffusion runtime; "
                    "provide rendered_warp_images and rendered_warp_masks instead"
                )
            if warp_images is None or warp_masks is None:
                raise NotImplementedError(
                    "Native Lyra-1 dynamic synthesis requires explicit rendered_warp_images "
                    "and rendered_warp_masks"
                )
            reference_image = kwargs.pop("reference_image", None)
            if reference_image is None:
                reference_image = self._dynamic_reference(visual_input)
        else:
            reference_image = visual_input
        if reference_image is None:
            raise ValueError("Lyra-1 requires a visual input")
        for compatibility_key in ("image_path", "input_path", "output_path", "task_type"):
            kwargs.pop(compatibility_key, None)
        if kwargs:
            supported = {"negative_prompt", "condition_augment_sigma", "center_depth", "center_depth_quantile"}
            unsupported = sorted(set(kwargs) - supported)
            if unsupported:
                raise TypeError(f"unsupported Lyra-1 inference options: {unsupported}")

        trajectory_specs: list[tuple[str, int | None, float]]
        if multi_trajectory:
            if mode == "dynamic":
                raise NotImplementedError("Dynamic rendered warps describe one trajectory at a time")
            rng = random.Random(run_options["seed"])
            factor = float(total_movement_distance_factor or 1.0)
            trajectory_specs = [
                (name, index, rng.uniform(*distance_range) * factor)
                for name, (index, distance_range) in self._MULTI_TRAJECTORIES.items()
            ]
        else:
            trajectory_specs = [(trajectory, None, run_options["movement_distance"])]

        pipeline = self._native_pipeline()
        clip_name = self._clip_name(visual_input)
        generated: dict[str, dict[str, Any]] = {}
        for selected_trajectory, trajectory_index, selected_distance in trajectory_specs:
            target_root = output_path if trajectory_index is None else output_path / str(trajectory_index)
            rgb_dir = target_root / "rgb"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            result = pipeline(
                images=reference_image,
                prompt=prompt,
                negative_prompt=kwargs.get("negative_prompt"),
                trajectory=selected_trajectory,
                output_path=rgb_dir / f"{clip_name}.mp4",
                return_dict=True,
                rendered_warp_images=warp_images,
                rendered_warp_masks=warp_masks,
                condition_augment_sigma=float(kwargs.get("condition_augment_sigma", 0.001)),
                center_depth=float(kwargs.get("center_depth", 1.0)),
                center_depth_quantile=bool(kwargs.get("center_depth_quantile", False)),
                **{**run_options, "movement_distance": selected_distance},
            )
            self._save_reconstruction_inputs(result, output_root=target_root, clip_name=clip_name)
            generated[selected_trajectory] = result

        primary = generated.get(trajectory) or next(iter(generated.values()))
        return {
            "status": "completed",
            "mode": mode,
            "prompt": prompt,
            "trajectory": trajectory,
            "generated_root": str(output_path),
            "generated_video_path": primary["generated_video_path"],
            "video": primary["video"],
            "fps": run_options["fps"],
            "input_path": str(visual_input) if isinstance(visual_input, (str, Path)) else "<in-memory>",
            "trajectories": generated,
        }


__all__ = ["Lyra1Runtime"]
