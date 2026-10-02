# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Public WorldCrafter Base/Fast pipeline using shared base-model components."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.pipelines.pipeline_utils import PipelineABC


class WorldCrafterPipeline(PipelineABC):
    """Camera-controlled I2V/T2V with 33-frame chunks and RepEncoder memory."""

    MODEL_ID = "worldcrafter"
    MODEL_PATH_OPTION = "checkpoint_path"

    def __init__(self, model_path: Path, *, model_type: str, device: str, options: dict):
        super().__init__(model_id=f"worldcrafter-{model_type}", device=device, **options)
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.options = options
        self._runtime = None

    @classmethod
    def from_pretrained(
        cls, model_path: Any = None, required_components: Mapping | None = None,
        device: str = "cuda", model_id: str | None = None, **kwargs: Any,
    ) -> "WorldCrafterPipeline":
        from worldfoundry.core.io.paths import resolve_local_hf_model_path

        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        if model_path is not None and not isinstance(model_path, Mapping):
            options["checkpoint_path"] = model_path
        options.update(dict(required_components or {}))
        options.update(kwargs)
        cls._strip_framework_loading_options(options)
        for key in ("runtime_profile", "variant_id", "pipeline_binding", "repo_root"):
            options.pop(key, None)
        model_type = options.pop("model_type", "fast" if model_id == "worldcrafter-fast" else "base")
        if model_type not in {"base", "fast"}:
            raise ValueError("WorldCrafter model_type must be 'base' or 'fast'")
        checkpoint = options.pop("checkpoint_path", options.pop("model_path", options.pop("pretrained_model_path", None)))
        checkpoint = checkpoint or f"TencentARC/WorldCrafter-{model_type.title()}"
        path = Path(resolve_local_hf_model_path(str(checkpoint), required_files=("inference_config.json",)))
        allowed = {"height", "width", "seed", "memory_fov_h_deg", "memory_fov_v_deg",
                   "memory_fov_samples_per_axis", "attention_backend", "enable_compile", "shared_model_path"}
        unknown = set(options) - allowed
        if unknown:
            raise TypeError(f"Unsupported WorldCrafter loading options: {sorted(unknown)}")
        if (options.get("height", 384), options.get("width", 640)) != (384, 640):
            raise ValueError("Released WorldCrafter weights require height=384, width=640")
        return cls(path, model_type=model_type, device=device, options=options)

    def __call__(
        self, prompt: str = "", images: Any = None, video: Any = None,
        output_path: Any = None, return_dict: bool = False, *,
        camera_path: Any = None, actions: str | None = None, actions_file: Any = None,
        mode: str | None = None, **kwargs: Any,
    ) -> Any:
        import numpy as np
        from worldfoundry.core.media.processing.image_utils import load_pil_image
        from worldfoundry.synthesis.visual_generation.worldcrafter.camera import (
            parse_trajectory, build_trajectory, save_trajectory,
        )
        from worldfoundry.synthesis.visual_generation.worldcrafter.runtime import WorldCrafter, load_camera

        kwargs.pop("operator_kwargs", None)
        image_path = kwargs.pop("image_path", None)
        if images is not None and image_path is not None:
            raise ValueError("Provide images or image_path, not both")
        images = images if images is not None else image_path
        if video is not None:
            raise ValueError("WorldCrafter's released entry supports I2V/T2V; use Base resume_from for continuation")
        mode = mode or ("i2v" if images is not None else "t2v")
        if mode not in {"i2v", "t2v"} or (mode == "i2v") != (images is not None):
            raise ValueError("I2V requires one image; T2V must not receive images")
        if sum(x is not None for x in (camera_path, actions, actions_file)) != 1:
            raise ValueError("Provide exactly one of camera_path, actions or actions_file")
        if not output_path:
            raise ValueError("output_path is required")
        output = Path(output_path).expanduser().resolve()
        inputs = output.parent / (output.stem + "_inputs")
        if actions_file is not None:
            actions = Path(actions_file).read_text(encoding="utf-8")
        if actions is not None:
            events, controls = parse_trajectory(actions)
            poses, records = build_trajectory(events, **controls)
            camera_path = save_trajectory(inputs, poses, records, events=events, options=controls)
        elif isinstance(camera_path, (str, Path)):
            camera_path = Path(camera_path).expanduser().resolve()
        else:
            inputs.mkdir(parents=True, exist_ok=True)
            poses = np.asarray(camera_path)
            camera_path = inputs / "camera.npy"
            np.save(camera_path, poses, allow_pickle=False)
        load_camera(camera_path, kwargs.get("num_chunks"))
        if images is not None:
            if isinstance(images, (list, tuple)):
                if len(images) != 1:
                    raise ValueError("WorldCrafter I2V accepts exactly one image")
                images = images[0]
            if isinstance(images, (str, Path)):
                images = Path(images).expanduser().resolve()
                if not images.is_file():
                    raise FileNotFoundError(images)
            else:
                inputs.mkdir(parents=True, exist_ok=True)
                image = load_pil_image(images)
                images = inputs / "image.png"
                image.save(images)
        for key in ("chunk_output_dir", "state_output_dir", "resume_from", "local_camera_path"):
            if kwargs.get(key) is not None:
                kwargs[key] = Path(kwargs[key]).expanduser().resolve()
        for key, expected in (("height", 384), ("width", 640)):
            if int(kwargs.pop(key, expected)) != expected:
                raise ValueError("Released WorldCrafter weights require height=384, width=640")
        if kwargs.pop("model_type", self.model_type) != self.model_type:
            raise ValueError("Select model_type when loading the pipeline")
        if int(kwargs.get("fps", 16)) < 1:
            raise ValueError("fps must be positive")
        if self.model_type == "fast":
            if kwargs.get("num_inference_steps", 6) != 6 or kwargs.get("guidance_scale", 1.0) != 1.0:
                raise ValueError("Fast requires six contract-owned steps and CFG=1")
        import inspect
        allowed = set(inspect.signature(WorldCrafter.generate).parameters) - {"self", "mode", "camera_path", "image_path", "prompt", "output_path"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(f"Unsupported WorldCrafter generation options: {sorted(unknown)}")
        kwargs.setdefault("negative_prompt", "")
        kwargs.setdefault("seed", self.options.get("seed", 42))
        if self._runtime is None:
            self._runtime = WorldCrafter.from_pretrained(
                self.model_path, model_type=self.model_type, device=self.device, **self.options,
            )
        result = self._runtime.generate(
            mode=mode, camera_path=camera_path, image_path=images,
            prompt=prompt, output_path=output, **kwargs,
        )
        payload = {"status": "success", "model_id": f"worldcrafter-{self.model_type}",
                   "video": str(result.video_path), "artifact_path": str(result.video_path),
                   "metadata_path": str(result.summary_path), "metadata": result.summary}
        return payload if return_dict else payload["video"]


__all__ = ["WorldCrafterPipeline"]
