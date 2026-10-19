# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Native DreamDojo 2B GR-1 action-conditioned video generation."""

from __future__ import annotations

import math
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.io import write_json
from worldfoundry.core.io.paths import hfd_root_path

DEFAULT_DREAMDOJO_CHECKPOINTS = hfd_root_path() / "nvidia--DreamDojo" / "2B_GR1_post-train"


class DreamDojoRuntime:
    """Generate 13-frame robot-view chunks from normalized 384-D actions."""

    MODEL_ID = "dreamdojo"
    DISPLAY_NAME = "DreamDojo"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        checkpoints_dir: str | Path = DEFAULT_DREAMDOJO_CHECKPOINTS,
        native_pipeline: Any = None,
        variant: str | None = None,
        student_checkpoint_path: str | Path | None = None,
        **options: Any,
    ) -> None:
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "robot_world_model"
        self.device = device
        self.checkpoints_dir = Path(checkpoints_dir).expanduser()
        self.native_pipeline = native_pipeline
        self.variant = variant or ("student" if model_id == "dreamdojo-student" else "teacher")
        if self.variant not in {"teacher", "student"}:
            raise ValueError("DreamDojo variant must be teacher or student")
        self.student_checkpoint_path = student_checkpoint_path
        self.options = options
        self._realtime_session: Any = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> DreamDojoRuntime:
        """Configure native loading; no weights are loaded until inputs validate."""
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoints_dir"] = pretrained_model_path
        options.update(kwargs)
        options.setdefault("model_id", model_id or cls.MODEL_ID)
        options["device"] = device or options.get("device", "cuda")
        return cls(**options)

    def _checkpoint_path(self) -> Path:
        if self.variant == "student":
            if self.student_checkpoint_path is None:
                raise ValueError(
                    "DreamDojo student requires an explicit student_checkpoint_path; teacher weights are not a fallback"
                )
            root = Path(self.student_checkpoint_path).expanduser().resolve()
            if root.is_file():
                return root
            if (root / ".metadata").is_file():
                from .convert_distcp import convert_distcp

                return convert_distcp(root, root.parent)
            checkpoint = root / "model_ema_bf16.pt"
            if checkpoint.is_file():
                return checkpoint
            if (root / "model" / ".metadata").is_file():
                from .convert_distcp import convert_distcp

                return convert_distcp(root / "model", root)
            raise FileNotFoundError(f"DreamDojo distilled student checkpoint is missing: {root}")
        root = self.checkpoints_dir.resolve()
        if root.is_file():
            return root
        latest = root / "latest_checkpoint.txt"
        if latest.is_file():
            root = root / latest.read_text(encoding="utf-8").strip()
        elif (root / "iter_000050000").is_dir():
            root = root / "iter_000050000"
        checkpoint = root / "model_ema_bf16.pt"
        if not checkpoint.is_file() and (root / "model").is_dir():
            from .convert_distcp import convert_distcp

            convert_distcp(root / "model", root)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DreamDojo converted EMA checkpoint is missing: {checkpoint}")
        return checkpoint

    def _load_pipeline(self) -> Any:
        if self.native_pipeline is None:
            import torch

            from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline, RuntimePolicy
            from worldfoundry.base_models.diffusion_model.optimizations.policy import parse_offload_policy

            overrides = dict(self.options.get("checkpoint_overrides") or {})
            overrides.setdefault("transformer", str(self._checkpoint_path()))
            if self.options.get("vae_path") is not None:
                overrides["vae"] = str(self.options["vae_path"])
            if self.options.get("text_encoder_path") is not None:
                overrides.update(
                    {role: str(self.options["text_encoder_path"]) for role in ("text-encoder", "tokenizer")}
                )
            component_options = dict(self.options.get("component_options") or {})
            if self.options.get("text_embeddings_path") is not None:
                component_options["conditioner:main"] = {"text_embeddings_path": self.options["text_embeddings_path"]}
            self.native_pipeline = NativeDiffusionPipeline.from_pretrained(
                "dreamdojo-student" if self.variant == "student" else "dreamdojo",
                policy=RuntimePolicy(
                    device=torch.device(self.device),
                    dtype=torch.bfloat16,
                    offload=parse_offload_policy(self.options.get("offload_mode", "none")),
                ),
                checkpoint_overrides=overrides,
                component_options=component_options,
            )
        return self.native_pipeline

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] = (),
        output_path: str | Path | None = None,
        fps: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute native inference and export generated pixels, not a plan.

        Actions are already normalized unified 384-D controls. Longer sequences
        consist of complete 12-action chunks; each chunk observes the previous
        chunk's final generated frame. Dataset normalization is caller-owned.
        """
        if self.variant == "student":
            return self._predict_student(prompt, images, video, interactions, output_path, fps, **kwargs)
        import torch
        from PIL import Image

        from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
        from worldfoundry.base_models.diffusion_model.models.encoders.dreamdojo import robot_actions
        from worldfoundry.core.utils import load_pil_image

        if video is not None:
            raise ValueError("DreamDojo 2B GR-1 accepts one initial image and normalized robot actions")
        image = images if images is not None else kwargs.get("ref_image_path")
        if isinstance(image, (list, tuple)):
            if len(image) != 1:
                raise ValueError("DreamDojo requires exactly one initial image")
            image = image[0]
        if image is None:
            raise ValueError("DreamDojo requires an initial image")
        image = load_pil_image(image)
        value = kwargs.get("actions", kwargs.get("robot_action_sequence", interactions))
        if kwargs.get("action_file") is not None:
            value = kwargs["action_file"]
        if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (str, Path)):
            value = value[0]
        actions = robot_actions(value)
        action_count = actions.shape[1]
        if action_count % 12:
            raise ValueError("DreamDojo requires complete chunks of 12 normalized 384-D actions")
        height, width = kwargs.get("height", 480), kwargs.get("width", 640)
        for value in (height, width):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 16:
                raise ValueError("DreamDojo height and width must be positive integer multiples of 16")
        steps = kwargs.get("num_inference_steps", 35)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("DreamDojo num_inference_steps must be a positive integer")
        sampling = SamplingConfig(
            num_inference_steps=steps,
            guidance_scale=float(kwargs.get("guidance_scale", 7.0)),
            seed=int(kwargs.get("seed", 42)),
        )
        rate = fps if fps is not None else 5
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("DreamDojo fps must be finite and positive")
        if not math.isfinite(sampling.guidance_scale):
            raise ValueError("DreamDojo guidance_scale must be finite")
        pipeline = self._load_pipeline()
        chunks = []
        for index, offset in enumerate(range(0, action_count, 12)):
            request = DiffusionRequest(
                prompt=prompt,
                negative_prompt=kwargs.get("negative_prompt"),
                height=height,
                width=width,
                num_frames=13,
                sampling=SamplingConfig(
                    num_inference_steps=sampling.num_inference_steps,
                    guidance_scale=sampling.guidance_scale,
                    seed=sampling.seed + index,
                ),
                inputs={"image": image, "actions": actions[:, offset : offset + 12], "fps": rate},
            )
            output = pipeline(request)
            pixels = output.sample
            if not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != (1, 3, 13, height, width):
                raise RuntimeError("DreamDojo decoder must return generated [1,3,13,H,W] pixels")
            chunks.append(pixels.detach().cpu() if index == 0 else pixels[:, :, 1:].detach().cpu())
            final = pixels[0, :, -1].detach().float().cpu().clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0)
            image = Image.fromarray(final.numpy())
        pixels = torch.cat(chunks, dim=2)
        return self._export_pixels(
            pixels, output_path, fps=rate, seed=sampling.seed, output_dir=kwargs.get("output_dir")
        )

    def _export_pixels(
        self, pixels: Any, output_path: str | Path | None, *, fps: float, seed: int, **kwargs: Any
    ) -> dict[str, Any]:
        """Export generated teacher or student pixels through the shared video codec."""
        from worldfoundry.core.media.codecs.video import save_image_or_video_tensor
        from worldfoundry.core.media.processing.postprocess import VideoChunk

        target = (
            Path(output_path or Path(kwargs.get("output_dir") or ".") / "dreamdojo_world.mp4").expanduser().resolve()
        )
        artifact = target.with_suffix(".mp4") if target.suffix == ".json" else target
        report = target if target.suffix == ".json" else target.with_suffix(".json")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        save_image_or_video_tensor(pixels, str(artifact), fps=fps)
        payload = {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_world",
            "artifact_path": str(artifact),
            "artifact_files": [str(artifact)],
            "metadata_path": str(report),
            "runtime": "worldfoundry-native-diffusion",
            "backend_quality": "in_tree_runtime",
            "num_frames": pixels.shape[2],
            "fps": fps,
            "action_dimension": 384,
            "seed": seed,
            "variant": self.variant,
        }
        write_json(report, payload)
        return {
            **payload,
            "video_chunk": VideoChunk(frames=pixels, layout="bcthw", value_range="-1,1", metadata={"fps": fps}),
        }

    def _student_session(self) -> Any:
        """Build the resident student adapter without allocating teacher state."""
        if self.variant != "student":
            raise ValueError("DreamDojo realtime requires variant='student' and distilled student weights")
        if self._realtime_session is None:
            from .realtime import DreamDojoStudentSession

            self._realtime_session = DreamDojoStudentSession(self._load_pipeline())
        return self._realtime_session

    def prepare_realtime(self) -> dict[str, Any]:
        """Load resident student components and advertise native frame/action cadence."""
        return {"realtime_spec": self._student_session().realtime_spec().to_payload()}

    def configure_realtime(self, images: Any, prompt: str = "", **kwargs: Any) -> dict[str, Any]:
        """Configure one image/text/seed context before consuming normalized actions."""
        from worldfoundry.core.utils import load_pil_image

        from .realtime import student_sampling_options

        options = {
            key: kwargs[key]
            for key in ("height", "width", "num_inference_steps", "guidance_scale", "fps", "seed")
            if key in kwargs
        }
        student_sampling_options(**{key: value for key, value in options.items() if key != "seed"})
        image = load_pil_image(images)
        return self._student_session().configure(image, prompt, **options)

    def stream_realtime(self, interactions: Any = None, *, actions: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Consume 12 new actions first, then 4 per STEP, without restarting the rollout."""
        del kwargs
        return self._student_session().generate(actions if actions is not None else interactions)

    def realtime_next_output_frames(self) -> int:
        """Report the first/steady output budget for the serving scheduler."""
        return self._student_session().next_output_frames()

    def reset_realtime(self) -> None:
        """Release only session history, retaining all loaded student components."""
        if self._realtime_session is not None:
            self._realtime_session.reset()

    def _predict_student(
        self,
        prompt: str,
        images: Any,
        video: Any,
        interactions: Sequence[Any],
        output_path: str | Path | None,
        fps: float | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run an isolated distilled-student rollout and export all generated frames."""
        import torch

        chunks = [
            result["video_chunk"].frames.detach().cpu()
            for result in self.stream_student(
                prompt=prompt,
                images=images,
                video=video,
                interactions=interactions,
                fps=fps,
                **kwargs,
            )
        ]
        return self._export_pixels(
            torch.cat(chunks, dim=2),
            output_path,
            fps=10,
            seed=int(kwargs.get("seed", 42)),
            output_dir=kwargs.get("output_dir"),
        )

    def stream_student(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Any = (),
        fps: float | None = None,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, None]:
        """Yield isolated native student chunks; closing releases only that rollout."""
        if self.variant != "student":
            raise ValueError("DreamDojo chunk streaming requires the distilled student")
        from worldfoundry.base_models.diffusion_model.models.encoders.dreamdojo import robot_actions
        from worldfoundry.core.utils import load_pil_image

        from .realtime import DreamDojoStudentSession, student_sampling_options

        if video is not None:
            raise ValueError("DreamDojo student requires one initial image")
        image = images if images is not None else kwargs.get("ref_image_path")
        if isinstance(image, (list, tuple)):
            if len(image) != 1:
                raise ValueError("DreamDojo student requires exactly one initial image")
            image = image[0]
        if image is None:
            raise ValueError("DreamDojo student requires an initial image")
        image = load_pil_image(image)
        value = kwargs.get("action_file", kwargs.get("actions", kwargs.get("robot_action_sequence", interactions)))
        if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (str, Path)):
            value = value[0]
        actions = robot_actions(value)
        count = actions.shape[1]
        if count < 12 or count % 4:
            raise ValueError("DreamDojo student requires 12 initial actions followed by complete 4-action chunks")
        options = {
            "height": kwargs.get("height", 480),
            "width": kwargs.get("width", 640),
            "num_inference_steps": kwargs.get("num_inference_steps", 4),
            "guidance_scale": kwargs.get("guidance_scale", 1.0),
            "fps": 10 if fps is None else fps,
        }
        student_sampling_options(**options)
        seed = int(kwargs.get("seed", 42))
        session = DreamDojoStudentSession(self._load_pipeline())
        try:
            session.configure(image, prompt, seed=seed, **options)
            for start, end in [(0, 12), *((index, index + 4) for index in range(12, count, 4))]:
                yield session.generate(actions[:, start:end])
        finally:
            session.reset()
