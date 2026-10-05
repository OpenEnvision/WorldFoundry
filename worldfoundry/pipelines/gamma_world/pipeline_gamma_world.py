"""Native WorldFoundry pipeline for Gamma-World."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.models.encoders.gamma_world import component as gamma_conditioning
from worldfoundry.base_models.diffusion_model.optimizations import (
    RuntimePolicy,
    parse_offload_policy,
    parse_torch_dtype,
)
from worldfoundry.pipelines.cosmos.reason1_options import reason1_component_options
from worldfoundry.pipelines.native_diffusion_video import NativeTextToVideoPipeline

_MODE_TO_MODEL_ID = {
    "causal_few_step": "gamma-world-causal-few-step",
    "causal-few-step": "gamma-world-causal-few-step",
    "causal": "gamma-world-causal",
    "bidirectional": "gamma-world-bidirectional",
}


class GammaWorldPipeline(NativeTextToVideoPipeline):
    """Generate synchronized two-player rollouts through native diffusion infra."""

    MODEL_ID = "gamma-world"
    OWNER = "Gamma-World"
    CHECKPOINT_ROLES = ("transformer", "vae")
    GENERATION_TYPE = "i2v"
    ACCEPTS_IMAGES = True
    REQUIRES_IMAGES = True
    DEFAULT_HEIGHT = 320
    DEFAULT_WIDTH = 480
    DEFAULT_NUM_FRAMES = 189
    DEFAULT_NUM_INFERENCE_STEPS = 4
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_NEGATIVE_PROMPT = gamma_conditioning.DEFAULT_NEGATIVE_PROMPT
    DEFAULT_FPS = 16
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 5.0}
    REQUEST_INPUT_DEFAULTS = {
        "n_players": 2,
        "actions": None,
        "num_conditional_frames": 1,
    }
    REQUEST_INPUT_ALIASES = {"action_paths": "actions"}

    @classmethod
    def _validate_num_frames(cls, options: Mapping[str, Any]) -> None:
        for name in ("num_frames", "frame_num", "frames"):
            value = options.get(name)
            if value is not None:
                break
        else:
            value = cls.DEFAULT_NUM_FRAMES
        integer_value = (
            isinstance(value, Integral) and not isinstance(value, bool)
        ) or (
            isinstance(value, str) and value.strip().lstrip("+-").isdigit()
        )
        if not integer_value:
            raise ValueError(
                "Gamma-World num_frames must be the integer 189 per view "
                "(official state_t=48 latent frames); "
                f"received {value!r}"
            )
        if int(value) != 189:
            raise ValueError(
                "Gamma-World released checkpoints require num_frames=189 per view "
                "(official state_t=48 latent frames); "
                f"received {value!r}"
            )

    @classmethod
    def preflight_generation_defaults(
        cls,
        generation_defaults: Mapping[str, Any] | None,
        *,
        model_path: str | Mapping[str, Any] | None = None,
    ) -> None:
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(generation_defaults or {})
        cls._validate_num_frames(options)

    @classmethod
    def plan(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        cls._validate_num_frames(cls._options(model_path, required_components, kwargs))
        return super().plan(model_path, required_components, **kwargs)

    @classmethod
    def _checkpoint_overrides(
        cls,
        model_path: str | Mapping[str, Any] | None,
        options: Mapping[str, Any],
    ) -> dict[str, str] | None:
        overrides = dict(super()._checkpoint_overrides(model_path, options) or {})
        reason = options.get("text_encoder_path", options.get("reason1_path"))
        if isinstance(reason, (str, Path)):
            overrides["text-encoder"] = str(reason)
            overrides["text-tokenizer"] = str(reason)
        return overrides or None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "GammaWorldPipeline":
        options = cls._options(model_path, required_components, kwargs)
        cls._validate_num_frames(options)
        component_options = dict(cls._component_options(options) or {})
        encoder_options = dict(component_options.get("conditioner:main", {}))
        resolved_encoder_options = reason1_component_options(options)
        for name, value in resolved_encoder_options.items():
            public_name = "text_encoder_run_on_cpu" if name == "run_on_cpu" else f"text_{name}"
            if public_name in options or name not in encoder_options:
                encoder_options[name] = value
        component_options["conditioner:main"] = encoder_options
        from worldfoundry.core.model_loading.text_embeddings import validate_prompt_encoder_options

        validate_prompt_encoder_options(
            run_on_cpu=encoder_options["run_on_cpu"],
            embedding_cache_size=encoder_options["embedding_cache_size"],
            embedding_cache_max_bytes=encoder_options["embedding_cache_max_bytes"],
        )
        mode = str(options.pop("mode", "causal_few_step")).strip().lower()
        try:
            recipe_model_id = _MODE_TO_MODEL_ID[mode]
        except KeyError as error:
            raise ValueError(f"unsupported Gamma-World mode: {mode!r}") from error
        native = NativeDiffusionPipeline.from_pretrained(
            recipe_model_id,
            policy=RuntimePolicy(
                device=device,
                dtype=parse_torch_dtype(
                    options.get("torch_dtype", options.get("weight_dtype", options.get("dtype"))),
                    owner=cls.OWNER,
                ),
                offload=parse_offload_policy(
                    options.get("offload_mode", "block"),
                    allow_disk=False,
                    owner=cls.OWNER,
                ),
            ),
            checkpoint_overrides=cls._checkpoint_overrides(model_path, options),
            component_options=component_options,
            extensions=options.get("extensions", ()),
        )
        pipeline = cls(native_pipeline=native, device=device)
        pipeline.recipe_model_id = recipe_model_id
        if recipe_model_id != "gamma-world-causal-few-step":
            pipeline.DEFAULT_NUM_INFERENCE_STEPS = 35
            pipeline.DEFAULT_GUIDANCE_SCALE = 5.0
        return pipeline

    @staticmethod
    def _sample_directory(value: object) -> Path | None:
        if not isinstance(value, (str, Path)):
            return None
        root = Path(value).expanduser()
        if root.is_dir() and (root / "first_frame.png").is_file():
            return root
        return None

    def __call__(
        self,
        prompt: str | list[str] = "",
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        **kwargs: Any,
    ) -> Any:
        requested_mode = kwargs.pop("mode", None)
        if requested_mode is not None:
            normalized_mode = str(requested_mode).strip().lower()
            try:
                requested_model_id = _MODE_TO_MODEL_ID[normalized_mode]
            except KeyError as error:
                raise ValueError(f"unsupported Gamma-World mode: {requested_mode!r}") from error
            if requested_model_id != self.recipe_model_id:
                raise ValueError(
                    "Gamma-World mode is selected while loading the pipeline; "
                    f"loaded {self.recipe_model_id!r}, received {requested_mode!r} at inference"
                )
        self._validate_num_frames(kwargs)
        guidance = kwargs.pop("guidance", None)
        if guidance is not None:
            if kwargs.get("guidance_scale") is not None:
                raise ValueError("pass either guidance or guidance_scale, not both")
            kwargs["guidance_scale"] = guidance
        sample_dir = self._sample_directory(kwargs.pop("eval_dir", None))
        if sample_dir is None:
            sample_dir = self._sample_directory(images)
        if sample_dir is not None:
            images = sample_dir / "first_frame.png"
            if not prompt:
                prompt_path = sample_dir / "prompt.txt"
                if not prompt_path.is_file():
                    raise FileNotFoundError(f"Gamma sample is missing {prompt_path}")
                prompt = prompt_path.read_text(encoding="utf-8").strip()
            if kwargs.get("actions") is None and kwargs.get("action_paths") is None:
                n_players = int(kwargs.get("n_players", 2))
                paths = [sample_dir / f"action_{index}.json" for index in range(n_players)]
                if not all(path.is_file() for path in paths) and n_players == 2:
                    paths = [sample_dir / "action_left.json", sample_dir / "action_right.json"]
                if all(path.is_file() for path in paths):
                    kwargs["actions"] = paths
        if interactions not in (None, (), []):
            if kwargs.get("actions") is not None or kwargs.get("action_paths") is not None:
                raise ValueError("pass Gamma actions through either interactions or actions, not both")
            kwargs["actions"] = interactions
        if self.recipe_model_id in {"gamma-world-causal", "gamma-world-causal-few-step"}:
            actions = kwargs.get("actions")
            if actions is None:
                actions = kwargs.get("action_paths")
            if actions is None or (isinstance(actions, (list, tuple)) and not actions):
                raise ValueError(
                    "Gamma-World causal checkpoints require an action track for each view; "
                    "pass actions/action_paths or use a sample directory with action JSONs"
                )
        return super().__call__(
            prompt=prompt,
            images=images,
            video=video,
            interactions=None,
            **kwargs,
        )


__all__ = ["GammaWorldPipeline"]
