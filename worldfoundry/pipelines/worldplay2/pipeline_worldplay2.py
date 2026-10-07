"""WorldPlay2 public boundary reuses the common native visual adapter."""

from __future__ import annotations

from pathlib import Path

from ..native_diffusion import NativeVisualDiffusionPipeline


class WorldPlay2Pipeline(NativeVisualDiffusionPipeline):
    MODEL_ID = "worldplay2"
    OWNER = "WorldPlay2"
    GENERATION_TYPE = "i2v"
    CHECKPOINT_ROLES = ("high", "low", "t5", "tokenizer", "vae")
    ACCEPTS_IMAGES = True
    REQUIRES_IMAGES = True
    ACCEPTS_INTERACTIONS = True
    DEFAULT_HEIGHT = 448
    DEFAULT_WIDTH = 832
    DEFAULT_NUM_FRAMES = 125
    DEFAULT_NUM_INFERENCE_STEPS = 4
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_FPS = 16
    DEFAULT_NEGATIVE_PROMPT = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
        "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )
    OUTPUT_LAYOUT = "FHWC"
    OUTPUT_VALUE_RANGE = "0,1"
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 7.0}
    SCHEDULER_OPTION_ALIASES = {"sample_shift": "shift"}
    REQUEST_INPUT_DEFAULTS = {
        "actions": None, "perspective": "tps", "prompts": None, "prompt_event": None,
        "chunk_length": 4, "sink_size": 1, "temporal_size": 1,
        "yaw_rotation_speed_deg": 3.0, "pitch_rotation_speed_deg": 1.0,
    }
    REQUEST_INPUT_ALIASES = {"action": "actions"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model_id in {"worldplay2-ar", "worldplay2-bi"}:
            self.DEFAULT_NUM_INFERENCE_STEPS = 40
            self.DEFAULT_GUIDANCE_SCALE = 3.5
            self.DEFAULT_SCHEDULER_OPTIONS = {"shift": 5.0}
            self.REQUEST_INPUT_DEFAULTS = {
                **type(self).REQUEST_INPUT_DEFAULTS,
                "chunk_length": 32 if self.model_id == "worldplay2-bi" else 4,
                "pitch_rotation_speed_deg": 3.0,
            }

    @classmethod
    def _requested_model_id(cls, options):
        modes = {"few_step": "worldplay2", "fast": "worldplay2", "ar": "worldplay2-ar", "bi": "worldplay2-bi"}
        mode = options.get("mode")
        if mode is not None:
            if str(mode) not in modes:
                raise ValueError("WorldPlay2 mode must be 'few_step', 'ar', or 'bi'")
            return modes[str(mode)]
        selected = str(options.get("variant", options.get("variant_id", options.get("model_id", cls.MODEL_ID))))
        aliases = {"worldplay2-fast": "worldplay2", **modes,
                   "aejion/WorldPlay2-Fast": "worldplay2", "aejion/WorldPlay2-AR": "worldplay2-ar",
                   "aejion/WorldPlay2-BI": "worldplay2-bi"}
        selected = aliases.get(selected, selected)
        if selected not in {"worldplay2", "worldplay2-ar", "worldplay2-bi"}:
            raise ValueError(f"unsupported WorldPlay2 variant: {selected!r}")
        return selected

    @classmethod
    def _checkpoint_overrides(cls, model_path, options):
        explicit = options.get("checkpoint_overrides")
        if explicit is not None:
            return super()._checkpoint_overrides(model_path, options)
        source = options.get("checkpoint_path", options.get("checkpoint_dir", options.get("model_path", model_path)))
        overrides = {}
        if isinstance(source, (str, Path)):
            overrides.update(high=str(source), low=str(source))
        base = options.get("base_model_path")
        if base is not None:
            overrides.update(t5=str(base), tokenizer=str(base), vae=str(base))
        for option, role in (("high_noise_ckpt", "high"), ("low_noise_ckpt", "low"),
                             ("vae_path", "vae"), ("text_encoder_path", "t5"), ("tokenizer_path", "tokenizer")):
            if options.get(option) is not None:
                overrides[role] = str(options[option])
        return overrides or None
