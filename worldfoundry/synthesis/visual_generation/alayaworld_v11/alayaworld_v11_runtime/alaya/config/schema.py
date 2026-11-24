from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

LTX_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, "
    "distorted proportions, unnatural skin tones, deformed facial features, "
    "asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, "
    "incorrect depth of field, background too sharp, background clutter, "
    "distracting reflections, harsh shadows, inconsistent lighting direction, "
    "color banding, cartoonish rendering, 3D CGI look, unrealistic materials, "
    "uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, "
    "wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, "
    "robotic voice, echo, background noise, off-sync audio, incorrect dialogue, "
    "added dialogue, repetitive speech, jittery movement, awkward pauses, "
    "incorrect timing, unnatural transitions, inconsistent framing, tilted camera, "
    "flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, "
    "or AI artifacts."
)


@dataclass
class RunConfig:
    name: str = "rollout_from_bcd"
    seed: int = 42
    output_dir: str = "./logs_rollout"
    log_dir: str = "./logs/rollout_from_bcd"


@dataclass
class PathsConfig:
    transformer: str = ""
    base_transformer: str = ""
    continue_transformer: str = ""
    resume_checkpoint: str | None = None
    resume_reset_step: bool = False
    dmd_resume: str | None = None
    real_score_model: str | None = None
    vae: str = ""
    gemma: str = ""
    history_encoder: str | None = None
    video_base_dir: str = ""
    annotation_base_dir: str = ""
    # ---- da3 inference release additions (defaults keep vigeo behavior unchanged) ----
    model: str = ""  # merged one-file checkpoint (DiT+VAE+text-enc+history-enc)
    merged_base_dir: str | None = None
    da3_repo: str | None = None
    da3_model: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    da3_cache: str | None = None
    taehv: str = ""

    @property
    def effective_transformer(self) -> str:
        return self.model or self.continue_transformer or self.transformer or self.base_transformer

    @property
    def effective_vae(self) -> str:
        # da3: VAE lives in the same merged one-file weights unless a separate `vae` is set.
        return self.vae or self.effective_transformer


@dataclass
class SampleConfig:
    height: int = 544
    width: int = 960
    fps: float = 24.0
    temporal_stride: int = 8
    spatial_stride: int = 32  # da3 inference addition (unused by vigeo paths)


@dataclass
class ConditionLayout:
    type: str = "nearby"
    i2v_prob: float = 0.9
    v2v_prob: float = 0.1
    v2v_ratio_min: float = 0.2
    v2v_ratio_max: float = 0.6


@dataclass
class OutputLayout:
    latent_frames: list[int] = field(default_factory=lambda: [8])
    probs: list[float] = field(default_factory=lambda: [1.0])


@dataclass
class LayoutConfig:
    sink_latent_frames: int = 1
    max_gap_sec: float = 20.0
    history_latent_frames: int = 60
    condition: ConditionLayout = field(default_factory=ConditionLayout)
    output: OutputLayout = field(default_factory=OutputLayout)
    # For K=8, sample window starts from the jsonl valid_k8_starts field; the dataloader pre-rolls
    # gap_steps and cond_mode so the output aligns exactly with video[s, s+8)
    k8_use_valid_starts: bool = False
    # K=4 variant; requires a valid_k4_starts field in the jsonl
    k4_use_valid_starts: bool = False
    # Load the sink frame from a random position far from the target instead of the frame just before it,
    # so the model must treat the sink as a global identity anchor rather than predicting the target
    # from a neighbouring frame, which strengthens action-conditioned learning.
    sink_remote: bool = False
    # Minimum latent distance between sink and target (8 latents is about 2.7s at 24fps, stride 8).
    sink_remote_min_distance: int = 8
    # Variable-length training: short segments train at their natural length
    # instead of being rejected as too short. Required for bidirectional pretraining, where most
    # segments are under 20s.
    # Keep False for autoregressive / self-rollout training, which needs long clips.
    variable_length: bool = False


@dataclass
class MemoryConfig:
    compress_t: int = 1
    compress_h: int = 2
    compress_w: int = 2
    lr_compress_t: int = 1
    lr_compress_h: int = 2
    lr_compress_w: int = 2
    gate_init: float = 0.5
    use_self_attn: bool = True
    use_lr_branch: bool = True
    train: bool = True
    drop_prob: float = 0.0


@dataclass
class SpatialMemoryConfig:
    enabled: bool = False
    context_mode: str = "retrieval"
    depth_backend: str = "constant"
    # ---- ViGeo 3D lifting (pointmap geometry; requires context_mode=vigeo_prefix_last_frame) ----
    vigeo_checkpoint: str = ""
    vigeo_device: str = "auto"
    vigeo_num_tokens: int = 1369
    vigeo_stream_chunk_size: int = 16
    vigeo_cache_budget: int = 262144
    vigeo_single_frame_scale: float = 1.0
    vigeo_generated_source_pose_mode: str = "cplan"
    vigeo_prefix_frames: int | None = None
    use_warped_context: bool = True
    retrieval_max_coverage: bool = True
    retrieval_depth_threshold: float = 0.1
    num_context_frames: int = 1
    require_full_context: bool = True
    retrieval_views: int = 1
    # Bank source ordering: "nearest" ranks by camera distance (suited to translation-dominated data);
    # "coverage" ranks by target coverage, which matters for in-place rotation where distances tie
    # and older frames that actually cover the target would otherwise never be selected.
    retrieval_sort: str = "nearest"
    # Orientation cost weight for ViGeo source pre-selection: score = translation distance + weight *
    # minimum viewing-angle difference in radians (0 disables the orientation term).
    retrieval_rotation_weight: float = 0.0
    # false: the target frame uses the dataset placeholder intrinsics instead of ViGeo-fitted ones
    wbench_fitted_intrinsic: bool = True
    cache_stride: int = 1
    skip_recent_latents: int = 1
    downsample: int = 4
    dropout: float = 0.0
    force_all_invalid: bool = False
    constant_depth: float = 1.0
    include_sink: bool = False
    include_nearby: bool = True
    # Dynamic-object motion mask (ViGeo backend only, applied when ingesting into the bank).
    # Ego-motion is compensated first: the most recent bank frames are
    # warped into the new camera and regions whose photometric residual exceeds the threshold are
    # treated as dynamic and removed from valid_mask. Default off = no behaviour change.
    vigeo_dynamic_mask: bool = False
    # Photometric residual threshold used for the dynamic test (mean absolute channel difference in [-1,1])
    vigeo_dynamic_mask_threshold: float = 0.25


@dataclass
class ControlConfig:
    candidates: list[list[str]] = field(default_factory=lambda: [[]])
    probs: list[float] = field(default_factory=lambda: [1.0])
    action_scale: str = "0.14,0.075,0.22,0.17,0.70,0.16"
    action_freq_scale: float = 1000.0
    action_freq_dim_per_axis: int = 32
    action_learning_rate: float | None = None
    action_history_memory: bool = False

    def uses(self, mode: str) -> bool:
        return any(mode in candidate for candidate in self.candidates)


@dataclass
class ValidationDatasetConfig:
    source: str = "sekai_game"
    split: str = "val"
    filter: str | None = None
    video_base_dir: str | None = None
    annotation_base_dir: str | None = None
    # When set, the i2v first frame is taken from the last frame of external videos in this directory
    first_frame_video_dir: str | None = None
    prompt_pool_file: str | None = None
    # ===== fields specific to source == "wbench_navi" =====
    root: str | None = None  # benchmark data root (cases / images / masks)
    case_ids: list[str] = field(default_factory=list)  # empty = all cases
    image_dir: str | None = None  # custom first-frame directory (reusing the actions of pose_case_id)
    pose_case_id: str | None = None
    pose_actions: list[str] = field(default_factory=list)
    sekai_jsonl: str | None = None
    sekai_video_base: str | None = None
    sekai_caption_base: str | None = None
    sekai_random_n: int = 0
    sekai_seed: int = 42
    # ===== fields specific to source == "custom_i2v" =====
    pose_jsonl: str | None = None
    pose_offset: int = 0
    poses_per_image: int = 1
    pose_stride: int = 40
    captions_json: str | None = None


@dataclass
class ValidationLayoutConfig:
    condition: str = "i2v"
    output_latent_frames: int = 8
    condition_latent_frames: int = 1
    history_latent_frames: int | None = None
    max_gap_sec: float | None = None
    height: int | None = None
    width: int | None = None


@dataclass
class ValidationModeConfig:
    dataset: ValidationDatasetConfig = field(default_factory=ValidationDatasetConfig)
    layout: ValidationLayoutConfig = field(default_factory=ValidationLayoutConfig)
    control: list[str] = field(default_factory=list)
    rollout_rounds: int = 1
    use_memory: bool = True
    # First rollout round that may consume history-encoder tokens. History is
    # still updated before this round, so 1 means R0 has no memory and R1 uses R0.
    memory_start_round: int = 0
    action_cfg_scale: float = 1.0
    prompt_schedule: list[str] = field(default_factory=list)
    skip_spatial_bank_after_magic_chunks: int = 0
    # >1: generate N videos from the same first frame using N different prompts
    prompt_variants: int = 1
    # Also draw random prompts for samples that carry no event caption of their own
    prompt_variants_all_random: bool = False
    # Base seed for prompt-variant sampling; different configs give different prompt combinations
    prompt_variants_seed: int = 0
    # ===== fields specific to the wbench_navi validation mode =====
    all_samples: bool = False  # true: split the whole dataset across ranks (padded), ignoring max_samples
    wbench_chunks_per_turn: int = 3  # rollout rounds per turn (one action)
    wbench_all_turns: bool = False  # true: use every turn (interaction turns reuse the previous navigation action)
    wbench_output_dir: str | None = None  # output directory for case_<id>_combined.mp4
    wbench_forward_speed_per_latent: float = 0.16
    wbench_yaw_deg_per_latent: float = 6.0
    wbench_pitch_deg_per_latent: float = 6.0
    # true: use the robust XYZ median of the ViGeo pointmap inside subject_mask as the
    # third-person orbit pivot; false falls back to a fixed pivot on the centre ray.
    wbench_orbit_subject_depth: bool = False
    # true: treat the third-person subject as a camera-locked foreground; the 3D bank warps
    # only the background while the subject keeps its first-frame appearance and screen position.
    wbench_lock_subject_foreground: bool = False
    # Whether first-person samples also lock the foreground (the mask follows hands, a cart, etc.).
    # When on, that region of the first frame is pasted back each round with coverage=1.
    wbench_lock_subject_foreground_first_person: bool = False
    # Refresh the anchor content every round (its position stays locked to the first-frame mask).
    wbench_subject_anchor_refresh: bool = False
    # Blend factor for the anchor refresh: anchor = a * latest frame + (1 - a) * previous anchor.
    wbench_subject_anchor_refresh_alpha: float = 1.0
    # Adaptive foreground lock: paste the anchor only in rounds where the subject looks lost.
    # Criterion: at ingest time, compare the mask region of the newest generated frame with the anchor
    #   using mean absolute difference (normalised to the pixel range); >= lost_mae means lost.
    wbench_subject_anchor_adaptive: bool = False
    wbench_subject_anchor_lost_mae: float = 0.10
    # Rounds that drop the spatial/warp condition, producing a text-only driven window. Empty = off.
    # Typical use [1,2]: keep round 0 (anchored on the first frame), open the window for the next two
    # rounds; new content is ingested into the bank and later warps take over. Note that pasting the
    # subject anchor back is disabled in the dropped rounds as well.
    wbench_spatial_drop_rounds: list[int] = field(default_factory=list)
    # Pixel dilation radius when excluding the subject from the 3D bank (suppresses edge ghosting).
    wbench_subject_mask_dilation_pixels: int = 12


@dataclass
class TtcConfig:
    """da3 inference: Pathwise Test-Time Correction (run.py --ttc)."""

    levels: list[int] = field(default_factory=lambda: [750, 500, 250])
    strength: float = 1.2
    ref_action: bool = False


@dataclass
class ValidationConfig:
    enabled: bool = False
    ttc: TtcConfig = field(default_factory=lambda: TtcConfig())  # da3 inference
    before_train: bool = False
    interval: int = 200
    first_step: int = 200
    max_samples: int = 1
    sample_offset: int = 0  # validation start offset: start_idx = sample_offset + rank * max_samples
    dynamic_rounds: bool = False
    sampling_steps: int = 30
    scheduler: str = "shift"
    cfg_scale: float = 3.0
    negative_prompt: str = LTX_DEFAULT_NEGATIVE_PROMPT
    stg_scale: float = 1.0
    stg_blocks: list[int] = field(default_factory=lambda: [28])
    rescale_scale: float = 0.7
    step_dir_suffix: str = ""
    save_videos: bool = True
    save_joystick: bool = True
    # ViGeo validation rollout: how generated chunks are handed over and how predictions are decoded
    vigeo_handoff_mode: str = "rgb_reencode"
    vigeo_pred_decode_mode: str = "chunk"
    # Inference probe: align the DC of the first chunk latent to the motion condition. Off by default.
    vigeo_seam_dc_correct: bool = False
    video_history_latent_frames: int | None = None
    camera_extension: str = "static"
    camera_forward_step_per_latent: float = 0.05
    modes: dict[str, ValidationModeConfig] = field(default_factory=dict)


@dataclass
class LoraConfig:
    enabled: bool = True
    train: bool = True
    rank: int = 128
    alpha: int = 128
    targets: list[str] = field(
        default_factory=lambda: [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ]
    )


@dataclass
class RuntimeConfig:
    dtype: str = "bf16"
    attention_type: str = "flash_attention_3"
    gradient_checkpointing: bool = True
    vae_chunk_size: int = 33
    vae_decode_chunk_latents: int | None = None
    vae_decode_overlap_latents: int = 0  # da3: >0 => seamless overlap-tiling decode
    dataloader_workers: int = 2
    dataloader_pin_memory: bool = False
    dataloader_prefetch_factor: int = 2
    fsdp: bool = True
    norm_by_fps: bool = True
    norm_by_max_frames: bool = True
    positional_embedding_max_pos: str = "20,2048,2048"
    # Delete invalid spatial warped-context tokens instead of masking them, which keeps
    # self_attention_mask None so the flash path stays available. Requires batch_size=1.
    compact_spatial_tokens: bool = False
    # Capacity of the in-process text-embedding LRU cache (0 = off); the key is the final prompt string
    # A hit reuses the cached CPU context and skips the text encoder entirely
    # Repeated prompts (including the negative prompt) always hit; matching is exact, so there is no semantic risk
    text_embed_cache_entries: int = 0
    # On-disk cache directory (None = RAM LRU only). Shared across restarts and stages; misses are written back
    text_embed_cache_dir: str | None = None
    # Pre-encode every reachable dataset prompt into the disk cache before training starts
    precache_text_embeds: bool = False
    # Whole-clip VAE latent cache directory (None = off). When set, dataset window starts are quantised
    # to the latent grid and the trainer encodes the first 17 latents fresh, slicing the tail from cache
    # (bit-identical; see vae_latent_cache.py). Prebuild it with scripts/tools/precache_vae_latents.py;
    # a miss or a misaligned window falls back to a fresh full-window encode.
    vae_latent_cache_dir: str | None = None


@dataclass
class Da3InferConfig:
    """Unified config-driven entry for the da3 case-demo inference (alaya.inference).

    With enabled=true, `CONFIG_PATH=<cfg> bash scripts/finetune/train.sh` dispatches
    to the da3 pipeline instead of a trainer — the same launch form as every other
    stage. Fields mirror inference/run.py CLI flags (which remain available and
    override nothing here; the two entries are independent)."""

    enabled: bool = False
    input: str = "playground/case1/case1"  # case prefix: <prefix>_image.* + _camera.pt + _prompt.txt
    rounds: int = 1000  # actual = min(rounds, camera length); ~45 => ~1 min
    # "default" here (not reduce-overhead): cudagraph pool checkpointing conflicts
    # with the torchrun elastic launcher used by train.sh. The CLI entry
    # (inference/run.sh) still defaults to reduce-overhead.
    compile: str = "default"  # default | reduce-overhead | max-autotune | none
    flex_attn: bool = True
    joystick: bool | None = None  # None: follow validation.save_joystick
    ttc: bool = False
    video_crf: int = 28
    skill_sec: float = 4.0
    skill_prompt: str | None = None
    skill_keep_wrap: bool = False


def _update_dataclass(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"unknown config key {type(obj).__name__}.{key}")
        current = getattr(obj, key)
        if key == "modes" and isinstance(obj, ValidationConfig):
            if not isinstance(value, dict):
                raise ValueError("validation.modes must be a mapping")
            current.clear()
            for mode_name, mode_value in value.items():
                if not isinstance(mode_name, str) or not mode_name:
                    raise ValueError("validation mode names must be non-empty strings")
                if not isinstance(mode_value, dict):
                    raise ValueError(f"validation.modes.{mode_name} must be a mapping")
                mode_cfg = ValidationModeConfig()
                _update_dataclass(mode_cfg, mode_value)
                current[mode_name] = mode_cfg
            continue
        if key == "dataset" and isinstance(obj, ValidationModeConfig) and isinstance(value, str):
            current.source = value
            continue
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)


def _validate_control(control: ControlConfig) -> None:
    if len(control.candidates) != len(control.probs):
        raise ValueError("control.candidates/probs length mismatch")
    if not control.candidates:
        raise ValueError("control.candidates must contain at least one candidate")
    allowed = {"action"}
    for candidate in control.candidates:
        if not isinstance(candidate, list):
            raise ValueError("each control candidate must be a list, e.g. ['action'] or []")
        unknown = set(candidate) - allowed
        if unknown:
            raise ValueError(f"unknown control modes: {sorted(unknown)}; allowed={sorted(allowed)}")
        if len(candidate) != len(set(candidate)):
            raise ValueError(f"duplicated modes in control candidate: {candidate}")
    total = sum(float(p) for p in control.probs)
    if total <= 0 or any(float(p) < 0 for p in control.probs):
        raise ValueError("control.probs must be non-negative and sum to > 0")


def _validate_validation(validation: ValidationConfig, *, allow_empty_modes: bool = False) -> None:
    if not validation.enabled:
        return
    if validation.interval <= 0:
        raise ValueError("validation.interval must be > 0")
    if validation.first_step < 0:
        raise ValueError("validation.first_step must be >= 0")
    if validation.max_samples <= 0:
        raise ValueError("validation.max_samples must be > 0")
    if validation.sampling_steps <= 0:
        raise ValueError("validation.sampling_steps must be > 0")
    if validation.scheduler not in {"shift", "linear_quadratic", "uniform"}:
        raise ValueError("validation.scheduler must be one of: shift, linear_quadratic, uniform")
    if validation.cfg_scale < 1.0:
        raise ValueError("validation.cfg_scale must be >= 1")
    if validation.stg_scale < 0.0:
        raise ValueError("validation.stg_scale must be >= 0")
    if validation.rescale_scale < 0.0:
        raise ValueError("validation.rescale_scale must be >= 0")
    if validation.camera_extension not in {"static", "forward"}:
        raise ValueError("validation.camera_extension must be one of: static, forward")
    if float(validation.camera_forward_step_per_latent) < 0.0:
        raise ValueError("validation.camera_forward_step_per_latent must be >= 0")
    if not validation.modes and not allow_empty_modes:
        raise ValueError("validation.modes must contain at least one mode when validation.enabled=true")
    for mode_name, mode in validation.modes.items():
        if mode.rollout_rounds <= 0:
            raise ValueError(f"validation.modes.{mode_name}.rollout_rounds must be > 0")
        if float(mode.action_cfg_scale) < 1.0:
            raise ValueError(f"validation.modes.{mode_name}.action_cfg_scale must be >= 1")
        if mode.layout.condition not in {"hc", "i2v", "v2v"}:
            raise ValueError(f"validation.modes.{mode_name}.layout.condition must be one of: hc, i2v, v2v")
        if mode.layout.output_latent_frames <= 0:
            raise ValueError(f"validation.modes.{mode_name}.layout.output_latent_frames must be > 0")
        if mode.layout.condition_latent_frames < 0:
            raise ValueError(f"validation.modes.{mode_name}.layout.condition_latent_frames must be >= 0")
        unknown = set(mode.control) - {"action"}
        if unknown:
            raise ValueError(
                f"unknown validation.modes.{mode_name}.control modes: {sorted(unknown)}; allowed=['action']"
            )
        if len(mode.control) != len(set(mode.control)):
            raise ValueError(f"duplicated modes in validation.modes.{mode_name}.control: {mode.control}")
