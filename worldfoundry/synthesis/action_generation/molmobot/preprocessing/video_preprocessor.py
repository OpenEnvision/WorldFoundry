"""Released MolmoBot multimodal config with an image-only inference builder."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import D
from .multicrop_preprocessor import MultiCropConfig
from .multimodal_preprocessor import MultimodalPreprocessor
from .text_preprocessor import TextPreprocessorConfig


@dataclass
class MultiModalVideoPreprocessorConfig(TextPreprocessorConfig):
    """Checkpoint-compatible preprocessor configuration.

    MolmoBot checkpoints inherit their schema from a video-capable upstream
    model. The action wrapper supplies camera images only, so video decoding,
    subtitle handling, and frame-selection models are intentionally not built.
    Their serialized fields remain here solely for strict YAML compatibility.
    """

    max_frames: int = 1
    frame_sample_mode: str = "uniform_last_frame"
    candidate_sampling_fps: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 16.0])
    cache_videos: bool = True
    loading_method: str = "torchcodec_exact"
    max_fps: list[float] = field(default_factory=lambda: [2.0])
    time_sampling: bool = True
    time_mode: str = "per-frame"
    subtitle_mode: str = "frame_1"
    max_crops: int = 1
    overlap_margins: tuple[float, float] = (4.0, 4.0)
    use_col_tokens: bool = False
    periodic_high_res_frame: int | None = None
    high_low_train_mode: str | None = "local_rnd"
    high_res_frame_sample_options: tuple[int, ...] | None = None
    periodic_sample_rate_training: dict[int, list[float]] | None = field(
        default_factory=lambda: {4: [0.9, 0.03, 0.03, 0.04], 3: [0.6, 0.2, 0.2]}
    )
    skip_low_res_in_high_low: bool = False
    pooling_w: int = 3
    pooling_h: int = 3
    high_res_pooling_w: int | None = 3
    high_res_pooling_h: int | None = 3
    query_based_resolution_selection: bool = False
    max_queries_for_resolution_selection: int = 8
    use_frame_special_tokens: bool = True
    frame_sel_clip_identifier: str = "google/siglip2-so400m-patch14-384"
    image_padding_mask: bool | int = False
    max_subtitle_tokens: int | None = None
    image: MultiCropConfig | None = None
    topk: float | None = None
    prune_from_frame: int = 0

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "legacy_image_mask" in config:
            del config["legacy_image_mask"]
        if "tokenizer" in config:
            del config["tokenizer"]
        return config

    def build(self, tokenizer, image_preprocessor, text_seq_len=None, max_sequence_length=None):
        if self.image_padding_mask:
            raise ValueError("MolmoBot image inference does not support image_padding_mask.")
        if self.image is None:
            raise ValueError("MolmoBot checkpoint config is missing its image preprocessor.")
        image, multi_image = self.image.build_image_preprocessor(
            tokenizer,
            image_preprocessor,
            image_padding_mask=False,
        )
        return MultimodalPreprocessor.build(
            text_preprocessor=self.build_text_preprocessor(tokenizer, max_sequence_length),
            image_preprocessor=image,
            multi_image_preprocessor=multi_image,
            text_seq_len=text_seq_len,
        )


__all__ = ["MultiModalVideoPreprocessorConfig"]
