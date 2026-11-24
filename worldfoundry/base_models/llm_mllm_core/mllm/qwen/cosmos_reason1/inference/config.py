"""Inference and distributed-layout configuration for the bundled Reason1 encoder."""

import attrs


@attrs.define
class ParallelConfig:
    compile: bool = False
    data_parallel_shard_degree: int = -1
    data_parallel_replicate_degree: int = 1
    tensor_parallel_degree: int = 1
    context_parallel_degree: int = 1
    enable_cpu_offload: bool = False


@attrs.define
class ExperimentalConfig:
    pipeline_parallel_degree: int = 1
    enable_async_tensor_parallel: bool = False


@attrs.define
class Float8Config:
    enable_float8_linear: bool = False


@attrs.define
class FSDP2ModelConfig:
    tokenizer_type: str
    max_batch_size: int = 1
    max_seq_len: int = 128000
    use_fsdp2: bool = True
    use_rope_from_torchtitan: bool = False
    vision_encoder: str = "openai/clip-vit-base-patch32"
    mm_projector: str | None = None
    precision: str = "bfloat16"
    parallel: ParallelConfig = attrs.field(factory=ParallelConfig)
    experimental: ExperimentalConfig = attrs.field(factory=ExperimentalConfig)
    float8: Float8Config = attrs.field(factory=Float8Config)
    seed: int = 0
    deterministic: bool = False
    num_tiles: int = 1
    add_tile_tag: bool = False
    add_image_start_end_tag: bool = False
    add_answer_tag: bool = True
    tile_tag_type: str | None = "space_separated"
    use_cache: bool = False
    cp_size: int | None = None
    ep_size: int | None = None

    def __getitem__(self, item):
        return getattr(self, item)


__all__ = ["ExperimentalConfig", "FSDP2ModelConfig", "Float8Config", "ParallelConfig"]
