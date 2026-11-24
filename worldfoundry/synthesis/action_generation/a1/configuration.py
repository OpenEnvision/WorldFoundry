"""Checkpoint-derived configuration for the in-tree A1 inference runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return default


def _known_kwargs(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls) if item.init}
    return {key: value for key, value in values.items() if key in names}


@dataclass
class A1Config:
    """Inference-only subset of the released AffordVLA configuration.

    Defaults describe the public Qwen2-7B + CLIP-L/14 architecture.  A local
    checkpoint configuration always takes precedence and weight shapes are
    subsequently used to validate the action interface.
    """

    d_model: int = 3584
    n_heads: int = 28
    n_kv_heads: int = 4
    n_layers: int = 28
    mlp_hidden_size: int = 37888
    vocab_size: int = 152064
    embedding_size: int = 152064
    additional_vocab_size: int = 128
    max_sequence_length: int = 4096
    head_dim: int | None = None
    qkv_bias: bool = True
    include_bias: bool = False
    weight_tying: bool = False
    rope: bool = True
    rope_theta: float = 1_000_000.0
    layer_norm_eps: float = 1e-6
    block_type: str = "sequential"
    activation_type: str = "swiglu"
    layer_norm_type: str = "rms"
    llm_causal_attention: bool = False

    image_size: int = 336
    image_patch_size: int = 14
    image_emb_dim: int = 1024
    image_num_heads: int = 16
    image_num_layers: int = 24
    image_mlp_dim: int = 4096
    image_num_pos: int = 577
    image_norm_eps: float = 1e-5
    vit_layers: tuple[int, ...] = (-2, -9)
    image_pooling_h: int = 2
    image_pooling_w: int = 2
    image_pooling_2d: str = "attention_meanq"
    image_projector: str = "mlp"
    image_padding_embed: str | None = "pad_and_partial_pad"
    crop_mode: str = "overlap-and-resize-c2"
    max_crops: int = 12
    use_col_tokens: bool = True
    pad_value: float = 0.0

    action_head: str = "l1_regression"
    action_dim: int = 7
    fixed_action_dim: int = 32
    action_token_dim: int = 32
    num_actions_chunk: int = 10
    proprio_dim: int = 32
    use_proprio: bool = True
    right_end_effector_dim: int = 7
    left_end_effector_dim: int = 7
    mobile_base_dim: int = 3
    action_use_left_eef: bool = False
    action_use_mobile_base: bool = False
    num_diffusion_steps: int = 1000
    num_diffusion_inference_steps: int = 30
    action_head_dit_hidden_size: int = 1152
    action_head_dit_depth: int = 28
    action_head_dit_num_heads: int = 16
    action_head_flow_matching_dim: int = 1024
    action_head_flow_matching_layers: int = 28
    action_head_flow_matching_heads: int = 8
    action_head_flow_matching_intermediate_size: int = 2048
    action_head_flow_matching_kv_heads: int = 4
    action_head_flow_matching_pvf_function: str = "2d_attn_mask"

    tokenizer_identifier: str = "Qwen/Qwen2-7B"
    normalization: str = "bounds_q99"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> A1Config:
        root = _mapping(raw)
        if isinstance(root.get("model"), Mapping):
            root = _mapping(root["model"])

        action_block = _mapping(root.get("action_head"))
        action_head = _first(action_block, "type", "name", "kind") if action_block else root.get("action_head")
        vision = _mapping(root.get("vision_backbone"))
        tokenizer = _mapping(root.get("tokenizer"))

        merged = dict(root)
        merged.update(action_block)
        aliases = {
            "action_head": action_head or "l1_regression",
            "tokenizer_identifier": _first(tokenizer, "identifier", "name_or_path", default=cls.tokenizer_identifier),
            "image_size": int(tuple(_first(vision, "image_default_input_size", default=(336, 336)))[0]),  # noqa: RUF015 - Keeps eager list construction and IndexError.
            "image_patch_size": _first(vision, "image_patch_size", default=14),
            "image_emb_dim": _first(vision, "image_emb_dim", default=1024),
            "image_num_heads": _first(vision, "image_num_heads", default=16),
            "image_num_layers": _first(vision, "image_num_layers", default=24),
            "image_mlp_dim": _first(vision, "image_mlp_dim", default=4096),
            "image_num_pos": _first(vision, "image_num_pos", default=577),
            "image_norm_eps": _first(vision, "image_norm_eps", default=1e-5),
        }
        merged.update(aliases)
        if "fixed_action_dim" not in merged and "action_dim" in merged:
            merged["fixed_action_dim"] = merged["action_dim"]
        merged.setdefault("action_token_dim", merged.get("fixed_action_dim", merged.get("action_dim", 7)))
        merged.setdefault("proprio_dim", merged.get("fixed_action_dim", merged.get("action_dim", 7)))
        merged.setdefault("use_proprio", True)
        merged["vit_layers"] = tuple(merged.get("vit_layers") or (-2, -9))
        cfg = cls(**_known_kwargs(cls, merged))
        cfg.validate()
        return cfg

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> A1Config:
        root = Path(checkpoint_dir).expanduser().resolve()
        candidates = (root / "model.yaml", root / "config.yaml", root / "model.yml", root / "config.yml")
        for path in candidates:
            if path.is_file():
                payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if not isinstance(payload, Mapping):
                    raise TypeError(f"A1 configuration must be a mapping: {path}")
                return cls.from_mapping(payload)
        raise FileNotFoundError(f"A1 checkpoint requires model.yaml or config.yaml under {root}")

    def with_weight_shapes(self, shapes: Mapping[str, tuple[int, ...]]) -> A1Config:
        """Resolve ambiguous action dimensions from checkpoint tensors."""

        updates: dict[str, Any] = {}
        l1_input = shapes.get("action_head.model.layer_norm1.weight")
        l1_output = shapes.get("action_head.model.fc2.weight")
        if l1_input:
            if l1_input[0] % self.d_model:
                raise ValueError("A1 L1 action head input width is not divisible by the language hidden size")
            updates["action_token_dim"] = l1_input[0] // self.d_model
            updates["action_head"] = "l1_regression"
        if l1_output:
            updates["fixed_action_dim"] = l1_output[0]

        flow_action = shapes.get("action_head.action_in_proj.weight")
        flow_state = shapes.get("action_head.state_proj.weight")
        if flow_action:
            updates.update(action_head="flow_matching", action_dim=flow_action[1], fixed_action_dim=flow_action[1])
        if flow_state:
            updates["proprio_dim"] = flow_state[1]

        diffusion_action = shapes.get("action_head.action_adaptor.0.weight")
        if diffusion_action:
            updates.update(
                action_head="diffusion",
                action_dim=diffusion_action[1],
                fixed_action_dim=diffusion_action[1],
                action_token_dim=diffusion_action[1],
            )
        cfg = replace(self, **updates) if updates else self
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("A1 language architecture dimensions must be positive")
        if self.d_model % self.n_heads and self.head_dim is None:
            raise ValueError("A1 d_model must be divisible by n_heads when head_dim is absent")
        if self.image_size % self.image_patch_size:
            raise ValueError("A1 image_size must be divisible by image_patch_size")
        patch_side = self.image_size // self.image_patch_size
        if patch_side % self.image_pooling_h or patch_side % self.image_pooling_w:
            raise ValueError("A1 image patch grid must be divisible by its pooling factors")
        if self.action_head not in {"l1_regression", "diffusion", "flow_matching"}:
            raise ValueError(f"Unsupported A1 action head: {self.action_head!r}")
        for name in ("action_dim", "fixed_action_dim", "action_token_dim", "num_actions_chunk"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"A1 {name} must be positive")

    def build_llm_config(self, *, device: str | None = None):
        from worldfoundry.synthesis.action_generation.molmobot.modeling.llm import (
            ActivationType,
            AttentionType,
            BlockType,
            LayerNormType,
            LlmConfig,
        )

        return LlmConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            qkv_bias=self.qkv_bias,
            n_layers=self.n_layers,
            mlp_hidden_size=self.mlp_hidden_size,
            activation_type=ActivationType(self.activation_type),
            block_type=BlockType(self.block_type),
            rope=self.rope,
            rope_theta=self.rope_theta,
            attention_type=AttentionType.sdpa,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            layer_norm_type=LayerNormType(self.layer_norm_type),
            layer_norm_eps=self.layer_norm_eps,
            max_sequence_length=self.max_sequence_length,
            include_bias=self.include_bias,
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            additional_vocab_size=self.additional_vocab_size,
            weight_tying=self.weight_tying,
            use_position_ids=True,
            normalize_input_embeds=False,
            activation_checkpoint=None,
            compile="blocks",
        )

    def build_vision_config(self):
        from worldfoundry.synthesis.action_generation.molmobot.modeling.image_vit import (
            ActivationType,
            AttentionType,
            VisionBackboneType,
            VitConfig,
        )
        from worldfoundry.synthesis.action_generation.molmobot.modeling.vision_backbone import (
            ImagePaddingEmbed,
            ImagePooling2DType,
            ImageProjectType,
            MolmoVisionBackboneConfig,
        )

        vit = VitConfig(
            image_model_type=VisionBackboneType.openai,
            image_default_input_size=(self.image_size, self.image_size),
            image_patch_size=self.image_patch_size,
            image_pos_patch_size=self.image_patch_size,
            image_emb_dim=self.image_emb_dim,
            image_num_heads=self.image_num_heads,
            image_num_key_value_heads=self.image_num_heads,
            image_num_layers=self.image_num_layers,
            image_head_dim=self.image_emb_dim // self.image_num_heads,
            image_mlp_dim=self.image_mlp_dim,
            image_mlp_activations=ActivationType.gelu,
            image_num_pos=self.image_num_pos,
            image_norm_eps=self.image_norm_eps,
            attention_type=AttentionType.sdpa,
            attention_dropout=0.0,
            residual_dropout=0.0,
            normalize="openai",
            resize_mode="default",
            pad_value=self.pad_value,
            activation_checkpointing=False,
        )
        padding = ImagePaddingEmbed(self.image_padding_embed) if self.image_padding_embed else None
        return MolmoVisionBackboneConfig(
            vit=vit,
            image_pooling_2d=ImagePooling2DType(self.image_pooling_2d),
            pooling_attention_mask=False,
            image_projector=ImageProjectType(self.image_projector),
            image_padding_embed=padding,
            vit_layers=self.vit_layers,
            skip_unused_layers=False,
            use_deepstack=False,
            image_feature_dropout=0.0,
            connector_activation_checkpointing=False,
            compile_vit="blocks",
            compile_connector="dynamic",
            normalize_on_gpu=False,
        )


__all__ = ["A1Config"]
