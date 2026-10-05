"""Construct checkpoint-compatible LTX DiT graphs from serialized config.

Used by :mod:`.ltx` when the recipe loads an official LTX-2 audio-video or
LTX-Video 0.9.x transformer.  This module is **not** a
:class:`~...contracts.Denoiser`; it only instantiates
:class:`~..networks.ltx.model.LTXModel` with ABI-checked defaults.

LTX-Video 0.9.x omitted several serialized fields.  Their runtime defaults
are part of the released checkpoint ABI: interleaved RoPE and top-left
patch coordinates.  LTX-2.x changed both, so inheriting the newer values
would produce valid-shaped but semantically corrupt attention.
"""

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModel, LTXModelType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import LTXRopeType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.text_projection import create_caption_projection
from worldfoundry.core.configuration import require_config_value as check_config_value
from worldfoundry.core.nn import DEFAULT_TRANSFORMER_OPS, TransformerOpsConfig


class LTXModelConfigurator:
    """Build the LTX-2 joint audio-video transformer from a config dict."""

    @classmethod
    def from_config(cls, config: dict, ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS) -> LTXModel:
        """Instantiate :class:`LTXModel` with audio-video ABI checks."""
        # Build caption projections for 19B models (projection handled in transformer).
        caption_projection, audio_caption_projection = _build_caption_projections(config, is_av=True)

        config = config.get("transformer", {})

        check_config_value(config, "dropout", 0.0)
        check_config_value(config, "attention_bias", True)
        check_config_value(config, "num_vector_embeds", None)
        check_config_value(config, "activation_fn", "gelu-approximate")
        check_config_value(config, "num_embeds_ada_norm", 1000)
        check_config_value(config, "use_linear_projection", False)
        check_config_value(config, "only_cross_attention", False)
        check_config_value(config, "cross_attention_norm", True)
        check_config_value(config, "double_self_attention", False)
        check_config_value(config, "upcast_attention", False)
        check_config_value(config, "standardization_norm", "rms_norm")
        check_config_value(config, "norm_elementwise_affine", False)
        check_config_value(config, "qk_norm", "rms_norm")
        check_config_value(config, "positional_embedding_type", "rope")
        check_config_value(config, "use_audio_video_cross_attention", True)
        check_config_value(config, "share_ff", False)
        check_config_value(config, "av_cross_ada_norm", True)
        check_config_value(config, "use_middle_indices_grid", True)
        check_config_value(config, "num_attention_heads", config.get("audio_num_attention_heads", float("nan")))

        return LTXModel(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=config.get("num_attention_heads", 32),
            attention_head_dim=config.get("attention_head_dim", 128),
            in_channels=config.get("in_channels", 128),
            out_channels=config.get("out_channels", 128),
            num_layers=config.get("num_layers", 48),
            cross_attention_dim=config.get("cross_attention_dim", 4096),
            norm_eps=config.get("norm_eps", 1e-06),
            ops=ops,
            positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=config.get("positional_embedding_max_pos", [20, 2048, 2048]),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", True),
            audio_num_attention_heads=config.get("audio_num_attention_heads", 32),
            audio_attention_head_dim=config.get("audio_attention_head_dim", 64),
            audio_in_channels=config.get("audio_in_channels", 128),
            audio_out_channels=config.get("audio_out_channels", 128),
            audio_cross_attention_dim=config.get("audio_cross_attention_dim", 2048),
            audio_positional_embedding_max_pos=config.get("audio_positional_embedding_max_pos", [20]),
            av_ca_timestep_scale_multiplier=config.get("av_ca_timestep_scale_multiplier", 1),
            rope_type=LTXRopeType(config.get("rope_type", "split")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
            apply_gated_attention=config.get("apply_gated_attention", False),
            caption_projection=caption_projection,
            audio_caption_projection=audio_caption_projection,
            cross_attention_adaln=config.get("cross_attention_adaln", False),
        )


class LTXVideoOnlyModelConfigurator:
    """Build the LTX-Video 0.9.x video-only transformer from a config dict."""

    MODEL_CLS = LTXModel

    @classmethod
    def from_config(cls, config: dict, ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS) -> LTXModel:
        """Instantiate a video-only :class:`LTXModel` with 0.9.x ABI defaults."""
        # Build caption projection for 19B model (projection handled in transformer).
        caption_projection, _ = _build_caption_projections(config, is_av=False)

        config = config.get("transformer", {})

        # LTX-Video 0.9.x predates these serialized fields; their runtime
        # defaults are part of the released checkpoint ABI.  In particular,
        # the 0.9.x transformer uses the legacy interleaved RoPE layout and
        # top-left patch coordinates.  LTX-2.x changed both defaults, so
        # silently inheriting the newer values produces valid-shaped but
        # semantically corrupt attention activations.
        config.setdefault("cross_attention_norm", True)
        config.setdefault("use_middle_indices_grid", False)
        config.setdefault("rope_type", "interleaved")

        check_config_value(config, "dropout", 0.0)
        check_config_value(config, "attention_bias", True)
        check_config_value(config, "num_vector_embeds", None)
        check_config_value(config, "activation_fn", "gelu-approximate")
        check_config_value(config, "num_embeds_ada_norm", 1000)
        check_config_value(config, "use_linear_projection", False)
        check_config_value(config, "only_cross_attention", False)
        check_config_value(config, "cross_attention_norm", True)
        check_config_value(config, "double_self_attention", False)
        check_config_value(config, "upcast_attention", False)
        check_config_value(config, "standardization_norm", "rms_norm")
        check_config_value(config, "norm_elementwise_affine", False)
        check_config_value(config, "qk_norm", "rms_norm")
        check_config_value(config, "positional_embedding_type", "rope")
        if not isinstance(config["use_middle_indices_grid"], bool):
            raise ValueError("Config value use_middle_indices_grid must be boolean")

        return cls.MODEL_CLS(
            model_type=LTXModelType.VideoOnly,
            num_attention_heads=config.get("num_attention_heads", 32),
            attention_head_dim=config.get("attention_head_dim", 128),
            in_channels=config.get("in_channels", 128),
            out_channels=config.get("out_channels", 128),
            num_layers=config.get("num_layers", 48),
            cross_attention_dim=config.get("cross_attention_dim", 4096),
            norm_eps=config.get("norm_eps", 1e-06),
            ops=ops,
            positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=config.get("positional_embedding_max_pos", [20, 2048, 2048]),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", False),
            rope_type=LTXRopeType(config.get("rope_type", "interleaved")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
            apply_gated_attention=config.get("apply_gated_attention", False),
            caption_projection=caption_projection,
            cross_attention_adaln=config.get("cross_attention_adaln", False),
        )


def _build_caption_projections(
    config: dict,
    is_av: bool,
) -> tuple[torch.nn.Module | None, torch.nn.Module | None]:
    """Build caption projections for the transformer when projection is NOT in the text encoder.
    19B models: projection is in the transformer (caption_proj_before_connector=False).
    22B models: projection is in the text encoder, so no projections are created here.
    Args:
        config: Full model config dict (must contain "transformer" key).
        is_av: Whether this is an audio-video model. When False, audio projection is skipped.
    Returns:
        Tuple of (video_caption_projection, audio_caption_projection), both None for 22B models.
    """
    transformer_config = config.get("transformer", {})
    if transformer_config.get("caption_proj_before_connector", False):
        return None, None

    with torch.device("meta"):
        caption_projection = create_caption_projection(transformer_config)
        audio_caption_projection = create_caption_projection(transformer_config, audio=True) if is_av else None
    return caption_projection, audio_caption_projection
