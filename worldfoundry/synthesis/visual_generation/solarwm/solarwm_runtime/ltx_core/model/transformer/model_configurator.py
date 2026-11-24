from typing import cast

from worldfoundry.base_models.diffusion_model.models.denoisers.ltx_configurator import (
    LTXModelConfigurator as CanonicalLTXModelConfigurator,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.ltx_configurator import (
    LTXVideoOnlyModelConfigurator as CanonicalVideoOnlyConfigurator,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.ltx_configurator import _build_caption_projections
from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModelType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import LTXRopeType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.solarwm import SolarWMModel as LTXModel
from worldfoundry.core.nn.transformer.ops import TransformerOpsConfig
from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.loader.transformer_ops import (
    DEFAULT_TRANSFORMER_OPS,
)

from ...loader.sd_ops import SDOps
from ...utils import check_config_value
from ..model_protocol import ModelConfigurator


class LTXModelConfigurator(CanonicalLTXModelConfigurator):
    """Construct the shared AV graph with SolarWM's attention and AdaLN policy."""

    MODEL_CLS = LTXModel

    @classmethod
    def from_metadata(cls, metadata: dict, ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS) -> LTXModel:
        return cast(LTXModel, cls.from_config(metadata.get("config", {}), ops=ops))


class LTXVideoOnlyModelConfigurator(CanonicalVideoOnlyConfigurator):
    """Use LTX-2.x split/midpoint defaults for SolarWM's video-only checkpoint."""

    MODEL_CLS = LTXModel

    @classmethod
    def from_metadata(cls, metadata: dict, ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS) -> LTXModel:
        config = dict(metadata.get("config", {}))
        transformer = dict(config.get("transformer", {}))
        check_config_value(transformer, "cross_attention_norm", True)
        check_config_value(transformer, "use_middle_indices_grid", True)
        transformer.setdefault("rope_type", "split")
        config["transformer"] = transformer
        return cast(LTXModel, cls.from_config(config, ops=ops))


class LTXAudioOnlyModelConfigurator(ModelConfigurator[LTXModel]):
    """
    Configurator for LTX audio only model.
    Builds an audio-only LTX model (``model_type=AudioOnly``) so the video
    transformer weights are never instantiated or loaded. Useful for
    text-to-audio inference where the video branch is unused.
    """

    @classmethod
    def from_metadata(cls, metadata: dict, ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS) -> LTXModel:
        config = metadata.get("config", {})
        # Build audio caption projection for 19B models (projection handled in transformer).
        _, audio_caption_projection = _build_caption_projections(config, is_av=True)

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
        check_config_value(config, "use_middle_indices_grid", True)

        return LTXModel(
            model_type=LTXModelType.AudioOnly,
            num_layers=config.get("num_layers", 48),
            norm_eps=config.get("norm_eps", 1e-06),
            ops=ops,
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", True),
            audio_num_attention_heads=config.get("audio_num_attention_heads", 32),
            audio_attention_head_dim=config.get("audio_attention_head_dim", 64),
            audio_in_channels=config.get("audio_in_channels", 128),
            audio_out_channels=config.get("audio_out_channels", 128),
            audio_cross_attention_dim=config.get("audio_cross_attention_dim", 2048),
            audio_positional_embedding_max_pos=config.get("audio_positional_embedding_max_pos", [20]),
            rope_type=LTXRopeType(config.get("rope_type", "split")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
            apply_gated_attention=config.get("apply_gated_attention", False),
            audio_caption_projection=audio_caption_projection,
            cross_attention_adaln=config.get("cross_attention_adaln", False),
        )


LTXV_MODEL_COMFY_RENAMING_MAP = (
    SDOps("LTXV_MODEL_COMFY_PREFIX_MAP")
    .with_matching(prefix="model.diffusion_model.")
    .with_replacement("model.diffusion_model.", "")
)

LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP = (
    SDOps("LTXV_AUDIO_ONLY_MODEL_COMFY_MAP")
    .with_matching(prefix="model.diffusion_model.", contains="audio_attn1")
    .with_matching(prefix="model.diffusion_model.", contains="audio_attn2")
    .with_matching(prefix="model.diffusion_model.", contains="audio_ff")
    .with_matching(prefix="model.diffusion_model.", contains="audio_patchify")
    .with_matching(prefix="model.diffusion_model.", contains="audio_proj_out")
    .with_matching(prefix="model.diffusion_model.", contains="audio_adaln_single")
    .with_matching(prefix="model.diffusion_model.", contains="audio_prompt")
    .with_matching(prefix="model.diffusion_model.", contains="audio_scale_shift_table")
    .with_replacement("model.diffusion_model.", "")
)
