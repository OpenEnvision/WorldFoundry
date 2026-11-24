from worldfoundry.base_models.diffusion_model.models.encoders.ltx.variants.joyai_echo import (
    EchoEmbeddings1DConnector as Embeddings1DConnector,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import (
    LTXRopeType,
)
from worldfoundry.synthesis.visual_generation.joyai_echo.ltx_attention_policy import build_attention_ops

from ...model.model_protocol import ModelConfigurator


class Embeddings1DConnectorConfigurator(ModelConfigurator[Embeddings1DConnector]):
    """Configurator for video embeddings connector."""

    @classmethod
    def from_config(cls, config: dict) -> Embeddings1DConnector:
        transformer_config = config.get("transformer", {})
        rope_type = LTXRopeType(transformer_config.get("rope_type", "interleaved"))
        double_precision_rope = transformer_config.get("frequencies_precision", False) == "float64"
        pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])

        # Video connector dimensions
        num_attention_heads = transformer_config.get("connector_num_attention_heads", 30)
        attention_head_dim = transformer_config.get("connector_attention_head_dim", 128)
        num_layers = transformer_config.get("connector_num_layers", 2)

        connector = Embeddings1DConnector(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            positional_embedding_max_pos=pe_max_pos,
            attention_ops=build_attention_ops(),
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=transformer_config.get("connector_apply_gated_attention", False),
        )
        return connector


class AudioEmbeddings1DConnectorConfigurator(ModelConfigurator[Embeddings1DConnector]):
    """Configurator for audio embeddings connector with separate dimension settings."""

    @classmethod
    def from_config(cls, config: dict) -> Embeddings1DConnector:
        transformer_config = config.get("transformer", {})
        rope_type = LTXRopeType(transformer_config.get("rope_type", "interleaved"))
        double_precision_rope = transformer_config.get("frequencies_precision", False) == "float64"
        pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])

        # Audio connector dimensions - fall back to video connector config for backwards compatibility
        num_attention_heads = transformer_config.get(
            "audio_connector_num_attention_heads",
            transformer_config.get("connector_num_attention_heads", 30),
        )
        attention_head_dim = transformer_config.get(
            "audio_connector_attention_head_dim",
            transformer_config.get("connector_attention_head_dim", 128),
        )
        num_layers = transformer_config.get(
            "audio_connector_num_layers",
            transformer_config.get("connector_num_layers", 2),
        )

        connector = Embeddings1DConnector(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            positional_embedding_max_pos=pe_max_pos,
            attention_ops=build_attention_ops(),
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=transformer_config.get("connector_apply_gated_attention", False),
        )
        return connector
