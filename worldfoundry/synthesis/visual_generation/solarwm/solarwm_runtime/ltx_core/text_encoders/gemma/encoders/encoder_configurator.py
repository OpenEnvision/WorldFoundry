import hashlib
import logging
from os import fspath

import torch
from transformers import AutoModelForImageTextToText, PretrainedConfig

from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_processor import EmbeddingsProcessor
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.feature_extractor import (
    FeatureExtractorV1,
    FeatureExtractorV2,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.gemma import (
    populate_gemma3_buffers as _create_and_populate_gemma3,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.gemma import (
    populate_gemma4_dense_buffers as _create_and_populate_gemma4_dense,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.gemma import (
    populate_gemma4_unified_buffers as _create_and_populate_gemma4_unified,
)

from ....loader import KeyValueOperationResult, parse_model_version
from ....loader.module_ops import ModuleOps
from ....loader.sd_ops import SDOps
from ....model.model_protocol import ModelConfigurator
from ..embeddings_connector import AudioEmbeddings1DConnectorConfigurator, Embeddings1DConnectorConfigurator
from ..gemma_assets import GemmaAssets, build_gemma_hf_config, resolve_gemma_weight_paths
from .base_encoder import LTXGemmaTextEncoder, module_ops_from_gemma_root

logger = logging.getLogger(__name__)


def _load_gemma_config(gemma_model_path: str) -> PretrainedConfig:
    """Load the Hugging Face config from a Gemma directory or single-file TE checkpoint."""
    return build_gemma_hf_config(GemmaAssets.load(gemma_model_path))


def gemma_model_config(gemma_model_path: str) -> dict:
    """Return the Hugging Face Gemma config as a plain dict (includes ``model_type``)."""
    return _load_gemma_config(gemma_model_path).to_dict()


def gemma_model_type(gemma_model_path: str) -> str:
    """Return the ``model_type`` declared in a Gemma checkpoint config."""
    return gemma_model_config(gemma_model_path).get("model_type", "") or ""


def _path_bound_configurator(cls: type, gemma_model_path: str) -> type:
    """Subclass ``cls`` with a path-unique name so ``module_registry_key`` distinguishes HF roots.
    ``from_metadata`` loads structure from the bound path (not from safetensors metadata), so a
    shared ``__qualname__`` would collide across different Gemma trees.
    """
    digest = hashlib.sha256(fspath(gemma_model_path).encode()).hexdigest()[:16]
    return type(f"{cls.__name__}_{digest}", (cls,), {"gemma_model_path": gemma_model_path})


class GemmaTextEncoderConfigurator(ModelConfigurator[LTXGemmaTextEncoder]):
    gemma_model_path: str | None = None

    @classmethod
    def with_gemma_model_path(cls, gemma_model_path: str) -> type["GemmaTextEncoderConfigurator"]:
        """Return a configurator subclass bound to ``gemma_model_path``."""
        return _path_bound_configurator(cls, gemma_model_path)  # type: ignore[return-value]

    @classmethod
    def from_metadata(cls, metadata: dict) -> LTXGemmaTextEncoder:
        if cls.gemma_model_path is None:
            raise ValueError(
                "GemmaTextEncoderConfigurator must be specialized via "
                "GemmaTextEncoderConfigurator.with_gemma_model_path(gemma_model_path) before use."
            )

        gemma_config = _load_gemma_config(cls.gemma_model_path)

        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(gemma_config)

        return LTXGemmaTextEncoder(model=model)


class EmbeddingsProcessorConfigurator(ModelConfigurator[EmbeddingsProcessor]):
    gemma_model_path: str | None = None

    @classmethod
    def with_gemma_model_path(cls, gemma_model_path: str) -> type["EmbeddingsProcessorConfigurator"]:
        return _path_bound_configurator(cls, gemma_model_path)  # type: ignore[return-value]

    @classmethod
    def from_metadata(cls, metadata: dict) -> EmbeddingsProcessor:
        if cls.gemma_model_path is None:
            raise ValueError(
                "EmbeddingsProcessorConfigurator must be specialized via "
                "EmbeddingsProcessorConfigurator.with_gemma_model_path(gemma_model_path) before use."
            )
        config = metadata.get("config", {})
        transformer_config = config.get("transformer", {})
        gemma_config = _load_gemma_config(cls.gemma_model_path)
        _check_gemma_version(metadata, gemma_config, cls.gemma_model_path)

        # Create video embeddings connector (always needed)
        video_connector = Embeddings1DConnectorConfigurator.from_metadata(metadata)

        # Create audio embeddings connector
        audio_connector = AudioEmbeddings1DConnectorConfigurator.from_metadata(metadata)

        # Create feature extractor
        feature_extractor = _create_feature_extractor(transformer_config, gemma_config.text_config)

        return EmbeddingsProcessor(
            video_connector=video_connector,
            audio_connector=audio_connector,
            feature_extractor=feature_extractor,
        )


# Floor from which a checkpoint must declare gemma_source_checkpoint. Deliberately below the
# LTX-2.5 stamp so pre-release builds are covered too; LTX-2.3 / gemma3 checkpoints stay under it.
_GEMMA_SOURCE_CHECKPOINT_REQUIRED_SINCE = (2, 4, 0)


def _check_gemma_version(metadata: dict, gemma_config: PretrainedConfig, gemma_model_path: str) -> None:
    """Verify the checkpoint's expected Gemma version matches the Gemma model being loaded.
    Checkpoints that declare a ``gemma_source_checkpoint`` (LTX 2.5 / gemma4) must match its
    ``gemma_version`` against the Gemma config's top-level ``gemma_version``. Older
    checkpoints (no ``gemma_source_checkpoint``) are only checked for using a Gemma 3 text
    encoder, unless their declared ``model_version`` is at or above
    ``_GEMMA_SOURCE_CHECKPOINT_REQUIRED_SINCE``, in which case a ``gemma_source_checkpoint``
    was required and its absence is itself an error.
    """
    gemma_source_checkpoint = metadata.get("gemma_source_checkpoint")
    if gemma_source_checkpoint is not None:
        expected_gemma_version = gemma_source_checkpoint.get("gemma_version")
        actual_gemma_version = getattr(gemma_config, "gemma_version", None)
        if expected_gemma_version != actual_gemma_version:
            raise ValueError(
                f"Gemma version mismatch: checkpoint's gemma_source_checkpoint expects "
                f"gemma_version={expected_gemma_version!r}, but the Gemma config at "
                f"{gemma_model_path!r} has gemma_version={actual_gemma_version!r}."
            )
        return

    model_version = metadata.get("model_version")
    if model_version is not None and parse_model_version(model_version) >= _GEMMA_SOURCE_CHECKPOINT_REQUIRED_SINCE:
        actual_gemma_version = getattr(gemma_config, "gemma_version", None)
        if actual_gemma_version is None:
            logger.warning(
                "Checkpoint model_version=%r requires gemma_source_checkpoint.gemma_version but "
                "does not declare it, and the Gemma config at %r also has no gemma_version. "
                "Skipping version check — verify compatibility manually.",
                model_version,
                gemma_model_path,
            )
            return
        raise ValueError(
            f"Checkpoint model_version={model_version!r} must declare "
            "gemma_source_checkpoint.gemma_version, but none was found."
        )

    model_type = getattr(gemma_config, "model_type", None)
    if model_type != "gemma3":
        raise ValueError(
            f"Checkpoint has no gemma_source_checkpoint (model_version={model_version!r}); "
            f"expected a Gemma 3 config (model_type='gemma3') at {gemma_model_path!r}, got "
            f"model_type={model_type!r}."
        )


_V2_EXPECTED_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


def _create_feature_extractor(transformer_config: dict, gemma_text_config: object) -> torch.nn.Module:
    """Select and create the appropriate feature extractor based on config.
    Detection logic:
    - V1: V2 config keys absent → projection lives in transformer
    - V2: V2 config keys present with exact expected values → per-token RMS norm with dual aggregate embeds
    - Anything else: NotImplementedError (config drift)
    ``gemma_text_config`` is a transformers text-config (e.g. ``Gemma3TextConfig``); only
    its ``hidden_size`` and ``num_hidden_layers`` fields are read, so this stays
    family-agnostic.
    """
    embedding_dim = gemma_text_config.hidden_size
    num_layers = gemma_text_config.num_hidden_layers + 1  # +1 for the embedding layer
    flat_dim = embedding_dim * num_layers

    overlapping_keys = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
    if not overlapping_keys:
        aggregate_embed = torch.nn.Linear(flat_dim, embedding_dim, bias=False)
        return FeatureExtractorV1(aggregate_embed=aggregate_embed, is_av=True)

    missing_keys = _V2_EXPECTED_CONFIG.keys() - overlapping_keys
    if missing_keys:
        raise NotImplementedError("Partial V2 config — missing keys: " + ", ".join(sorted(missing_keys)))

    unexpected_value_keys = {k for k in overlapping_keys if transformer_config[k] != _V2_EXPECTED_CONFIG[k]}
    if unexpected_value_keys:
        raise NotImplementedError(
            "Unknown config: "
            + ", ".join(
                f"{k}={transformer_config[k]!r} (expected {_V2_EXPECTED_CONFIG[k]!r})" for k in unexpected_value_keys
            )
        )

    video_inner_dim = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
    audio_inner_dim = transformer_config["audio_num_attention_heads"] * transformer_config["audio_attention_head_dim"]
    return FeatureExtractorV2(
        video_aggregate_embed=torch.nn.Linear(flat_dim, video_inner_dim, bias=True),
        embedding_dim=embedding_dim,
        audio_aggregate_embed=torch.nn.Linear(flat_dim, audio_inner_dim, bias=True),
    )


# --- Split SDOps: Gemma LLM keys vs Embeddings Processor keys ---


def _with_tied_lm_head(ops: SDOps) -> SDOps:
    """Mirror the shared embed_tokens weight onto lm_head at load time.
    Gemma's lm_head (output layer) and embed_tokens (input embeddings) share one weight matrix,
    so the checkpoint stores it only under embed_tokens -- there is no lm_head weight in the file.
    When that one weight is loaded, also write it to lm_head, otherwise lm_head stays empty and
    generation fails.
    """
    return ops.with_kv_operation(
        operation=lambda key, value: [
            KeyValueOperationResult(key, value),
            KeyValueOperationResult("model.lm_head.weight", value),
        ],
        key_prefix="model.model.language_model.embed_tokens.weight",
    )


def _build_gemma3_llm_key_ops() -> SDOps:
    """Map Gemma 3 checkpoint keys to the loaded model's state_dict keys.
    Example, checkpoint key then loaded-model key:
        language_model.model.embed_tokens.weight
        model.model.language_model.embed_tokens.weight
    The checkpoint vision key is always vision_tower.vision_model.encoder.X; on the loaded
    model it lands at model.model.vision_tower.encoder.X. transformers 5.8 flattened
    SiglipVisionModel -- the inner SiglipVisionTransformer (previously at .vision_model) was
    hoisted up, so the vision_model. segment is gone.
    """
    return _with_tied_lm_head(
        SDOps("GEMMA3_LLM_KEY_OPS")
        .with_matching(prefix="language_model.model.")
        .with_matching(prefix="vision_tower.")
        .with_matching(prefix="multi_modal_projector.")
        .with_replacement("language_model.model.", "model.model.language_model.")
        .with_replacement("vision_tower.vision_model.", "model.model.vision_tower.")
        .with_replacement("multi_modal_projector.", "model.model.multi_modal_projector.")
    )


def _gemma4_unified_is_comfy_flat(gemma_model_path: str) -> bool:
    """True when weights use Comfy's flattened ``model.layers.*`` LM layout."""
    import safetensors

    for path in resolve_gemma_weight_paths(gemma_model_path):
        with safetensors.safe_open(path, framework="pt") as handle:
            keys = set(handle.keys())
        if "model.layers.0.post_feedforward_layernorm.weight" in keys:
            return True
        if any(k.startswith("model.language_model.") for k in keys):
            return False
    return False


def _build_gemma4_unified_llm_key_ops(*, comfy_flat: bool = False) -> SDOps:
    """Checkpoint-key remapping for Gemma 4 unified (e.g. gemma-4-12B, model_type gemma4_unified).
    * HF (``comfy_flat=False``): ``model.language_model.*`` / ``model.embed_*`` / ``model.vision_embedder.*``
    * Comfy-flat (``comfy_flat=True``): LM under ``model.layers.*`` / ``model.embed_tokens.*``, and
      vision/audio towers under Comfy ``vision_model.*`` / ``multi_modal_projector.*`` /
      ``audio_projector.*`` (plus legacy HF ``model.vision_embedder.*`` / ``model.embed_*`` for
      already-published TE files that only flattened the LM).
    Both land on ``model.model.language_model.*`` / ``model.model.embed_vision.*`` /
    ``model.model.embed_audio.*`` in the loaded module tree. Vision patch embedder loads as
    ``embed_vision.``; its projection loads under ``embed_vision.multimodal_embedder.``.
    """
    ops = SDOps("GEMMA4_UNIFIED_LLM_KEY_OPS").with_matching(prefix="model.")
    if comfy_flat:
        ops = (
            ops.with_matching(prefix="vision_model.")
            .with_matching(prefix="multi_modal_projector.")
            .with_matching(prefix="audio_projector.")
            .with_replacement("model.layers.", "model.model.language_model.layers.")
            .with_replacement("model.embed_tokens.", "model.model.language_model.embed_tokens.")
            .with_replacement("model.norm.", "model.model.language_model.norm.")
        )
    else:
        ops = ops.with_replacement("model.language_model.", "model.model.language_model.")
    # Legacy HF tower names first (nested HF roots, or TE files that only flattened the LM).
    # Comfy tower renames must run after these: ``audio_projector.`` → ``model.model.embed_audio.``
    # would otherwise match the legacy ``model.embed_audio.`` rule and double-prefix.
    ops = (
        ops.with_replacement("model.embed_audio.", "model.model.embed_audio.")
        .with_replacement(
            "model.embed_vision.embedding_projection.",
            "model.model.embed_vision.multimodal_embedder.embedding_projection.",
        )
        .with_replacement("model.vision_embedder.", "model.model.embed_vision.")
    )
    if comfy_flat:
        ops = (
            ops.with_replacement("vision_model.", "model.model.embed_vision.")
            .with_replacement(
                "multi_modal_projector.embedding_projection.",
                "model.model.embed_vision.multimodal_embedder.embedding_projection.",
            )
            .with_replacement("audio_projector.", "model.model.embed_audio.")
        )
    return _with_tied_lm_head(ops)


def _build_gemma4_dense_llm_key_ops() -> SDOps:
    """Checkpoint-key remapping for dense Gemma 4 instruct (e.g. gemma-4-E2B-it, model_type gemma4).
    Checkpoint keys already sit under ``model.<component>.*`` matching the HF module tree;
    LTXGemmaTextEncoder nests that HF module at ``.model``, so remap each top-level component under
    ``model.model.<component>``. (A naive ``model.`` → ``model.model.`` replace would also
    rewrite the ``model.`` substring inside ``language_model.``.)
    """
    return _with_tied_lm_head(
        SDOps("GEMMA4_DENSE_LLM_KEY_OPS")
        .with_matching(prefix="model.")
        .with_replacement("model.language_model.", "model.model.language_model.")
        .with_replacement("model.vision_tower.", "model.model.vision_tower.")
        .with_replacement("model.audio_tower.", "model.model.audio_tower.")
        .with_replacement("model.embed_vision.", "model.model.embed_vision.")
        .with_replacement("model.embed_audio.", "model.model.embed_audio.")
        .with_replacement("model.multi_modal_projector.", "model.model.multi_modal_projector.")
    )


EMBEDDINGS_PROCESSOR_KEY_OPS = (
    SDOps("EMBEDDINGS_PROCESSOR_KEY_OPS")
    # 1. Map the feature extractor (V1: aggregate_embed inside feature_extractor)
    .with_matching(prefix="text_embedding_projection.aggregate_embed.")
    .with_replacement("text_embedding_projection.aggregate_embed.", "feature_extractor.aggregate_embed.")
    # V2 dual aggregate embeds
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    .with_matching(prefix="text_embedding_projection.audio_aggregate_embed.")
    .with_replacement("text_embedding_projection.audio_aggregate_embed.", "feature_extractor.audio_aggregate_embed.")
    # 2. Map the connectors
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "video_connector.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_connector.")
)

VIDEO_ONLY_EMBEDDINGS_PROCESSOR_KEY_OPS = (
    SDOps("VIDEO_ONLY_EMBEDDINGS_PROCESSOR_KEY_OPS")
    # 1. Map the feature extractor (V1: aggregate_embed inside feature_extractor)
    .with_matching(prefix="text_embedding_projection.aggregate_embed.")
    .with_replacement("text_embedding_projection.aggregate_embed.", "feature_extractor.aggregate_embed.")
    # V2 video aggregate embed
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    # 2. Map the connectors
    .with_matching(prefix="model.diffusion_model.embeddings_connector.")
    .with_replacement("model.diffusion_model.embeddings_connector.", "embeddings_processor.video_connector.")
)


def get_gemma_ops(gemma_model_path: str) -> tuple[SDOps, tuple[ModuleOps, ...]]:
    """Return the SDOps and the full module-ops tuple for loading a Gemma text encoder.
    The module ops are the config-derived buffer population (family-specific) followed by the
    tokenizer/processor asset loaders (family-agnostic), ready to pass straight to a Builder.
    Supported ``model_type`` values:
    - ``gemma3`` — encode + enhance
    - ``gemma4_unified`` — encode only
    - ``gemma4`` — dense instruct, enhance only (same encoder/builder stack)
    """
    asset_ops = module_ops_from_gemma_root(gemma_model_path)
    model_type = gemma_model_type(gemma_model_path)
    if model_type == "gemma4_unified":
        gemma4_unified_model_ops = ModuleOps(
            name="GemmaModel",
            matcher=lambda module: isinstance(module, LTXGemmaTextEncoder),
            mutator=_create_and_populate_gemma4_unified,
        )
        comfy_flat = _gemma4_unified_is_comfy_flat(gemma_model_path)
        return _build_gemma4_unified_llm_key_ops(comfy_flat=comfy_flat), (gemma4_unified_model_ops, *asset_ops)
    if model_type == "gemma4":
        gemma4_dense_model_ops = ModuleOps(
            name="GemmaModel",
            matcher=lambda module: isinstance(module, LTXGemmaTextEncoder),
            mutator=_create_and_populate_gemma4_dense,
        )
        return _build_gemma4_dense_llm_key_ops(), (gemma4_dense_model_ops, *asset_ops)
    if model_type == "gemma3":
        gemma3_model_ops = ModuleOps(
            name="GemmaModel",
            matcher=lambda module: isinstance(module, LTXGemmaTextEncoder),
            mutator=_create_and_populate_gemma3,
        )
        return _build_gemma3_llm_key_ops(), (gemma3_model_ops, *asset_ops)
    raise ValueError(
        f"Unsupported Gemma text-encoder model_type {model_type!r} loaded from {gemma_model_path}. "
        "Supported model types are 'gemma3', 'gemma4', and 'gemma4_unified'."
    )
