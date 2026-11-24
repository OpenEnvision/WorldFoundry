# Inference-only RoboFlamingo source retained in-tree.
import open_clip
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, PreTrainedModel

from worldfoundry.base_models.llm_mllm_core.mllm.open_flamingo.src.factory import _infer_decoder_layers_attr_name
from worldfoundry.base_models.llm_mllm_core.mllm.open_flamingo.src.flamingo_lm import FlamingoLMMixin
from worldfoundry.base_models.llm_mllm_core.mllm.open_flamingo.src.utils import extend_instance

from .flamingo_bc import BCFlamingo
from .flamingo_mpt import MPTFlamingo


def _load_language_encoder(lang_encoder_path: str) -> PreTrainedModel:
    """Load the released MosaicGPT or a native Transformers LM from local assets."""
    config_dict, _ = PretrainedConfig.get_config_dict(lang_encoder_path, local_files_only=True)
    if config_dict.get("model_type") == "mosaic_gpt":
        from worldfoundry.base_models.llm_mllm_core.mllm.mosaic_gpt.configuration_mosaic_gpt import MosaicGPTConfig
        from worldfoundry.base_models.llm_mllm_core.mllm.mosaic_gpt.modeling_mosaic_gpt import MosaicGPT

        config = MosaicGPTConfig.from_dict(config_dict)
        return MosaicGPT.from_pretrained(lang_encoder_path, config=config, local_files_only=True)
    return AutoModelForCausalLM.from_pretrained(lang_encoder_path, local_files_only=True, trust_remote_code=False)


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = True,
    decoder_layers_attr_name: str | None = None,
    # this is the window size sampled from the episode
    window_size: int = 32,
    use_gripper=False,
    use_state=False,
    last_action=False,
    fusion_mode="",
    pad_length=-1,
    sep_resampler=False,
    sep_lm_head=False,
    return_feature=False,
    multi_step_action=1,
    llm_name="llama_9b",
    pooling="max",
    residual=False,
    tcp_rel=False,
    replan=-1,
    decoder_type="lstm",
    hidden_size=None,
    fwd_pred=False,
    fwd_pred_hand=False,
    no_image_patch=False,
    global_latent=1,
    refresh=-1,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    if not use_local_files:
        raise ValueError("RoboFlamingo only supports staged local model assets")
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    lang_encoder = _load_language_encoder(lang_encoder_path)
    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_encoder_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(lang_encoder, EmbeddingFnMixin)

    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    # print(lang_encoder.base_model_prefix)
    # print(getattr(lang_encoder, lang_encoder.base_model_prefix, lang_encoder))
    # print(lang_encoder)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    if "llama" in llm_name:
        Model_fn = BCFlamingo
    elif "mpt" in llm_name:
        Model_fn = MPTFlamingo
    else:
        raise NotImplementedError

    model = Model_fn(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["width"],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        window_size=window_size,
        use_gripper=use_gripper,
        use_state=use_state,
        fusion_mode=fusion_mode,
        last_action=last_action,
        pad_length=pad_length,
        sep_resampler=sep_resampler,
        sep_lm_head=sep_lm_head,
        return_feature=return_feature,
        multi_step_action=multi_step_action,
        llm=llm_name,
        pooling=pooling,
        residual=residual,
        tcp_rel=tcp_rel,
        replan=replan,
        decoder_type=decoder_type,
        hidden_size=hidden_size,
        refresh=refresh,
        fwd_pred=fwd_pred,
        fwd_pred_hand=fwd_pred_hand,
        no_image_patch=no_image_patch,
        global_latent=global_latent,
        **flamingo_kwargs,
    )

    # Inference runtime: freeze the complete model and avoid constructing any
    # optimizer-facing trainable parameter subset.
    model.requires_grad_(False)
    model.eval()

    return model, image_processor, text_tokenizer
