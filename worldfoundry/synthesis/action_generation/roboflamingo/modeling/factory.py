# Inference-only RoboFlamingo source retained in-tree.
import json
from pathlib import Path

import open_clip
from open_flamingo.src.factory import _infer_decoder_layers_attr_name
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
from transformers import AutoModelForCausalLM, AutoTokenizer

from .flamingo_bc import BCFlamingo
from .flamingo_mpt import MPTFlamingo


def _trusted_staged_mpt_code(lang_encoder_path: str) -> bool:
    """Allow the released RoboFlamingo MPT code only from its staged local layout."""
    model_dir = Path(lang_encoder_path).expanduser().resolve()
    if model_dir.name != "anas-awadalla--mpt-1b-redpajama-200b-dolly":
        return False
    official_dir = Path(__file__).resolve().parents[6] / "ckpts" / model_dir.name
    code_files = (
        "config.json", "attention.py", "configuration_mosaic_gpt.py", "gpt_blocks.py",
        "low_precision_layernorm.py", "mosaic_gpt.py", "param_init_fns.py",
    )
    if model_dir != official_dir and any(
        (model_dir / name).resolve() != official_dir / name for name in code_files
    ):
        return False
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("model_type") == "mosaic_gpt" and config.get("auto_map") == {
        "AutoConfig": "configuration_mosaic_gpt.MosaicGPTConfig",
        "AutoModelForCausalLM": "mosaic_gpt.MosaicGPT",
    }






def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = True,
    decoder_layers_attr_name: str = None,
    # this is the window size sampled from the episode
    window_size: int = 32,
    use_gripper=False,
    use_state=False,
    last_action=False,
    fusion_mode='',
    pad_length=-1,
    sep_resampler=False,
    sep_lm_head=False,
    return_feature=False,
    multi_step_action=1,
    llm_name='llama_9b',
    pooling='max',
    residual=False,
    tcp_rel=False,
    replan=-1,
    decoder_type='lstm',
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
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=True,
        trust_remote_code=_trusted_staged_mpt_code(lang_encoder_path),
    )
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

    if 'llama' in llm_name:
        Model_fn = BCFlamingo
    elif 'mpt' in llm_name:
        Model_fn = MPTFlamingo
    else:
        raise NotImplementedError

    model = Model_fn(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
            "width"
        ],
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
