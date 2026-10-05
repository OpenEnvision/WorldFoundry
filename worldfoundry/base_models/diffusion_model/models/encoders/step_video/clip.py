"""Hunyuan CLIP text encoder used by StepVideo (max length 77).

:class:`HunyuanClip` is the secondary prompt tower paired
with STEP1 in :class:`~.component.StepVideoPromptConditioner`.

Tokens are CLIP-length (77).  Primary long-context encode
is :class:`~.step_llm.STEP1TextEncoder`.

Borrowed Hunyuan CLIP graph, not Wan XLM-RoBERTa CLIP.
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
import os


def _load_local_bert_model(model_dir):
    try:
        return BertModel.from_pretrained(model_dir, local_files_only=True)
    except ValueError as exc:
        if "upgrade torch to at least v2.6" not in str(exc):
            raise
        # Transformers refuses every legacy .bin file on torch<2.6, including
        # an explicitly staged, trusted local checkpoint.  Keep Hub/network
        # loading disabled and use PyTorch's weights-only loader for this one
        # repository-owned asset.
        config = BertConfig.from_pretrained(model_dir, local_files_only=True)
        model = BertModel(config)
        state_dict = torch.load(
            os.path.join(model_dir, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        bert_state_dict = {
            key.removeprefix("bert."): value
            for key, value in state_dict.items()
            if key.startswith("bert.")
        }
        if bert_state_dict:
            state_dict = bert_state_dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        allowed_missing_keys = {"pooler.dense.weight", "pooler.dense.bias"}
        disallowed_missing_keys = set(missing_keys) - allowed_missing_keys
        if disallowed_missing_keys or unexpected_keys:
            raise RuntimeError(
                "legacy local BERT checkpoint did not match BertModel: "
                f"missing={sorted(disallowed_missing_keys)}, "
                f"unexpected={sorted(unexpected_keys)}"
            )
        return model


class HunyuanClip(nn.Module):
    """
        Hunyuan clip code copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py
        hunyuan's clip used BertModel and BertTokenizer, so we copy it.
    """
    def __init__(self, model_dir, max_length=77):
        """Init.

        Args:
            model_dir: The model dir.
            max_length: The max length.
        """
        super(HunyuanClip, self).__init__()
        
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, 'tokenizer'))
        self.text_encoder = _load_local_bert_model(os.path.join(model_dir, 'clip_text_encoder'))
        
    @torch.no_grad
    def forward(self, prompts, with_mask=True):
        """Forward.

        Args:
            prompts: The prompts.
            with_mask: The with mask.
        """
        self.device = next(self.text_encoder.parameters()).device
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(
            text_inputs.input_ids.to(self.device),
            attention_mask=text_inputs.attention_mask.to(self.device) if with_mask else None,
        )
        return prompt_embeds.last_hidden_state, prompt_embeds.pooler_output
        
