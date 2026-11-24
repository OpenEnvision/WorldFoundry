# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os

import torch

from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import HuggingfaceTokenizer
from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import umt5_xxl
from worldfoundry.core.model_loading import load_torch_state_dict

__all__ = ["T5EncoderModel"]


class T5EncoderModel:
    def __init__(
        self,
        model_max_length,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),  # noqa: B008 - Default evaluated once at definition, as upstream.
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model_max_length = model_max_length
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        model = (
            umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=dtype, device=device).eval().requires_grad_(False)
        )
        logging.info(f"loading {checkpoint_path}")
        model.load_state_dict(load_torch_state_dict(checkpoint_path, map_location="cpu"))
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=model_max_length, clean="whitespace")

        self.output_dim = self.model.dim
        self.y_embedder = None

    @property
    def t5(
        self,
    ):
        return self

    def encode(self, texts):
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        _seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask).float()
        return {"y": context[:, None], "y_mask": mask}

    def null(self, n):
        null_y = self.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
        return null_y
