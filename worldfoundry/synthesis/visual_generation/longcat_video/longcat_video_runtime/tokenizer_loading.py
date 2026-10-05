"""Tokenizer loading compatibility for the bundled LongCat-Video runtime."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer


def load_longcat_tokenizer(checkpoint_dir: str | Path):
    """Load LongCat's UMT5 tokenizer without applying a Mistral-only patch.

    Transformers 4.57 can mistake a local high-vocabulary tokenizer for a
    Mistral tokenizer when it inspects the checkpoint root instead of the
    ``tokenizer`` subfolder.  LongCat ships a T5/Metaspace tokenizer, for which
    the suggested Mistral regex rewrite is both semantically wrong and raises
    ``TypeError``.  Passing the flag explicitly as false suppresses that false
    positive while preserving the checkpoint's official tokenizer behavior.
    """

    return AutoTokenizer.from_pretrained(
        checkpoint_dir,
        subfolder="tokenizer",
        torch_dtype=torch.bfloat16,
        fix_mistral_regex=False,
    )
