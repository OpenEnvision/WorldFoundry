"""Thin adapter that feeds the native Wan UMT5 conditioner.

:class:`WanPrompter` is a convenience wrapper around
:class:`~.model.WanTextEncoder` / tokenizer for call sites
that still use a Diffusers-like ``encode_prompt`` API.

Native recipes should call
:class:`~.component.WanTextConditioner.encode` instead.

No new transformer weights live here.
"""

from __future__ import annotations

from worldfoundry.core.prompting import PromptProcessor

from .model import HuggingfaceTokenizer, WanTextEncoder


class WanPrompter(PromptProcessor):
    """Tokenize and encode Wan prompts while keeping loading outside the model."""

    def __init__(self, tokenizer_path=None, text_len: int = 512) -> None:
        super().__init__()
        self.text_len = int(text_len)
        self.text_encoder = None
        self.tokenizer = None
        self.fetch_tokenizer(tokenizer_path)

    def fetch_tokenizer(self, tokenizer_path=None) -> None:
        if tokenizer_path is not None:
            self.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_path,
                seq_len=self.text_len,
                clean="whitespace",
            )

    def fetch_models(self, text_encoder: WanTextEncoder | None = None) -> None:
        self.text_encoder = text_encoder

    def encode_prompt(self, prompt, positive: bool = True, device="cuda"):
        if self.tokenizer is None:
            raise RuntimeError("WanPrompter requires a tokenizer_path before encoding")
        if self.text_encoder is None:
            raise RuntimeError("WanPrompter requires fetch_models(text_encoder=...) before encoding")
        prompt = self.process_prompt(prompt, positive=positive)
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        prompt_embedding = self.text_encoder(ids, mask)
        # Some transformer versions return the hidden state as one of several
        # sibling views.  Mutating a per-sample slice then raises autograd's
        # "view is being modified inplace" error (Astra hits this path even
        # during inference).  Padding is contiguous for the Wan tokenizer, so
        # an out-of-place attention-mask multiply is equivalent and works for
        # both eager and grad-enabled callers.
        return prompt_embedding * mask.unsqueeze(-1).to(dtype=prompt_embedding.dtype)


__all__ = ["WanPrompter"]
