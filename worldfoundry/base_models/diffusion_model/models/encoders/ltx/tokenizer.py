"""LTX Gemma tokenizer with weight pairs for the prompt conditioners.

:class:`LTXVGemmaTokenizer` wraps a Gemma 3 or Gemma 4 tokenizer, ensures the
leading BOS the LTX text projections expect, and pads to a fixed length.
:class:`PaddingSide` selects left or right pad.

Token ids go into Gemma; hidden states then pass through
:mod:`.feature_extractor`.  Not a T5 / UMT5 tokenizer.
"""

from enum import Enum

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class PaddingSide(str, Enum):
    """Whether the LTX Gemma tokenizer pads on the left or right."""

    LEFT = "left"
    RIGHT = "right"


class LTXVGemmaTokenizer:
    """
    Tokenizer wrapper for Gemma models compatible with LTXV processes.
    This class wraps HuggingFace's `AutoTokenizer` for use with Gemma text encoders,
    ensuring correct settings and output formatting for downstream consumption.
    """

    def __init__(
        self,
        tokenizer: str | PreTrainedTokenizerBase,
        max_length: int = 256,
        padding_side: PaddingSide = PaddingSide.LEFT,
    ) -> None:
        """
        Initialize the tokenizer.
        Args:
            tokenizer: Pretrained tokenizer directory, or an already constructed tokenizer
                (for example one embedded in a single-file text-encoder checkpoint).
            max_length (int, optional): Max sequence length for encoding. Defaults to 256.
            padding_side (PaddingSide, optional): Side to pad on. Defaults to ``PaddingSide.LEFT``.
        """
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True, model_max_length=max_length)
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = max_length
        self.tokenizer.padding_side = padding_side.value
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> dict[str, list[tuple[int, int]]]:
        """
        Tokenize the given text and return token IDs and attention weights.
        Args:
            text (str): The input string to tokenize.
            return_word_ids (bool, optional): If True, includes the token's position (index) in the output tuples.
                                              If False (default), omits the indices.
        Returns:
            dict[str, list[tuple[int, int]]] OR dict[str, list[tuple[int, int, int]]]:
                A dictionary with a "gemma" key mapping to:
                    - a list of (token_id, attention_mask) tuples if return_word_ids is False;
                    - a list of (token_id, attention_mask, index) tuples if return_word_ids is True.
        Example:
            >>> tokenizer = LTXVGemmaTokenizer("path/to/tokenizer", max_length=8)
            >>> tokenizer.tokenize_with_weights("hello world")
            {'gemma': [(1234, 1), (5678, 1), (2, 0), ...]}
        """
        text = text.strip()
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("Tokenizer is missing bos_token_id; LTX prompt encoding requires a leading BOS.")
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids[0].tolist()
        # Gemma 3 tokenizers emit BOS; Gemma 4 tokenizers do not, so it is prepended here.
        if not input_ids or input_ids[0] != bos_id:
            input_ids = [bos_id, *input_ids][: self.max_length]
        padded = self.tokenizer.pad(
            {"input_ids": [input_ids]},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = padded.input_ids
        attention_mask = padded.attention_mask
        tuples = [
            (token_id, attn, i) for i, (token_id, attn) in enumerate(zip(input_ids[0], attention_mask[0], strict=True))
        ]
        out = {"gemma": tuples}

        if not return_word_ids:
            # Return only (token_id, attention_mask) pairs, omitting token position
            out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}

        return out
