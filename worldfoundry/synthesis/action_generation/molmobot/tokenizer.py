"""Tokenizer construction for MolmoBot inference checkpoints."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from transformers import AutoTokenizer

from .config import BaseConfig
from .torch_utils import barrier, get_local_rank

log = logging.getLogger(__name__)

IMAGE_PATCH_TOKEN = "<im_patch>"
IMAGE_LOW_RES_TOKEN = "<im_low>"
IM_START_TOKEN = "<im_start>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"
FRAME_START_TOKEN = "<frame_start>"
IM_END_TOKEN = "<im_end>"
FRAME_END_TOKEN = "<frame_end>"
IM_COL_TOKEN = "<im_col>"
IMAGE_PROMPT = "<|image|>"
VIDEO_PROMPT = "<|video|>"

EXTRA_TOKENS = (
    IM_START_TOKEN,
    IM_END_TOKEN,
    IMAGE_PATCH_TOKEN,
    IM_COL_TOKEN,
    LOW_RES_IMAGE_START_TOKEN,
    IMAGE_PROMPT,
    IMAGE_LOW_RES_TOKEN,
    FRAME_START_TOKEN,
    FRAME_END_TOKEN,
    VIDEO_PROMPT,
)


class HfTokenizerWrapper:
    """Narrow compatibility wrapper used by the Molmo preprocessors."""

    def __init__(self, tokenizer, bos_token_id=None):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id if bos_token_id is None else bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_id = -1
        special = get_special_token_ids(self)
        self.image_end_token_id = special[IM_END_TOKEN]
        self.image_start_token_id = special[IM_START_TOKEN]
        self.low_res_image_start_token_id = special[LOW_RES_IMAGE_START_TOKEN]
        self.frame_start_token_id = special[FRAME_START_TOKEN]
        self.frame_end_token_id = special[FRAME_END_TOKEN]
        self.image_col_token_id = special[IM_COL_TOKEN]
        self.image_patch_token_id = special[IMAGE_PATCH_TOKEN]
        self.image_low_res_token_id = special[IMAGE_LOW_RES_TOKEN]
        self.image_prompt_token_id = special[IMAGE_PROMPT]
        self.video_prompt_token_id = special[VIDEO_PROMPT]

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int], truncate_at_eos: bool = True) -> str:
        token_ids = [int(token_id) for token_id in token_ids]
        if token_ids and self.eos_token_id == self.bos_token_id and token_ids[0] == self.eos_token_id:
            token_ids = token_ids[1:]
        if truncate_at_eos and self.eos_token_id in token_ids:
            token_ids = token_ids[: token_ids.index(self.eos_token_id)]
        elif not truncate_at_eos:
            token_ids = [token_id for token_id in token_ids if token_id not in (self.eos_token_id, self.bos_token_id)]
        return self.tokenizer.decode(token_ids)

    def vocab_size(self) -> int:
        return len(self.tokenizer)


_TOKENIZER_CACHE: dict[tuple[str, str | None, bool, int | None], HfTokenizerWrapper] = {}


def _source_and_cache(identifier: str, tokenizer_dir: str | None) -> tuple[str, str | None]:
    """Resolve a tokenizer only from an explicit directory or hfd model store."""
    if tokenizer_dir:
        from worldfoundry.core.io.paths import resolve_worldfoundry_path

        directory = resolve_worldfoundry_path(tokenizer_dir)
        if directory.is_dir() and any(
            (directory / filename).is_file()
            for filename in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        ):
            return str(directory), None
        raise FileNotFoundError(f"MolmoBot tokenizer files were not found in the staged checkpoint: {directory}")
    from worldfoundry.core.io.paths import resolve_local_hf_model_path

    directory = resolve_local_hf_model_path(identifier, required_files=("tokenizer_config.json",))
    return str(directory), None


def _has_exact_token(tokenizer, token: str) -> bool:
    ids = tokenizer.encode(token, add_special_tokens=False)
    return len(ids) == 1 and tokenizer.convert_ids_to_tokens(ids[0]) == token


def _load_and_extend_tokenizer(
    source: str,
    cache_dir: str | None,
    has_extra_token: bool,
    pad_tokenizer_to: int | None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=cache_dir,
        local_files_only=True,
        trust_remote_code=False,
        # The released MolmoBot code uses Qwen's deterministic vocab/merges
        # tokenizer. Avoid requiring the much larger tokenizer.json fast-tokenizer
        # artifact when those canonical files are already available.
        use_fast=False,
    )
    if not has_extra_token:
        return tokenizer

    present = [_has_exact_token(tokenizer, token) for token in EXTRA_TOKENS]
    if any(present):
        if not all(present):
            raise ValueError(f"Tokenizer {source!r} contains only a subset of Molmo visual tokens.")
        if pad_tokenizer_to is not None:
            actual = [tokenizer.encode(token, add_special_tokens=False)[0] for token in EXTRA_TOKENS]
            expected = list(range(pad_tokenizer_to, pad_tokenizer_to + len(EXTRA_TOKENS)))
            if actual != expected:
                raise ValueError(
                    f"Tokenizer {source!r} has Molmo visual token IDs {actual}, but this "
                    f"checkpoint requires {expected}."
                )
        return tokenizer

    extra_tokens = list(EXTRA_TOKENS)
    if pad_tokenizer_to is not None:
        if len(tokenizer) > pad_tokenizer_to:
            raise ValueError(
                f"Tokenizer {source!r} has {len(tokenizer)} tokens, exceeding checkpoint "
                f"embedding size {pad_tokenizer_to}."
            )
        padding_count = pad_tokenizer_to - len(tokenizer)
        extra_tokens = [f"|<EXTRA_TOKENS_{index}>|" for index in range(padding_count)] + extra_tokens

    tokenizer.add_special_tokens({"additional_special_tokens": extra_tokens})
    if pad_tokenizer_to is not None:
        actual = [tokenizer.encode(token, add_special_tokens=False) for token in EXTRA_TOKENS]
        expected = [[pad_tokenizer_to + index] for index in range(len(EXTRA_TOKENS))]
        if actual != expected:
            raise ValueError(f"Failed to assign checkpoint-compatible visual token IDs: {actual} != {expected}.")
    return tokenizer


def build_tokenizer(
    tokenizer_type: str,
    has_extra_token: bool = True,
    tokenizer_dir: str | None = None,
    pad_tokenizer_to: int | None = None,
) -> HfTokenizerWrapper:
    source, cache_dir = _source_and_cache(tokenizer_type, tokenizer_dir)
    cache_key = (source, cache_dir, has_extra_token, pad_tokenizer_to)
    cached = _TOKENIZER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tokenizer = None
    if get_local_rank() == 0:
        for attempt in range(3):
            try:
                tokenizer = _load_and_extend_tokenizer(source, cache_dir, has_extra_token, pad_tokenizer_to)
                break
            except (OSError, TimeoutError) as error:
                if attempt == 2:
                    raise
                log.warning("Tokenizer load failed (%s); retrying.", error)
                time.sleep(1)
    barrier()
    if tokenizer is None:
        tokenizer = _load_and_extend_tokenizer(source, cache_dir, has_extra_token, pad_tokenizer_to)

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    if bos_token_id is None:
        raise ValueError(f"Tokenizer {source!r} has neither a BOS nor an EOS token.")
    wrapped = HfTokenizerWrapper(tokenizer, bos_token_id=bos_token_id)
    _TOKENIZER_CACHE[cache_key] = wrapped
    return wrapped


def get_special_token_ids(tokenizer) -> dict[str, int]:
    ids = [tokenizer.encode(token) for token in EXTRA_TOKENS]
    if any(len(token_ids) != 1 for token_ids in ids):
        raise ValueError(f"Molmo visual tokens are not atomic: {ids}")
    return {token: token_ids[0] for token, token_ids in zip(EXTRA_TOKENS, ids, strict=False)}


@dataclass
class TokenizerConfig(BaseConfig):
    identifier: str = "gpt2"
    tokenizer_dir: str | None = None

    def build(self, pad_tokenizer_to: int | None):
        return build_tokenizer(
            self.identifier,
            tokenizer_dir=self.tokenizer_dir,
            pad_tokenizer_to=pad_tokenizer_to,
        )


__all__ = [
    "EXTRA_TOKENS",
    "FRAME_END_TOKEN",
    "FRAME_START_TOKEN",
    "IMAGE_LOW_RES_TOKEN",
    "IMAGE_PATCH_TOKEN",
    "IMAGE_PROMPT",
    "IM_COL_TOKEN",
    "IM_END_TOKEN",
    "IM_START_TOKEN",
    "LOW_RES_IMAGE_START_TOKEN",
    "VIDEO_PROMPT",
    "HfTokenizerWrapper",
    "TokenizerConfig",
    "build_tokenizer",
    "get_special_token_ids",
]
