import logging
from pathlib import Path

import numpy as np
import sentencepiece

from worldfoundry.core.io.paths import resolve_local_hf_model_path, resolve_worldfoundry_path

_PALIGEMMA_TOKENIZER_LOCATION: str | None = None
_FAST_TOKENIZER_LOCATION: str | None = None


def configure_local_tokenizer_assets(*, paligemma: str, fast: str) -> None:
    """Set process-local tokenizer assets used by subsequently built policies."""

    global _PALIGEMMA_TOKENIZER_LOCATION, _FAST_TOKENIZER_LOCATION
    _PALIGEMMA_TOKENIZER_LOCATION = str(paligemma)
    _FAST_TOKENIZER_LOCATION = str(fast)


def _paligemma_tokenizer_file(location: str | Path | None = None) -> Path:
    requested = str(location or _PALIGEMMA_TOKENIZER_LOCATION or "")
    if not requested:
        raise FileNotFoundError(
            "OpenPI requires a local PaliGemma tokenizer path; set "
            "paligemma_tokenizer_path in worldfoundry/data/models/runtime/configs/vla_va_wam/openpi.yaml"
        )
    direct = resolve_worldfoundry_path(requested)
    if direct.is_file():
        return direct.resolve()
    if direct.is_dir():
        for name in ("paligemma_tokenizer.model", "tokenizer.model"):
            candidate = direct / name
            if candidate.is_file():
                return candidate.resolve()
    root = resolve_local_hf_model_path(requested)
    for name in ("paligemma_tokenizer.model", "tokenizer.model"):
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No local PaliGemma SentencePiece model was found under {root}")


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48, tokenizer_path: str | Path | None = None):
        self._max_len = max_len

        path = _paligemma_tokenizer_file(tokenizer_path)
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        fast_tokenizer_path: str | Path | None = None,
        paligemma_tokenizer_path: str | Path | None = None,
    ):
        self._max_len = max_len

        path = _paligemma_tokenizer_file(paligemma_tokenizer_path)
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate the in-tree FAST tokenizer without executing checkpoint code.
        fast_location = fast_tokenizer_path or _FAST_TOKENIZER_LOCATION
        if not fast_location:
            raise FileNotFoundError(
                "OpenPI FAST requires a local action tokenizer path; set fast_tokenizer_path in the runtime YAML"
            )
        from .action_tokenizer import UniversalActionTokenizer

        self._fast_tokenizer = UniversalActionTokenizer.from_pretrained(fast_location)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        tokens = prefix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens)

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################
