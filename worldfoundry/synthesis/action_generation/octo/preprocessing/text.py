from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config

_MODEL_CONFIG = load_vla_va_wam_runtime_config("octo")
_TEXT_ASSETS = _MODEL_CONFIG["text_assets"]


class TextProcessor(ABC):
    """
    Base class for text tokenization or text embedding.
    """

    @abstractmethod
    def encode(self, strings: Sequence[str]):
        raise NotImplementedError


class HFTokenizer(TextProcessor):
    def __init__(
        self,
        tokenizer_name: str,
        tokenizer_kwargs: dict | None = {  # noqa: B006 - Keeps the upstream default object and signature.
            "max_length": 64,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
        },
        encode_with_model: bool = False,
    ):
        from transformers import AutoTokenizer, FlaxAutoModel  # lazy import

        source = resolve_local_hf_model_path(
            tokenizer_name,
            required_files=("tokenizer.json",),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
        self.tokenizer_kwargs = tokenizer_kwargs
        self.encode_with_model = encode_with_model
        if self.encode_with_model:
            self.model = FlaxAutoModel.from_pretrained(str(source), local_files_only=True)

    def encode(self, strings: Sequence[str]):
        # this creates another nested layer with "input_ids", "attention_mask", etc.
        inputs = self.tokenizer(
            strings,
            **self.tokenizer_kwargs,
        )
        if self.encode_with_model:
            return np.array(self.model(**inputs).last_hidden_state)
        else:
            return dict(inputs)


class MuseEmbedding(TextProcessor):
    def __init__(self, model_path: str | None = None):
        import tensorflow_hub as hub  # lazy import
        import tensorflow_text  # noqa: F401 - Registers the TF Text ops the Hub model uses.

        from worldfoundry.core.io.paths import resolve_worldfoundry_path

        source = resolve_worldfoundry_path(model_path or str(_TEXT_ASSETS["muse_model_path"]))
        if not source.exists():
            raise FileNotFoundError(f"Octo MUSE text encoder is not staged locally: {source}")
        self.muse_model = hub.load(str(source))

    def encode(self, strings: Sequence[str]):
        import tensorflow as tf

        with tf.device("/cpu:0"):
            return self.muse_model(strings).numpy()


class CLIPTextProcessor(TextProcessor):
    def __init__(
        self,
        processor_path: str | None = None,
        tokenizer_kwargs: dict | None = {  # noqa: B006 - Keeps the upstream default object and signature.
            "max_length": 64,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
        },
    ):
        from transformers import CLIPProcessor

        source = resolve_local_hf_model_path(
            processor_path or str(_TEXT_ASSETS["clip_processor_path"]),
            required_files=("config.json", "preprocessor_config.json"),
        )
        self.processor = CLIPProcessor.from_pretrained(
            str(source),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.kwargs = tokenizer_kwargs

    def encode(self, strings: Sequence[str]):
        inputs = self.processor(
            text=strings,
            **self.kwargs,
        )
        inputs["position_ids"] = np.expand_dims(np.arange(inputs["input_ids"].shape[1]), axis=0).repeat(
            inputs["input_ids"].shape[0], axis=0
        )
        return inputs
