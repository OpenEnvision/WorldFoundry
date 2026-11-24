"""Image/text preprocessing used by MolmoBot action inference."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from .data_formatter import DataFormatter
from .image_preprocessor import load_image
from .preprocessor_utils import TensorSpec, TokenizedVisionData
from .text_preprocessor import InterleavedTextPreprocessor


@dataclasses.dataclass
class MultimodalPreprocessor:
    text_preprocessor: InterleavedTextPreprocessor
    image_preprocessor: Any = None
    multi_image_preprocessor: Any = None
    _output_shapes: dict[str, TensorSpec] | None = None

    @classmethod
    def build(cls, *args, text_seq_len=None, **kwargs) -> MultimodalPreprocessor:
        preprocessor = cls(*args, **kwargs)
        if text_seq_len is not None and preprocessor.text_preprocessor.max_sequence_length is None:
            visual = [
                item
                for item in (preprocessor.image_preprocessor, preprocessor.multi_image_preprocessor)
                if item is not None
            ]
            if not visual:
                raise ValueError("MolmoBot requires an image preprocessor.")
            mm_text_len = max(item.get_output_shapes()["tokens"].shape[0] for item in visual)
            preprocessor.text_preprocessor.max_sequence_length = mm_text_len + text_seq_len
        return preprocessor

    def __call__(
        self,
        messages: list[str],
        *,
        image=None,
        image_group=None,
        is_training: bool = False,
    ) -> dict[str, np.ndarray]:
        if is_training:
            raise ValueError("MolmoBot's in-tree preprocessor supports inference only.")
        if image is not None and image_group is not None:
            raise ValueError("Provide either a single image or an image group, not both.")

        tokenized: TokenizedVisionData | list[TokenizedVisionData] | None
        if image is not None:
            if self.image_preprocessor is None:
                raise ValueError("This checkpoint does not support single-image input.")
            tokenized = self.image_preprocessor(image, is_training=False)
        elif image_group is not None:
            if self.multi_image_preprocessor is None:
                raise ValueError("This checkpoint does not support multi-image input.")
            tokenized = self.multi_image_preprocessor(image_group, is_training=False)
        else:
            tokenized = None

        if tokenized is None:
            return self.text_preprocessor.tokenize_and_interleave(messages, [])
        if isinstance(tokenized, list):
            position_ids = None if tokenized[0].position_ids is None else [item.position_ids for item in tokenized]
            example = self.text_preprocessor.tokenize_and_interleave(
                messages,
                [item.tokens for item in tokenized],
                position_ids,
            )
            if tokenized[0].image_masks is not None:
                example["image_masks"] = np.concatenate([item.image_masks for item in tokenized])
            if tokenized[0].images is not None:
                images = []
                pooling = []
                for item in tokenized:
                    offset = sum(np.prod(previous.shape[:2]) for previous in images)
                    pooling.append(np.where(item.token_pooling >= 0, item.token_pooling + offset, item.token_pooling))
                    images.append(item.images)
                example["images"] = np.concatenate(images)
                example["token_pooling"] = np.concatenate(pooling)
            return example

        example = self.text_preprocessor.tokenize_and_interleave(
            messages,
            [tokenized.tokens],
            None if tokenized.position_ids is None else [tokenized.position_ids],
        )
        if tokenized.images is not None:
            example["images"] = tokenized.images
        if tokenized.image_masks is not None:
            example["image_masks"] = tokenized.image_masks
        if tokenized.token_pooling is not None:
            example["token_pooling"] = tokenized.token_pooling
        if tokenized.low_res_token_pooling is not None:
            example["low_res_token_pooling"] = tokenized.low_res_token_pooling
        if tokenized.other_data is not None:
            example.update(tokenized.other_data)
        return example

    def get_output_shapes(self) -> dict[str, TensorSpec]:
        if self._output_shapes is not None:
            return self._output_shapes
        preprocessors = [item for item in (self.image_preprocessor, self.multi_image_preprocessor) if item is not None]
        if not preprocessors:
            raise ValueError("MolmoBot requires at least one image preprocessor.")
        spec = TensorSpec.max_dictionaries(*(item.get_output_shapes() for item in preprocessors))
        max_seq_len = self.text_preprocessor.max_sequence_length
        if max_seq_len:
            if spec["tokens"].shape[0] > max_seq_len:
                raise ValueError(
                    f"Visual token bound {spec['tokens'].shape[0]} exceeds max sequence length {max_seq_len}."
                )
            spec["tokens"] = TensorSpec([max_seq_len], np.int64)
        else:
            spec["tokens"] = TensorSpec([None], np.int64)
        self._output_shapes = spec
        return spec


@dataclasses.dataclass
class ExamplePreprocessor:
    formatter: DataFormatter
    preprocessor: MultimodalPreprocessor
    for_inference: bool = True
    is_training: bool = False
    include_image: bool = False

    def __post_init__(self):
        if self.is_training or not self.for_inference:
            raise ValueError("MolmoBot's in-tree ExamplePreprocessor supports inference only.")

    def get_output_shapes(self) -> dict[str, TensorSpec]:
        return self.preprocessor.get_output_shapes()

    @property
    def tokenizer(self):
        return self.preprocessor.text_preprocessor.tokenizer

    def __call__(self, example, rng=np.random):
        example = dict(example)
        if "video" in example or "action" in example or "action_is_pad" in example:
            raise ValueError("Video decoding and action-label preprocessing are not part of MolmoBot inference.")
        if "image" not in example:
            raise ValueError("MolmoBot requires an image or image group.")

        raw_image = example["image"]
        if isinstance(raw_image, (list, tuple)):
            image = None
            image_group = [load_image(item) for item in raw_image]
        else:
            image = load_image(raw_image)
            image_group = None
            example["image"] = image

        messages, formatter_metadata = self.formatter(
            example,
            is_training=False,
            for_inference=True,
            rng=rng,
        )
        output = self.preprocessor(
            messages,
            image=image,
            image_group=image_group,
            is_training=False,
        )
        state = example.get("state")
        if state is not None:
            output["states"] = np.asarray(state, dtype=np.float32)
        if formatter_metadata or "metadata" in example:
            metadata = dict(example.get("metadata") or {})
            metadata.update(formatter_metadata or {})
            output["metadata"] = metadata
        return output


__all__ = ["ExamplePreprocessor", "MultimodalPreprocessor"]
