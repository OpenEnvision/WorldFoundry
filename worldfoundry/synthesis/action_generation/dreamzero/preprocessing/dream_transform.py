# Inference-only model preprocessing adapted from the official DreamZero source.
import ast
import html

import ftfy
import numpy as np
import regex as re
import torch
import tree
from einops import rearrange
from pydantic import Field, PrivateAttr
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.base import InvertibleModalityTransform
from worldfoundry.core.io.paths import resolve_local_hf_model_path

from .language import formalize_language
from .schema import DatasetMetadata, EmbodimentTag


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class HuggingfaceTokenizer:
    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, "whitespace")
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        load_kwargs = dict(kwargs)
        tokenizer_dir = resolve_local_hf_model_path(
            name,
            required_files=("tokenizer_config.json",),
        )
        load_kwargs["local_files_only"] = True
        load_kwargs["trust_remote_code"] = False
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, **load_kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop("return_mask", False)

        # arguments
        _kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            _kwargs.update({"padding": "max_length", "truncation": True, "max_length": self.seq_len})
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == "whitespace":
            text = whitespace_clean(basic_clean(text))
        # elif self.clean == 'lower':
        #     text = whitespace_clean(basic_clean(text)).lower()
        # elif self.clean == 'canonicalize':
        #     text = canonicalize(basic_clean(text))
        return text


def collate(features: list[dict], tokenizer: AutoTokenizer, num_views=3, embodiment_tag_mapping=None) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        if key == "text":
            output_values = []
            for elem in features:
                item = elem[key]
                try:
                    parsed_item = ast.literal_eval(item)
                    # Handle different return types from ast.literal_eval
                    if isinstance(parsed_item, (list, tuple)):  # noqa: SIM108 - Branches carry per-case comments.
                        processed_item = str(parsed_item[0])
                    else:
                        # If it's already a scalar (string, float, int, etc.), convert to string
                        processed_item = str(parsed_item)

                    if num_views > 1 and elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.AGIBOT.value]:
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into four views: The top-left view shows the camera view from the robot's head, the top-right view shows the camera view from the right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right view is a black screen (inactive view). The robot "
                            + processed_item.lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.OXE_DROID.value]:
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into three views: The top view shows the camera view from the robot's wrist, the bottom-left view shows the camera view from the left exterior camera, and the bottom-right view shows the camera view from the right exterior camera. During training, one of the two bottom exterior views may be a black screen (dropped view). The robot "
                            + processed_item.lower()
                        )
                    elif (
                        elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.GR1_UNIFIED.value]
                        or elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.MECKA_HANDS.value]
                    ):
                        processed_item = "A single view video shows that a human " + processed_item.lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.XDOF.value]:
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into four views: The top-left view shows the camera view from the robot's head, the top-right view shows the camera view from the right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right view is a black screen (inactive view). The robot "
                            + processed_item.lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.YAM.value]:
                        processed_item = (
                            "A multi-view video shows that a robot "
                            + processed_item.lower()
                            + " The video is split into four views: The top-left view shows the top camera, the top-right view shows the right camera, the bottom-left view shows the left camera, and the bottom-right view is a black screen. The robot "
                            + processed_item.lower()
                        )
                    else:
                        raise ValueError(f"Embodiment ID {elem['embodiment_id']} not supported.")
                    output_values.append(processed_item)
                except (ValueError, SyntaxError, TypeError):
                    # If parsing fails or item is already a string, use it directly
                    if num_views > 1 and elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.AGIBOT.value]:
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into four views: The top-left view shows the camera view from the robot's head, the top-right view shows the camera view from the right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right view is a black screen (inactive view). The robot "
                            + str(item).lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.OXE_DROID.value]:
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into three views: The top view shows the camera view from the robot's wrist, the bottom-left view shows the camera view from the left exterior camera, and the bottom-right view shows the camera view from the right exterior camera. During training, one of the two bottom exterior views may be a black screen (dropped view). The robot "
                            + str(item).lower()
                        )
                    elif (
                        elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.GR1_UNIFIED.value]
                        or elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.MECKA_HANDS.value]
                    ):
                        item = "A single view video shows that a human " + str(item).lower()
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.XDOF.value]:
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into four views: The top-left view shows the camera view from the robot's head, the top-right view shows the camera view from the right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right view is a black screen (inactive view). The robot "
                            + str(item).lower()
                        )
                    elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.YAM.value]:
                        item = (
                            "A multi-view video shows that a robot "
                            + str(item).lower()
                            + " The video is split into four views: The top-left view shows the top camera, the top-right view shows the right camera, the bottom-left view shows the left camera, and the bottom-right view is a black screen. The robot "
                            + str(item).lower()
                        )
                    else:
                        raise ValueError(f"Embodiment ID {elem['embodiment_id']} not supported.")  # noqa: B904 - Keeps the implicit exception context.
                    output_values.append(item)
            # print("output_values", output_values)
            ids, mask = tokenizer(output_values, return_mask=True, add_special_tokens=True)
            batch[key] = ids
            batch["text_attention_mask"] = mask
        elif key == "text_negative":
            values = [elem[key] for elem in features]
            ids, mask = tokenizer(values, return_mask=True, add_special_tokens=True)
            batch[key] = ids
            batch["text_attention_mask_negative"] = mask
        else:
            values = [elem[key] for elem in features]
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class DreamTransform(InvertibleModalityTransform):
    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(default_factory=list, description="Not used in this transform, kept for compatibility.")
    formalize_language: bool = Field(default=False, description="Formalize language if True.")

    embodiment_tag_mapping: dict[str, int] = Field(
        default_factory=dict,
        description="The projector index of each embodiment tag.",
    )

    always_use_default_instruction: bool = Field(
        default=False,
        description="Whether to always use the default instruction. For studying how much the language helps.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: str | None = PrivateAttr(default=None)
    _language_keys: list[str] | None = PrivateAttr(default=None)

    # XEmbDiT arguments
    default_instruction: str
    max_state_dim: int
    max_action_dim: int
    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None
    state_horizon: int
    action_horizon: int
    num_views: int = 3

    # Add tokenizer attribute
    tokenizer_path: str = Field(default="google/umt5-xxl", description="Path to the tokenizer.")
    _tokenizer: HuggingfaceTokenizer | None = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the tokenizer
        self._tokenizer = HuggingfaceTokenizer(name=self.tokenizer_path, seq_len=self.max_length, clean="whitespace")

    @property
    def tokenizer(self):
        return self._tokenizer

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert self.embodiment_tag is not None, "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():  # noqa: SIM118 - Keeps the keys view for mappings with custom membership.
            try:
                modality, _ = key.split(".")
                if "annotation" in key:
                    modality = "language"
            except:  # noqa: E722 - Also catches BaseException, as upstream.
                ### Handle language annotation special case
                if "annotation" in key:  # noqa: SIM108 - Branches carry per-case comments.
                    modality = "language"
                else:
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            self._language_keys = language_keys  # Store all keys for random selection
            if len(language_keys) == 1:
                self._language_key = language_keys[0]
            else:
                self._language_key = None  # Will be selected randomly in _prepare_language
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        images = batch["images"]  # [V, T, C, H, W]

        np_images = rearrange(images, "v t c h w -> (t v) h w c")
        if "language" in batch:
            lang = batch["language"]
            if isinstance(lang, (list, np.ndarray)):
                lang = lang[0]

        inputs = {}
        inputs["images"] = np_images
        inputs["text"] = lang

        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        if images.shape[0] > 1:
            v, t, c, h, w = images.shape

            # For DROID embodiment: 2x2 grid where the wrist view spans the full top row,
            # and the two exterior views occupy the bottom row.
            #
            # View indices (expected):
            # - View 0: left exterior
            # - View 1: right exterior
            # - View 2: wrist
            #
            # Layout:
            #   [wrist, wrist]     (wrist duplicated to have 2x width)
            #   [left_ext | right_ext]
            #
            if self.embodiment_tag == EmbodimentTag.OXE_DROID and v >= 3:
                left_exterior = images[0]  # (t, c, h, w)
                right_exterior = images[1]  # (t, c, h, w)
                wrist_image = images[2]  # (t, c, h, w)

                concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)

                # Top row: a SINGLE wrist view, resized to be 2x wider (same height).
                # We use nearest-neighbor upscaling by repeating pixels along width.
                wrist_wide = np.repeat(wrist_image, 2, axis=-1)  # (t, c, h, 2w)
                concat_images[0, :, :, :h, :] = wrist_wide

                concat_images[0, :, :, h:, :w] = left_exterior
                concat_images[0, :, :, h:, w:] = right_exterior

                return concat_images

            # For other embodiments: use 2x2 grid layout
            # Layout: [head, right]
            #         [left, black]

            # Create output tensor with doubled height and width
            concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)

            # Place images in the 2x2 grid
            # Left upper: head image (view 0)
            if v > 0:
                concat_images[0, :, :, :h, :w] = images[0]

            # Left bottom: left image (view 1)
            if v > 1:
                concat_images[0, :, :, h:, :w] = images[1]

            # Right top: right image (view 2)
            if v > 2:
                concat_images[0, :, :, :h, w:] = images[2]

            # Right bottom: black pixels (already zeros from initialization)

            return concat_images

        return images

    def _prepare_language(self, data: dict) -> tuple[str, bool]:
        selected_key = self._language_key
        if selected_key is None and self._language_keys:
            selected_key = self._language_keys[0]
        raw_language = data.get(selected_key, self.default_instruction)
        if isinstance(raw_language, np.ndarray):
            raw_language = raw_language.item() if raw_language.size == 1 else raw_language[0]
        if isinstance(raw_language, list):
            raw_language = raw_language[0]
        raw_language = str(raw_language)
        is_cotrain_instance = "<COTRAIN>" in raw_language
        for marker in ("<LAPA>", "<DREAM>", "<COTRAIN>"):
            raw_language = raw_language.replace(marker, "")
        if self.always_use_default_instruction:
            raw_language = self.default_instruction
        if self.formalize_language:
            raw_language = formalize_language(raw_language)
        return raw_language, is_cotrain_instance

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """

        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] % self.state_horizon == 0, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def apply_single(self, data: dict) -> dict:
        images = self._prepare_video(data).astype(np.uint8)
        language, is_cotrain_instance = self._prepare_language(data)
        vlm_outputs = self._apply_vlm_processing({"images": images, "language": language})
        state, state_mask, _ = self._prepare_state(data)
        transformed_data = {
            "state": state,
            "state_mask": state_mask,
            "text_negative": (
                "Vibrant colors, overexposed, static, blurry details, text, subtitles, "
                "style, artwork, painting, image, still, grayscale, dull, worst quality, "
                "low quality, JPEG artifacts, ugly, mutilated, deformed, disfigured."
            ),
            "embodiment_id": self.get_embodiment_tag(),
            "is_cotrain_instance": np.asarray(
                is_cotrain_instance or self.embodiment_tag == EmbodimentTag.MECKA_HANDS,
                dtype=bool,
            ),
        }
        for key, value in vlm_outputs.items():
            if key in transformed_data:
                raise KeyError(f"Duplicate DreamZero model input: {key}")
            transformed_data[key] = value
        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]  # noqa: B023 - Keeps the upstream late binding of the loop variable.
        processed = [self.apply_single(item) for item in data_split]
        return collate(processed, self.tokenizer, self.num_views, self.embodiment_tag_mapping)

    def apply(self, data: dict) -> dict:
        if data["video"].ndim == 5:
            data["video"] = data["video"][None, ...]
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        return self.apply_batch(data, batch_size) if is_batched else self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
