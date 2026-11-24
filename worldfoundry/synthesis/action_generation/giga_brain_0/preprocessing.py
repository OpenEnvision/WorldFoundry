"""GigaBrain-0 inference preprocessing and action postprocessing."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.synthesis.action_generation.openpi.modeling.action_tokenizer import (
    UniversalActionTokenizer,
)


class Normalize:
    """Normalizes a tensor using mean/std or quantile-based scaling."""

    def __init__(
        self, stats: dict[int, dict[str, list[float]]], *, use_quantiles: bool = False, enable_clamp: bool = False
    ):
        """Initializes the normalization transform.

        Args:
            stats: A dictionary mapping embodiment IDs to normalization statistics.
            use_quantiles: If True, use 1% and 99% quantiles for scaling.
                Otherwise, use mean and standard deviation.
            enable_clamp: If True, clamp the output to [-1, 1].
        """
        self.EPSILON = 1e-6
        self.use_quantiles = use_quantiles
        self.enable_clamp = enable_clamp

        required_attrs = ["mean", "std"]
        if self.use_quantiles:
            required_attrs = ["q01", "q99"]

        for attr in required_attrs:
            for key in stats:
                if attr not in stats[key]:
                    raise AttributeError(f"stats object is missing the following attribute: {attr}")

        if self.use_quantiles:
            self.q01 = {}
            self.q99 = {}
            for key in stats:
                self.q01[int(key)] = torch.tensor(stats[key]["q01"], dtype=torch.float32)
                self.q99[int(key)] = torch.tensor(stats[key]["q99"], dtype=torch.float32)
        else:
            self.mean = {}
            self.std = {}
            for key in stats:
                self.mean[int(key)] = torch.tensor(stats[key]["mean"], dtype=torch.float32)
                self.std[int(key)] = torch.tensor(stats[key]["std"], dtype=torch.float32)

    def to(self, device: str | torch.device):
        if self.use_quantiles:
            for key in self.q01:
                self.q01[key] = self.q01[key].to(device)
            for key in self.q99:
                self.q99[key] = self.q99[key].to(device)
        else:
            for key in self.mean:
                self.mean[key] = self.mean[key].to(device)
            for key in self.std:
                self.std[key] = self.std[key].to(device)
        return self

    def __call__(self, x: torch.Tensor, embodiment_id: int = 0) -> torch.Tensor:
        """Applies normalization to the input tensor.

        Args:
            x: The input tensor to normalize.
            embodiment_id: The embodiment ID to use for selecting normalization stats.

        Returns:
            The normalized tensor.
        """
        x_dim = x.shape[-1]
        if self.use_quantiles:
            x = (x - self.q01[embodiment_id][..., :x_dim]) / (
                self.q99[embodiment_id][..., :x_dim] - self.q01[embodiment_id][..., :x_dim] + self.EPSILON
            ) * 2.0 - 1.0
        else:
            x = (x - self.mean[embodiment_id][..., :x_dim]) / (self.std[embodiment_id][..., :x_dim] + self.EPSILON)

        if self.enable_clamp:
            x = x.clamp(-1.0, 1.0)

        return x


class Unnormalize:
    """Unnormalizes a tensor using mean/std or quantile-based scaling."""

    def __init__(self, stats: dict[int, dict[str, list[float]]], *, use_quantiles: bool = False):
        """Initializes the unnormalization transform.

        Args:
            stats: A dictionary mapping embodiment IDs to normalization statistics.
            use_quantiles: If True, use 1% and 99% quantiles for scaling.
                Otherwise, use mean and standard deviation.
        """
        self.EPSILON = 1e-6
        self.stats = stats
        self.use_quantiles = use_quantiles

        required_attrs = ["mean", "std"]
        if self.use_quantiles:
            required_attrs = ["q01", "q99"]

        for attr in required_attrs:
            for key in stats:
                if attr not in stats[key]:
                    raise AttributeError(f"stats object is missing the following attribute: {attr}")

        if self.use_quantiles:
            self.q01 = {}
            self.q99 = {}
            for key in stats:
                self.q01[int(key)] = torch.tensor(stats[key]["q01"], dtype=torch.float32)
                self.q99[int(key)] = torch.tensor(stats[key]["q99"], dtype=torch.float32)
        else:
            self.mean = {}
            self.std = {}
            for key in stats:
                self.mean[int(key)] = torch.tensor(stats[key]["mean"], dtype=torch.float32)
                self.std[int(key)] = torch.tensor(stats[key]["std"], dtype=torch.float32)

    def to(self, device: str | torch.device):
        if self.use_quantiles:
            for key in self.q01:
                self.q01[key] = self.q01[key].to(device)
            for key in self.q99:
                self.q99[key] = self.q99[key].to(device)
        else:
            for key in self.mean:
                self.mean[key] = self.mean[key].to(device)
            for key in self.std:
                self.std[key] = self.std[key].to(device)
        return self

    def __call__(self, x: torch.Tensor, embodiment_id: int = 0) -> torch.Tensor:
        """Applies unnormalization to the input tensor.

        Args:
            x: The input tensor to unnormalize.
            embodiment_id: The embodiment ID to use for selecting normalization stats.

        Returns:
            The unnormalized tensor.
        """
        x_dim = x.shape[-1]
        if self.use_quantiles:
            return (x + 1.0) / 2.0 * (
                self.q99[embodiment_id][..., :x_dim] - self.q01[embodiment_id][..., :x_dim] + self.EPSILON
            ) + self.q01[embodiment_id][..., :x_dim]
        else:
            return x * (self.std[embodiment_id][..., :x_dim] + self.EPSILON) + self.mean[embodiment_id][..., :x_dim]


class AbsoluteActions:
    """Repacks delta actions into absolute action space."""

    def __init__(self, mask: dict[int, list[bool]]):
        self.mask = {}
        assert mask is not None, "mask is required"
        for key in mask:
            self.mask[int(key)] = torch.tensor(mask[key])

    def to(self, device: str | torch.device):
        for key in self.mask:
            self.mask[key] = self.mask[key].to(device)
        return self

    def __call__(self, data: dict) -> dict:
        if "action" not in data or "observation.state" not in data:
            return data

        embodiment_id = data["embodiment_id"]

        state, action = data["observation.state"], data["action"]
        dims = self.mask[embodiment_id].shape[-1]
        action[..., :dims] += torch.where(
            self.mask[embodiment_id], state[..., :dims], torch.zeros_like(state[..., :dims])
        ).unsqueeze(-2)
        data["action"] = action
        return data


class PadStatesAndActions:
    """Zero-pads states and actions to the model action dimension."""

    def __init__(self, action_dim: int):
        """Initializes the padding transform.

        Args:
            action_dim: The target dimension to pad to.
        """
        self.action_dim = action_dim

    def _pad_to_dim(self, x: torch.Tensor, target_dim: int, axis: int = -1) -> torch.Tensor:
        """Pad an array to the target dimension with zeros along the specified
        axis."""
        current_dim = x.shape[axis]
        if current_dim < target_dim:
            shape = list(x.shape)
            shape[-1] = target_dim
            new_vector = torch.zeros(*shape, dtype=x.dtype, device=x.device)
            new_vector[..., :current_dim] = x
            x = new_vector
        return x

    def __call__(self, data: dict) -> dict:
        """Pads 'observation.state' and 'action' tensors in the data dict.

        Args:
            data: A dictionary containing 'observation.state' and optionally 'action'.

        Returns:
            The data dictionary with padded tensors.
        """
        data["observation.state"] = self._pad_to_dim(data["observation.state"], self.action_dim, axis=-1)
        if "action" in data:
            data["action"] = self._pad_to_dim(data["action"], self.action_dim, axis=-1)
        return data


# This function is used for the original PaliGemma model inference.


def resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = -1.0) -> tuple[torch.Tensor, dict]:
    """Resize an image to fit inside the given (width, height) while preserving
    aspect ratio, then pad with the specified value so that the final image
    exactly matches the target size.

    Args:
        img: Input image, shape (C, H, W), with values typically in [0, 1].
        width: Target width (W).
        height: Target height (H).
        pad_value: Value to use for padding, defaults to -1.

    Returns:
        A tuple containing:
        - A torch.Tensor of shape (C, height, width).
        - A dictionary with transformation parameters.
    """
    # Validate input dimensions
    if img.ndim != 3:
        raise ValueError(f"(C,H,W) expected, but got {img.shape}")

    cur_height, cur_width = img.shape[1:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img.unsqueeze(0), size=(resized_height, resized_width), mode="bilinear", align_corners=False
    ).squeeze(0)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded_img = F.pad(resized_img, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    transform_params = {
        "original_size": (cur_width, cur_height),
        "ratio": ratio,
        "padding": (pad_left, pad_top),
    }
    return padded_img, transform_params


class ImageTransform:
    """Deterministic multi-camera image preprocessing for inference."""

    def __init__(
        self,
        resize_imgs_with_padding: tuple[int, int],
        present_img_keys: list[str] | None = None,
        enable_depth_img: bool = False,
        depth_img_prefix_name: str | None = None,
    ):
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.present_img_keys = present_img_keys or [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
        self.enable_depth_img = enable_depth_img
        self.depth_img_prefix_name = depth_img_prefix_name

    def __call__(
        self,
        data: dict,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
        images = []
        image_masks = []
        transform_params = {}
        for key in self.present_img_keys:
            if key not in data:
                raise KeyError(f"Missing GigaBrain-0 camera input: {key}")
            image = data[key]
            if self.enable_depth_img:
                if self.depth_img_prefix_name is None:
                    raise ValueError("depth_img_prefix_name is required for depth input")
                depth_key = key.replace("observation.images", self.depth_img_prefix_name)
                depth = data.get(depth_key)
                depth = depth[0:1] if depth is not None else torch.zeros_like(image[0:1])
                image = torch.cat([image, depth], dim=0)

            target_width, target_height = self.resize_imgs_with_padding
            if image.shape[-2:] != (target_height, target_width):
                image, params = resize_with_pad(
                    image,
                    target_width,
                    target_height,
                    pad_value=0,
                )
                if key == "observation.images.cam_high":
                    transform_params["resize_with_pad"] = params
            image = image * 2.0 - 1.0
            images.append(image)
            image_masks.append(torch.ones((), dtype=torch.bool, device=image.device))
        return images, image_masks, transform_params


class PromptTokenizerTransform:
    """Create deterministic inference prompts and decode FAST action tokens."""

    def __init__(
        self,
        tokenizer_model_path: str,
        fast_tokenizer_path: str,
        max_length: int,
        discrete_state_input: bool = True,
        include_subtask: bool = True,
        text_token_length: int | None = 257152,
        autoregressive_inference_mode: bool = False,
    ):
        self.device: str | torch.device = "cpu"
        local_tokenizer_path = resolve_local_hf_model_path(
            tokenizer_model_path,
            required_files=("tokenizer_config.json",),
        )
        self.paligemma_tokenizer = AutoTokenizer.from_pretrained(
            local_tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.paligemma_tokenizer.add_bos_token = True
        self.fast_tokenizer = UniversalActionTokenizer.from_pretrained(fast_tokenizer_path)
        self.discrete_state_input = discrete_state_input
        self.include_subtask = include_subtask
        self.fast_skip_tokens = 128
        self.max_length = max_length
        self.text_token_length = text_token_length
        self.autoregressive_inference_mode = autoregressive_inference_mode

    def to(self, device: str | torch.device):
        self.device = device
        return self

    def __call__(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if "task" not in data:
            raise KeyError("GigaBrain-0 prompt input is missing 'task'")
        cleaned = str(data["task"]).lower().strip().replace("_", " ")
        main_task, separator, remainder = cleaned.partition(" subtask: ")
        subtask = remainder.split("\n", 1)[0] if separator else None
        state = data.get("observation.state")

        if self.discrete_state_input and state is not None:
            bins = torch.linspace(-1, 1, 257, device=self.device)[:-1]
            discretized = torch.bucketize(state, bins) - 1
            state_text = " ".join(str(value.item()) for value in discretized)
            if self.include_subtask and subtask:
                prompt = f"Task: {main_task}, Subtask: {subtask}, State: {state_text};\n"
            else:
                prompt = f"Task: {main_task}, State: {state_text};\n"
        else:
            prompt = f"Task: {main_task}\n"

        tokenized = self.paligemma_tokenizer(
            [prompt],
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        padded = self.paligemma_tokenizer.pad(
            {
                "input_ids": tokenized["input_ids"][0].tolist(),
                "attention_mask": tokenized["attention_mask"][0].tolist(),
            },
            padding="max_length",
            padding_side="left" if self.autoregressive_inference_mode else "right",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return (
            padded["input_ids"].to(dtype=torch.int32, device=self.device),
            padded["attention_mask"].to(dtype=torch.bool, device=self.device),
        )

    def extract_actions(
        self,
        tokens: list[list[int]],
        action_horizon: int,
        action_dim: int,
    ) -> torch.Tensor:
        if len(tokens) != 1:
            raise ValueError("GigaBrain-0 autoregressive action decoding supports batch size 1")
        sequence = tokens[0].tolist() if hasattr(tokens[0], "tolist") else list(tokens[0])
        bos = (
            self.paligemma_tokenizer("Action: ", add_special_tokens=False, return_tensors="pt")["input_ids"]
            .squeeze(0)
            .tolist()
        )
        eos = (
            self.paligemma_tokenizer("|<eos>", add_special_tokens=False, return_tensors="pt")["input_ids"]
            .squeeze(0)
            .tolist()
        )

        def find(pattern: list[int], start: int = 0) -> int:
            for index in range(start, len(sequence) - len(pattern) + 1):
                if sequence[index : index + len(pattern)] == pattern:
                    return index
            return -1

        start = find(bos)
        if start < 0:
            return torch.zeros((1, 0, action_dim), dtype=torch.float32)
        start += len(bos)
        end = find(eos, start)
        action_ids = sequence[start : end if end >= 0 else len(sequence)]

        vocab_size = self.paligemma_tokenizer.vocab_size
        if self.text_token_length is not None:
            vocab_size = min(vocab_size, self.text_token_length)
        base_token_id = vocab_size - 1 - self.fast_skip_tokens
        fast_tokens = [base_token_id - token for token in action_ids if base_token_id - token >= 0]
        if not fast_tokens:
            return torch.zeros((1, 0, action_dim), dtype=torch.float32)
        decoded = self.fast_tokenizer.decode(
            [fast_tokens],
            time_horizon=action_horizon,
            action_dim=action_dim,
        )
        return torch.as_tensor(decoded, dtype=torch.float32)
