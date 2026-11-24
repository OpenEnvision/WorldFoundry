"""Utils for evaluating the OpenVLA policy."""

import json
import os
from collections import deque
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from PIL import Image
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from veomni.models.vla.pi0 import PI0Policy, QwenPI0Policy

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
DEFAULT_NORM_STATS_PATH = "norm_stats.json"


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history, strict=False)
                ]
            )

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions - 1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()

        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action


def center_crop_image(image: np.ndarray | Image.Image) -> Image.Image:
    crop_scale = 0.9
    side_scale = float(np.sqrt(np.clip(crop_scale, 0.0, 1.0)))  # side length scale
    out_size = (224, 224)

    # Convert input to PIL Image
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype.kind == "f":
            # If floats likely in [0,1], map to [0,255]
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        elif arr.dtype == np.uint16:
            # Map 16-bit to 8-bit
            arr = (arr / 257).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        pil = Image.fromarray(arr)
    elif isinstance(image, Image.Image):
        pil = image
    else:
        raise TypeError("image must be a numpy array or PIL.Image.Image")

    # Force RGB for consistent output
    pil = pil.convert("RGB")
    W, H = pil.size

    # Compute centered crop box (integer pixels)
    crop_w = max(1, int(round(W * side_scale)))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
    crop_h = max(1, int(round(H * side_scale)))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
    left = (W - crop_w) // 2
    top = (H - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    cropped = pil.crop((left, top, right, bottom))
    resized = cropped.resize(out_size, resample=Image.BILINEAR)
    return resized


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    # channel last to channel first if necessary
    if img.shape[1] not in (1, 3) and img.shape[-1] in (1, 3):
        img = img.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


class PolicyPreprocessMixin:
    """
    A mixin class that provides preprocessing utilities for observations.
    Can be mixed into any policy class to add image, state, action, language handling.
    """

    def prepare_images(self, observation: dict[str, Tensor]):
        """Normalize, resize, and pad images and stack them into a tensor.

        Args:
            observation (dict[str, Tensor])

        Returns:
            images (torch.Tensor): (*b, n, c, h, w) images in range [-1.0, 1.0]
            img_masks (torch.Tensor): (*b, n) masks for images, True if image is present, False if missing
        """
        _dtype = observation["state"].dtype
        _bsize = observation["state"].shape[0]
        device = observation["state"].device
        images, img_masks = [], []
        for key in IMAGE_KEYS:
            if key in observation["image"]:
                # resize, pad, and normalize
                img = observation["image"][key]  # torch.Size([1, 3, 224, 224])

                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)

                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
                img = self.image_processor(img)["pixel_values"]
                images.append(img)
                img_masks.append(True)
            else:
                img = np.zeros_like(img)
                images.append(img)
                img_masks.append(False)
        # import ipdb; ipdb.set_trace()
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0).to(device=device)
        elif isinstance(images[0], np.ndarray):
            images = torch.from_numpy(np.stack(images, axis=0)).to(device=device)  # torch.Size([3, 256, 1176])
        img_masks = torch.tensor(img_masks, dtype=torch.bool).to(device=device)  # (*b, n)

        return images, img_masks

    def prepare_state(self, observation):
        state = torch.from_numpy(observation["state"])
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = F.pad(state, (0, self.config.max_state_dim - state.shape[1]))
        return state

    def prepare_language(self, observation: dict[str, Tensor]):
        """If `prompt` is provided, modify it to PaliGemma format and tokenize it.
        If `lang_tokens` and `lang_masks` are provided, use them directly.

        PaliGemma expects prefix prompts to be formatted as:
        <images> .... <images> <bos> prompt <sep>, where <sep> uses `\\n`.
        So here we format the prompt to start with `<bos>` and end with `\\n`.
        Later, we will concatenate the images and language tokens into a single sequence.

        Args:
            observation (dict[str, Tensor])

        Returns:
            lang_tokens (torch.Tensor): (*b, l) language tokens
            lang_masks (torch.Tensor): (*b, l) masks for language tokens, True if token is present, False if missing
        """
        lang_tokens = observation.get("lang_tokens")
        lang_masks = observation.get("lang_masks")
        prompt = observation.get("prompt")

        # either provide `prompt` or (`lang_tokens`, `lang_masks`)
        if prompt is None and (lang_tokens is None or lang_masks is None):
            raise ValueError("Either 'prompt' or ('lang_tokens', 'lang_masks') must be provided in the observation.")

        device = observation["state"].device
        if prompt is not None and (lang_tokens is None or lang_masks is None):
            prompt = [p if p.startswith("<bos>") else f"<bos>{p}" for p in prompt]
            prompt = [p if p.endswith("\n") else f"{p}\n" for p in prompt]
            tokenized_prompt = self.language_tokenizer.__call__(
                prompt,
                padding="max_length",
                padding_side="right",
                max_length=self.config.tokenizer_max_length,
                return_tensors="pt",
            )
            lang_tokens = tokenized_prompt["input_ids"].to(device=device)
            lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)
        else:
            lang_tokens = observation["lang_tokens"].to(device=device)
            lang_masks = observation["lang_masks"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    @torch.no_grad
    def select_action(self, observation: dict[str, Tensor], noise: Tensor | None = None):
        self.eval()
        images, img_masks = self.prepare_images(observation)
        state = self.prepare_state(observation)
        lang_tokens, lang_masks = self.prepare_language(observation)
        device = "cuda"
        dtype = torch.bfloat16

        actions = self.model.sample_actions(
            images.to(dtype=dtype, device=device),
            img_masks.to(device=device),
            lang_tokens.to(device=device),
            lang_masks.to(device=device),
            state.to(dtype=dtype, device=device),
        )
        return actions


class QwenPI0InferencePolicy(PolicyPreprocessMixin, QwenPI0Policy):
    """Qwen inference policy implementation."""

    # Only combine necessary functions


class PI0InfernecePolicy(PolicyPreprocessMixin, PI0Policy):
    """Infernece policy implementation."""

    # Only combine necessary functions


def merge_qwen_config(policy_config, qwen_config):
    config_dict = qwen_config.to_dict() if hasattr(qwen_config, "to_dict") else qwen_config

    text_keys = {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
        "tokenizer_path",
    }

    for key in text_keys:
        if key in config_dict:
            setattr(policy_config, key, config_dict[key])
            print(f"✅ Merged LLM: {key} = {config_dict[key]}")

    if "vision_config" in config_dict:
        policy_config.vision_config = qwen_config.vision_config
    else:
        print("⚠️ Warning: 'vision_config' not found in qwen_config!")

    return policy_config


class QwenPiServer:
    """
    policy wrapper to support action ensemble or chunk execution
    """

    def __init__(
        self,
        path_to_pi_model="",
        adaptive_ensemble_alpha=0.1,
        action_ensemble_horizon=8,
        use_length=1,  # to control the execution length of the action chunk, -1 denotes using action ensemble
        use_bf16=True,
    ) -> None:
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.use_length = use_length

        self.task_description = None

        self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)

        self.vla = self.load_vla(path_to_pi_model)
        self.vla = self.vla.to("cuda").eval()
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
        self.global_step = 0
        self.last_action_chunk = None

    def init_norm(self, states_path=DEFAULT_NORM_STATS_PATH, state_dim=14, action_dim=14):
        """
        TODO: show be rewritten as a dict
        """
        with open(states_path) as f:
            norm_stats = json.load(f)["stack_bowls_three-aloha-agilex_clean_50_rep"]
        self.state_mean = np.array(norm_stats["norm_stats"]["state"]["mean"][:state_dim], dtype=np.float32)
        self.state_std = np.array(norm_stats["norm_stats"]["state"]["std"][:state_dim], dtype=np.float32)
        self.action_mean = np.array(norm_stats["norm_stats"]["actions"]["mean"][:action_dim], dtype=np.float32)
        self.action_std = np.array(norm_stats["norm_stats"]["actions"]["std"][:action_dim], dtype=np.float32)

    def state_normalizer(self, unnorm_state):
        state = (unnorm_state - self.state_mean) / (self.state_std + 1e-6)
        return state

    def action_unnormalizer(self, norm_action):
        action = norm_action * (self.action_std + 1e-6) + self.action_mean
        return action

    def load_vla(self, path_to_pi_model) -> QwenPI0Policy:
        # load model
        print(f"loading model from: {path_to_pi_model}")
        config = PreTrainedConfig.from_pretrained(path_to_pi_model)

        base_model_path = os.environ["WORLDFOUNDRY_QWEN_PI_BASE_MODEL_PATH"]

        qwen_config = AutoConfig.from_pretrained(base_model_path)
        qwen_config.tokenizer_path = base_model_path
        # merge transformers extra config to lerobot
        config = merge_qwen_config(config, qwen_config)

        # load processors
        language_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        processor = AutoProcessor.from_pretrained(base_model_path)

        print("Initializing model ... ")

        if "qwen" in path_to_pi_model:  # noqa: SIM108 - Branches carry per-case comments.
            policy = QwenPI0InferencePolicy(config)
        else:
            # Shuai: Not tested yet
            policy = PI0InfernecePolicy(config)

        # Merge multiple safetensor weights
        all_safetensors = glob(os.path.join(path_to_pi_model, "*.safetensors"))
        merged_weights = {}

        for file_path in tqdm(all_safetensors):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118 - Keeps the keys view for mappings with custom membership.
                    merged_weights[key] = f.get_tensor(key)

        policy.load_state_dict(merged_weights, strict=True)

        # Load data processors
        policy.language_tokenizer = language_tokenizer
        policy.processor = processor
        policy.image_processor = processor.image_processor

        print("Model initialized ... ")

        self.init_norm()
        return policy

    def reset(self) -> None:
        if self.use_length == -1:
            self.action_ensembler.reset()

        self.global_step = 0
        self.last_action_chunk = None

    def infer(self, observation, center_crop=True):
        """Generates an action with the VLA policy."""

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if observation.get("reset"):
            self.reset()
            return {"action": None}

        unnorm_state = observation["state"]
        observation["state"] = self.state_normalizer(observation["state"])

        if self.use_length == -1 or self.global_step % self.use_length == 0:
            normalized_actions = self.vla.select_action(observation).squeeze(0)
            self.last_action_chunk = normalized_actions.float().cpu().numpy()
            self.last_state = unnorm_state

        if self.use_length > 0:
            action = self.last_action_chunk[self.global_step % self.use_length]
        elif self.use_length == -1:  # do ensemble
            action = self.action_ensembler.ensemble_action(normalized_actions)

        action = self.action_unnormalizer(action[:14])  # TODO: remove the hard code dim!
        action = action[:14]  # + self.last_state[0,:14]
        print(f"on server step: {self.global_step} state {observation['state'][:7]} action {action[:7]}")

        self.global_step += 1

        return {"action": action}
