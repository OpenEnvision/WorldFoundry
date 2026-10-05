# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hy-VLA top-level modeling.

Modified by WorldFoundry from the upstream inference implementation: imports
are package-relative so the runtime is self-contained in the WorldFoundry tree;
the flattened HunYuanVL-MoT config is registered in-tree; unused imports and
no-op f-string prefixes were removed for lint compliance. Policy inference
behavior is unchanged.

Defines :class:`HyVLA` (the HuggingFace-style inference entry point) and
:class:`HyVLAFlowMatching` (the inner ``nn.Module`` that owns the dual-tower
VLM + action expert and implements flow-matching sampling). Training forwards,
losses, label preparation, optimizer helpers, and checkpoint writing are
intentionally omitted from this inference-only integration.
"""
import copy
import json
import math
import os
import sys
from collections import deque
from contextlib import suppress
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoConfig, AutoTokenizer

from worldfoundry.core.io.paths import resolve_local_hf_model_path

from .configuration_hunyuan_vl_mot import (
    HunYuanVLMoTConfig,
)
from .configuration_hy_vla import HyVLAConfig
from .modeling_dual_tower import (
    HyDualTower,
    HyDualTowerConfig,
)

with suppress(ValueError):
    AutoConfig.register("hunyuan_vl_mot", HunYuanVLMoTConfig)

# ---------------------------------------------------------------------------
# Batch-dictionary key naming convention used by the data loader and the
# simulator wrapper. Defined here so the modeling code is self-contained.
# ---------------------------------------------------------------------------
OBS_ROBOT = "observation.state"
OBS_IMAGES = "observation.images"

# Single-file safetensors layout used by ``HyVLA.from_pretrained``.
_SAFETENSORS_SINGLE_FILE = "model.safetensors"

# Default Hy-Embodied VLM repository on the HuggingFace Hub. Used as the
# fallback when neither (a) the VLA checkpoint is self-contained nor
# (b) the user supplies a ``vlm_model_path=...`` override at load time.
_DEFAULT_VLM_REPO = "tencent/HY-Embodied-0.5"

def _is_self_contained_vlm_dir(path: str) -> bool:
    """Return True iff ``path`` is a local directory shipping a ``tokenizer.json``
    AND its ``config.json`` (if any) advertises a transformer ``model_type``
    -- i.e. it looks like a vendor VLM directory rather than a HyVLA ckpt
    directory (whose ``config.json`` is the HyVLAConfig schema with no
    ``model_type``).
    """
    if not (path and os.path.isdir(path)):
        return False
    if not os.path.isfile(os.path.join(path, "tokenizer.json")):
        return False
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        # Tokenizer-only dir: still usable as a tokenizer source.
        return True
    try:
        with open(cfg_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        return False
    return bool(data.get("model_type"))

def _resolve_vlm_path(config: HyVLAConfig, override: str | None = None) -> str:
    """Resolve a usable VLM path for tokenizer + AutoConfig loading.

    Priority order:

      1. Explicit ``override`` (kwarg from ``HyVLA.from_pretrained``).
      2. The VLA ckpt's own directory (``config.pretrained_path``) when it
         passes :func:`_is_self_contained_vlm_dir` -- meaning it ships a
         ``tokenizer.json`` and either no ``config.json`` or one that looks
         like a vendor VLM config (has ``model_type``).
      3. ``config.vlm_model_path`` if it is either a HuggingFace repo id or
         a local directory that actually exists.
      4. Built-in default ``_DEFAULT_VLM_REPO``.

    A short stderr line is emitted whenever a non-default source wins, so
    the resolution is never silent.
    """
    candidates: list[tuple[str, str]] = []
    if override:
        candidates.append(("override", override))
    own_path = getattr(config, "pretrained_path", None)
    if own_path and os.path.isdir(own_path) and os.path.isfile(
        os.path.join(own_path, "tokenizer.json")
    ):
        # A HyVLA ckpt is "self-contained" iff it ships a tokenizer.json
        # AND either (a) carries an embedded vendor VLM AutoConfig payload
        # via ``vlm_config_dict``, or (b) the directory itself is a vendor
        # VLM directory (its config.json advertises a transformer
        # ``model_type``). Both variants legitimately want to read the
        # tokenizer from the ckpt directory.
        if getattr(config, "vlm_config_dict", None) or _is_self_contained_vlm_dir(own_path):
            candidates.append(("ckpt-self-contained", own_path))
    ckpt_value = getattr(config, "vlm_model_path", "") or ""
    if ckpt_value:
        candidates.append(("ckpt", ckpt_value))
    candidates.append(("default", _DEFAULT_VLM_REPO))

    for source, path in candidates:
        if not path:
            continue
        if os.path.isdir(path):
            if source != "ckpt":
                print(
                    f"[modeling_hy_vla] VLM path resolved from {source}: {path}",
                    file=sys.stderr, flush=True,
                )
            return path
        try:
            resolved = resolve_local_hf_model_path(
                path,
                required_files=("config.json", "tokenizer.json"),
            )
        except FileNotFoundError:
            resolved = None
        if resolved is not None:
            if source != "ckpt":
                print(
                    f"[modeling_hy_vla] VLM path resolved from {source}: {resolved}",
                    file=sys.stderr, flush=True,
                )
            return str(resolved)
        print(
            f"[modeling_hy_vla] {source} VLM path does not exist locally: {path!r}, "
            f"trying next fallback...",
            file=sys.stderr, flush=True,
        )

    raise ValueError(
        "Could not resolve a usable Hy-Embodied VLM path. Set "
        "Stage HyVLAConfig.vlm_model_path under WORLDFOUNDRY_HFD_ROOT with "
        "hfd, or pass a complete local directory."
    )


# ---------------------------------------------------------------------------
# VLM AutoConfig loading: returns a vendor ``HunYuanVLMoTConfig``
# (text_config + vision_config nested form) so downstream dual_tower
# construction has a single config schema to worry about. There are two
# checkpoint flavours the loader handles, both fork-native:
#
#   (a) Self-contained VLA ckpt (the released layout):
#       embeds ``vlm_config_dict`` with ``model_type=hunyuan_vl_mot`` and a
#       populated ``text_config`` block. We instantiate ``HunYuanVLMoTConfig``
#       directly from the embedded dict -- no disk / network access.
#
#   (b) Bare VLM directory or HF Hub repo id (e.g. ``tencent/HY-Embodied-0.5``):
#       resolved via ``AutoConfig.from_pretrained``. The HY-Embodied-0.5
#       release ships an ``auto_map`` whose target file is not bundled, so
#       we pin ``trust_remote_code=False`` and, on the resulting ValueError,
#       fall back to reading ``config.json`` by hand (after stripping
#       ``auto_map``) and routing through ``HunYuanVLMoTConfig``.
#
# Returns: a ``HunYuanVLMoTConfig`` (always).
# ---------------------------------------------------------------------------


def _load_vlm_autoconfig(config_or_path):
    """Load the upstream VLM ``AutoConfig`` and return a ``HunYuanVLMoTConfig``.

    Accepts either:
      * a ``HyVLAConfig`` instance -- in which case ``config.vlm_config_dict``
        is the authoritative source: no disk / network access is needed.
      * a string ``model_path`` (local dir or HF repo id) -- in which case
        we first try plain ``AutoConfig.from_pretrained``; on failure caused
        by a broken ``auto_map`` (typical for HY-Embodied-0.5), we fall back
        to reading ``config.json`` as a dict and stripping ``auto_map``.
    """
    # ---- Path 1: embedded vlm_config_dict (self-contained VLA ckpt) ----
    if isinstance(config_or_path, HyVLAConfig):
        cfg = config_or_path
        embedded = getattr(cfg, "vlm_config_dict", None)
        if embedded:
            data = dict(embedded)
            mt = data.get("model_type")
            if mt != "hunyuan_vl_mot" or "text_config" not in data:
                raise ValueError(
                    "vlm_config_dict embedded in HyVLAConfig is not in the "
                    "fork-native nested schema (expected "
                    f"model_type='hunyuan_vl_mot' with a 'text_config' "
                    f"block, got model_type={mt!r}, "
                    f"has_text_config={'text_config' in data})."
                )
            print(
                "[modeling_hy_vla] VLM AutoConfig loaded from embedded "
                "vlm_config_dict (nested hunyuan_vl_mot schema).",
                file=sys.stderr, flush=True,
            )
            data.pop("model_type", None)
            return HunYuanVLMoTConfig(**data)
        # Fall through: raw-VLM bootstrap (``pretrain_source`` in
        # {``vlm``, ``scratch``}); resolve from ``cfg.vlm_model_path``.
        model_path = _resolve_vlm_path(cfg)
        if not model_path:
            raise ValueError(
                "_load_vlm_autoconfig: HyVLAConfig has no "
                "``vlm_config_dict`` AND no ``vlm_model_path``. "
                "Self-contained ckpts must embed ``vlm_config_dict``; "
                "raw-VLM bootstrap flows must set ``vlm_model_path``."
            )
    else:
        model_path = str(
            resolve_local_hf_model_path(
                config_or_path,
                required_files=("config.json", "tokenizer.json"),
            )
        )
    # ---- Path 2: AutoConfig.from_pretrained (vendor-registered class) ----
    # ``trust_remote_code=False`` is required so transformers raises a
    # deterministic ValueError on broken ``auto_map`` entries (which our
    # except block below repairs) instead of prompting on stdin.
    try:
        loaded = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=False,
            local_files_only=True,
        )
        if isinstance(loaded, HunYuanVLMoTConfig):
            return loaded
        raise TypeError(
            f"AutoConfig at {model_path!r} dispatched to "
            f"{type(loaded).__name__}, expected HunYuanVLMoTConfig. The "
            "in-tree HunYuanVLMoTConfig must be importable and registered "
            "with Transformers AutoConfig."
        )
    except (OSError, ValueError) as exc:
        msg = str(exc).lower()
        if "auto_map" not in msg and "does not appear to have a file named" not in msg \
                and "trust_remote_code" not in msg:
            raise

    # ---- Path 3: read config.json by hand (auto_map strip) -----------
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found in local VLM directory {model_path!r}")

    with open(cfg_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    data.pop("auto_map", None)
    if data.get("model_type") != "hunyuan_vl_mot":
        raise ValueError(
            f"VLM config.json at {cfg_path!r} has "
            f"model_type={data.get('model_type')!r}, expected "
            "'hunyuan_vl_mot'. The fork-native loader requires the"
            " upstream HY-Embodied schema."
        )
    data.pop("model_type", None)
    return HunYuanVLMoTConfig(**data)


def _get_safe_dtype(dtype: torch.dtype, device: str | torch.device) -> torch.dtype:
    """Return ``dtype`` clamped to one supported on ``device``.

    MPS does not support float64; everything else does.
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    return dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = _get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1, mode="bilinear"):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    interpolate_params = {
        'size': (resized_height, resized_width),
        'mode': mode,
    }

    if mode != "nearest":
        interpolate_params['align_corners'] = False

    resized_img = F.interpolate(img, **interpolate_params)


    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on the center of image
    pw = pad_width // 2
    ph = pad_height // 2
    padded_img = F.pad(resized_img, (pw, pad_width - pw, ph, pad_height - ph), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


class HyVLA(nn.Module):
    """Hy-VLA policy entrypoint: a flow-matching action expert wrapped
    around the Hy-Embodied (HunYuanVL-MoT) VLM.

    ``from_pretrained`` loads a directory containing ``config.json`` (parsed
    by ``HyVLAConfig``) and ``model.safetensors``.
    """

    config_class = HyVLAConfig
    name = "hy"

    def __init__(
        self,
        config: HyVLAConfig,
        *,
        inference_dtype: torch.dtype | None = None,
    ):
        """
        Args:
            config: ``HyVLAConfig`` instance describing the policy.
        """
        super().__init__()
        if not isinstance(config, HyVLAConfig):
            raise TypeError(
                f"HyVLA expects a HyVLAConfig instance; got {type(config)!r}."
            )
        self.config = config

        # Resolve the tokenizer source. Two cases:
        #   * Self-contained VLA ckpt (config has an embedded
        #     ``vlm_config_dict``): the tokenizer MUST live next to the
        #     ckpt's ``config.json`` (i.e. at ``config.pretrained_path``).
        #   * Non-self-contained (raw VLM bootstrap, ``pretrain_source`` in
        #     {``vlm``, ``scratch``}): use ``config.vlm_model_path`` (local
        #     dir or HF repo id).
        if getattr(config, "vlm_config_dict", None):
            own = getattr(config, "pretrained_path", None)
            if not own or not os.path.isdir(own):
                raise ValueError(
                    "Self-contained HyVLA ckpt is missing a usable "
                    "``pretrained_path``. ``HyVLA.from_pretrained`` should "
                    "have populated this from the ckpt directory; if you "
                    "are constructing ``HyVLA(config)`` manually, set "
                    "``config.pretrained_path`` to the ckpt directory "
                    "that ships ``tokenizer.json`` alongside its "
                    "``config.json``."
                )
            if not os.path.isfile(os.path.join(own, "tokenizer.json")):
                raise FileNotFoundError(
                    f"Self-contained HyVLA ckpt at {own!r} is missing "
                    "``tokenizer.json``. HyVLA ckpts MUST ship their "
                    "tokenizer alongside ``config.json``."
                )
            tokenizer_model_path = own
        else:
            tokenizer_model_path = config.vlm_model_path
            if not tokenizer_model_path:
                raise ValueError(
                    "Non-self-contained HyVLA construction requires "
                    "``config.vlm_model_path`` (local dir or HF repo id) "
                    "to source the tokenizer + VLM AutoConfig."
                )
        print(f"[modeling_hy_vla] Tokenizer model_path: {tokenizer_model_path}", flush=True)

        # Tokenizer is a plain ``PreTrainedTokenizerFast``. We pin
        # ``trust_remote_code=False`` so transformers does not prompt on
        # stdin when other ``auto_map`` entries are present in
        # ``config.json``.
        self.language_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_path,
            trust_remote_code=False,
            local_files_only=True,
            fix_mistral_regex=True,
        )

        self.model = HyVLAFlowMatching(
            config,
            self.language_tokenizer,
            inference_dtype=inference_dtype,
        )

        # Honor the MEM video-encoder switch. Safe to call multiple times.
        self.enable_video_encoder_if_needed()

        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        *,
        config: HyVLAConfig | None = None,
        force_download: bool = False,
        cache_dir: str | None = None,
        local_files_only: bool = True,
        revision: str | None = None,
        token: str | bool | None = None,
        map_location: str = "cpu",
        torch_dtype: torch.dtype | None = None,
        strict: bool = False,
        vlm_model_path: str | None = None,
        **kwargs,
    ) -> "HyVLA":
        """Load a ``HyVLA`` from a local directory or a HuggingFace repo id.

        The target location must contain ``config.json`` (loadable by
        ``HyVLAConfig.from_pretrained``) and ``model.safetensors``.

        Args:
            pretrained_name_or_path: Local dir or HuggingFace repo id of
                the *VLA* checkpoint (the action-flow expert).
            vlm_model_path: Optional override for the *upstream* Hy-Embodied
                VLM repo (used to source the tokenizer + AutoConfig when
                the ckpt is not self-contained). Takes precedence over the
                value recorded in the ckpt's ``config.json``; that field
                in turn defaults to ``tencent/HY-Embodied-0.5``.
        """
        import safetensors.torch as _st

        if not local_files_only or force_download:
            raise ValueError(
                "HyVLA loading is local-only; stage checkpoints with hfd under "
                "WORLDFOUNDRY_HFD_ROOT first"
            )
        del cache_dir, revision, token
        checkpoint_dir = resolve_local_hf_model_path(
            pretrained_name_or_path,
            required_files=("config.json", _SAFETENSORS_SINGLE_FILE, "tokenizer.json"),
        )

        if config is None:
            config = HyVLAConfig.from_pretrained(
                checkpoint_dir,
                local_files_only=True,
            )
        else:
            # A caller supplied a fresh HyVLAConfig. To honour the
            # self-contained contract we still inject the checkpoint's own
            # ``vlm_config_dict`` (and record ``pretrained_path``) so both
            # ``__init__`` and ``HyVLAFlowMatching.__init__`` resolve the
            # VLM AutoConfig + tokenizer purely from the ckpt directory.
            ckpt_cfg_path = None
            ckpt_dir = checkpoint_dir
            if ckpt_dir.is_dir():
                cand = ckpt_dir / "config.json"
                if cand.is_file():
                    ckpt_cfg_path = str(cand)
            if ckpt_cfg_path is None:
                raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")
            with open(ckpt_cfg_path, "r", encoding="utf-8") as _fp:
                _ckpt_cfg = json.load(_fp)
            _embedded = _ckpt_cfg.get("vlm_config_dict")
            if _embedded and not getattr(config, "vlm_config_dict", None):
                config.vlm_config_dict = _embedded
            if not getattr(config, "pretrained_path", None):
                config.pretrained_path = str(checkpoint_dir)

        # Honour an explicit VLM override by writing it into the config so
        # both ``HyVLA.__init__`` and ``HyVLAFlowMatching.__init__`` pick
        # it up.
        if vlm_model_path is not None:
            config.vlm_model_path = str(
                resolve_local_hf_model_path(
                    vlm_model_path,
                    required_files=("config.json", "tokenizer.json"),
                )
            )

        model_id = str(checkpoint_dir)
        instance = cls(config, inference_dtype=torch_dtype, **kwargs)

        model_file = os.path.join(model_id, _SAFETENSORS_SINGLE_FILE)

        _st.load_model(instance, model_file, strict=strict, device=map_location)
        instance.to(map_location)
        instance.eval()
        return instance

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def enable_video_encoder_if_needed(self) -> None:
        """Turn on the MEM space-time path in the SigLIP vision tower iff
        ``self.config.use_video_encoder`` is True.

        Safe to call multiple times: the underlying ``apply_video_encoder_patch``
        is idempotent (a second invocation sees ``use_video_encoder=True``
        on the wrapper and returns immediately), will not re-wrap
        already-wrapped blocks, nor introduce new learnable parameters.
        The method remains public because the official inference sequence
        invokes it explicitly after loading.
        """
        if not getattr(self.config, "use_video_encoder", False):
            return
        # Lazy import: ``space_time_attention`` pulls in flash_attn at import
        # time, which we do not want to require for VLA-without-video runs.
        from .space_time_attention import apply_video_encoder_patch

        stride = getattr(self.config, "spacetime_layer_stride", 4)
        past_drop_layer = getattr(self.config, "past_drop_layer", None)
        max_num_frames = getattr(self.config, "max_num_frames", 18)
        visual = self.model.dual_tower.vlm.model.visual
        apply_video_encoder_patch(
            visual,
            spacetime_layer_stride=stride,
            past_drop_layer=past_drop_layer,
            max_num_frames=max_num_frames,
        )

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)
            results = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state,
                noise=noise, vis_attn=self.config.vis_attn,
            )
            if self.config.vis_attn:
                actions, _ = results
            else:
                actions = results

            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            # ``sample_actions`` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the same inference path as :meth:`forward_evaluate`."""
        return self.forward_evaluate(batch)

    @torch.no_grad()
    def forward_evaluate(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run inference and return the predicted action chunk.

        This is the official upstream policy-inference entrypoint.
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        results = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=None, vis_attn=self.config.vis_attn,
        )
        if self.config.vis_attn:
            pred_actions, att_vis_output = results
            info = {"pred": pred_actions, "attn": att_vis_output}
        else:
            pred_actions = results
            info = {"pred": pred_actions}
        # Unpad pred to real action dims (drop trailing zero-padded channels).
        original_action_dim = self.config.action_feature.shape[0]
        info["pred"] = info["pred"][:, :, :original_action_dim]
        return info

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images: resize to 224x224 with
        aspect-ratio-preserving padding and convert pixel range from
        [0, 1] to [-1, 1] as expected by SigLIP.

        Input contract:
            ``batch[observation.images.<cam>]`` may be either
            * 4D ``(B, C, H, W)`` -- current frame only (default), or
            * 5D ``(B, K, C, H, W)`` -- K-frame history with current
              frame at ``[:, -1]`` (emitted by the MEM-aware data
              collator when history is enabled).
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        use_video = getattr(self.config, "use_video_encoder", False)

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Normalize input dim: allow both 4D (B,C,H,W) and 5D (B,K,C,H,W).
            if img.dim() == 5:
                if not use_video:
                    # Drop history; keep the current frame only: (B, 1, C, H, W).
                    img = img[:, -1:]
            elif img.dim() == 4:
                # Single-frame input: (B, C, H, W).
                if use_video:
                    # The MEM video encoder requires a temporal axis. Promote
                    # to (B, 1, C, H, W) so the downstream 5D path is
                    # well-formed (effectively a K=1 video).
                    img = img.unsqueeze(1)
            else:
                raise ValueError(
                    f"Unexpected image tensor rank {img.dim()} for key '{key}'. Expected 4D or 5D."
                )

            if self.config.resize_imgs_with_padding is not None:
                if img.dim() == 5:
                    bs, history_len, c, h, w = img.shape
                    img = img.reshape(bs * history_len, c, h, w)
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
                    if use_video:
                        # Restore the temporal axis: (B*K, C, H, W) -> (B, K, C, H, W).
                        _, c2, h2, w2 = img.shape
                        img = img.reshape(bs, history_len, c2, h2, w2)
                else:
                    bs, c, h, w = img.shape
                    img = img.reshape(bs * 1, c, h, w)
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            # Mask shape always follows batch size B (regardless of 4D/5D).
            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the inference task prompt."""
        device = next(v.device for k, v in batch.items() if k.startswith(OBS_IMAGES))
        tasks = batch["task"]

        # clean text
        tasks = [task.strip().replace("_", " ").replace("\n", " ") for task in tasks]

        # hy prompt has to end with <｜hy_Assistant｜>
        tasks = [task if task.endswith("<｜hy_Assistant｜>") else f"{task}<｜hy_Assistant｜>" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            add_special_tokens=False,
            return_token_type_ids=False,
        )

        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)
        return lang_tokens, lang_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_ROBOT], self.config.max_state_dim)
        return state

class HyVLAFlowMatching(nn.Module):
    """Hy-VLA flow-matching action expert.

    Owns the dual-tower (VLM + action expert) and flow-matching sampler.
    Wrapped by :class:`HyVLA`, which adds checkpoint loading and the
    action-chunk inference queue.

    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │action│        │
    │  ┌──────────►│expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x N   │        │
    │ │         │  └▲──▲──┘        │
    │ │   VLM   │   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config, language_tokenizer, *, inference_dtype=None):
        super().__init__()
        self.config = config
        self.language_tokenizer = language_tokenizer

        # Source the upstream VLM AutoConfig via _load_vlm_autoconfig:
        # self-contained ckpts read it from ``self.config.vlm_config_dict``
        # (no disk / network access); otherwise it is resolved from
        # ``_resolve_vlm_path``.
        vlm_inner_config = _load_vlm_autoconfig(self.config)

        # Expert config = VLM config with ``hidden_size`` overridden by
        # ``proj_width``. Released ckpt: ``hidden_size=1024`` (vs the VLM's
        # 2048) and ``intermediate_size=2048``; everything else (layers,
        # heads, vocab, rope) is shared with the VLM.
        import copy as _copy
        expert_inner_config = _copy.deepcopy(vlm_inner_config)
        expert_inner_config.hidden_size = self.config.proj_width
        expert_inner_config.intermediate_size = 2048
        if hasattr(expert_inner_config, "dense_list"):
            expert_inner_config.dense_list = [self.config.proj_width, 0]

        dual_tower_config = HyDualTowerConfig(
            vlm_config=vlm_inner_config,
            expert_config=expert_inner_config,
            attention_implementation=self.config.attention_implementation,
            config=self.config  # outer HyVLAConfig (kept for proj_width etc.)
        )
        self.dual_tower = HyDualTower(
            dual_tower_config,
            inference_dtype=inference_dtype,
        )

        # Projections are float32
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for the dual-tower transformer processing.

        Layout (per sample):
            <bos><hy_User>
            for each image:
                <vision_start>
                image_patch_grid interleaved with <vision_split> at the end of every row
                <vision_end>
            language_tokens
        """
        embs = []
        pad_masks = []
        att_masks = []
        modality_mask = []

        # Special tokens (BOS / role / vision boundaries / split / assistant)
        img = images[0]
        # add <｜hy_begin▁of▁sentence｜><｜hy_User｜>
        bos_token = torch.full((img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_begin▁of▁sentence｜>"))
        bos_token = bos_token.to(img.device)
        bos_emb = self.dual_tower.embed_language_tokens(bos_token)
        hy_user_token = torch.full((img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_User｜>"))
        hy_user_token = hy_user_token.to(img.device)
        hy_user_emb = self.dual_tower.embed_language_tokens(hy_user_token)

        # add <｜hy_place▁holder▁no▁666｜> vision_start_token
        vision_start_token = torch.full((img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_place▁holder▁no▁666｜>"))
        vision_start_token = vision_start_token.to(img.device)
        vision_start_emb = self.dual_tower.embed_language_tokens(vision_start_token)

        # add <｜hy_place▁holder▁no▁666｜> vision_end_token
        vision_end_token = torch.full((img.shape[0], 1),self.language_tokenizer.convert_tokens_to_ids("<｜hy_place▁holder▁no▁667｜>"))
        vision_end_token = vision_end_token.to(img.device)
        vision_end_emb = self.dual_tower.embed_language_tokens(vision_end_token)

        # add <｜hy_place▁holder▁no▁666｜> vision_split_token
        vision_split_token = torch.full((img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_place▁holder▁no▁671｜>"))
        vision_split_token = vision_split_token.to(img.device)
        vision_split_emb = self.dual_tower.embed_language_tokens(vision_split_token)

        # 1. Add [bos_token, hy_user_token]
        embs.extend([bos_emb, hy_user_emb])
        pad_masks.append(torch.ones((images[0].shape[0], 2), dtype=torch.bool, device=images[0].device))
        att_masks.extend([1, 1])
        modality_mask.extend([False, False])

        # Track image-token index ranges so the visual-segment attention mask
        # tweak (see ``_apply_visual_segment_mask``) can address them later.
        image_idx_ranges = []      # per-row patch ranges (excludes split tokens)
        image_full_ranges = []     # full per-image span (patches + split rows)

        # 2. Add vision_start + image patches with row-wise split tokens + vision_end
        for i, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
            bs = img.shape[0]

            # vision_start
            embs.append(vision_start_emb)
            pad_masks.append(torch.ones((bs, 1), dtype=torch.bool, device=img.device))
            att_masks.append(1)
            modality_mask.append(False)

            # embed image (bs, num_patches, emb_dim)
            img_emb = self.dual_tower.embed_image(img).to(dtype=vision_start_emb.dtype)
            num_patches, emb_dim = img_emb.shape[1], img_emb.shape[2]
            grid_size = int(num_patches ** 0.5)
            assert grid_size * grid_size == num_patches, 'num_patches must be square'

            img_emb_grid = img_emb.view(bs, grid_size, grid_size, emb_dim)
            split_expanded = vision_split_emb.unsqueeze(1).expand(bs, grid_size, 1, emb_dim)
            img_emb_with_split = torch.cat([img_emb_grid, split_expanded], dim=2)
            img_emb_with_split = img_emb_with_split.view(bs, -1, emb_dim)
            embs.append(img_emb_with_split)

            row_len = grid_size + 1
            total_img_tokens = grid_size * row_len
            start_idx = len(att_masks)

            # Per-row patch ranges (exclude the trailing split token of each row).
            row_ranges = [
                (start_idx + r * row_len, start_idx + r * row_len + grid_size)
                for r in range(grid_size)
            ]
            image_idx_ranges.extend(row_ranges)

            # Full span of this image's visual segment (patches + split tokens).
            image_full_ranges.append((start_idx, start_idx + total_img_tokens))

            att_masks.extend([1] * total_img_tokens)
            # Each grid row: ``grid_size`` patch tokens (modality=True) + 1 split token (False).
            modality_mask.extend(([True] * grid_size + [False] * 1) * grid_size)

            img_mask_expanded = img_mask[:, None].expand(bs, total_img_tokens)
            pad_masks.append(img_mask_expanded)

            # vision_end
            embs.append(vision_end_emb)
            pad_masks.append(torch.ones((bs, 1), dtype=torch.bool, device=img.device))
            att_masks.append(1)
            modality_mask.append(False)

        # 3. Language tokens
        lang_emb = self.dual_tower.embed_language_tokens(lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks.extend([1] * num_lang_embs)
        modality_mask.extend([False] * num_lang_embs)

        # 4. Stack into tensors
        bsize = images[0].shape[0]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1).to(torch.bool)

        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, -1)

        modality_mask = torch.tensor(modality_mask, dtype=torch.bool, device=pad_masks.device)
        modality_mask = modality_mask[None, :].expand(bsize, -1)

        return embs, pad_masks, att_masks, modality_mask, image_idx_ranges, image_full_ranges



    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions and timestep for the action expert.

        Emits a single absolute state token from ``state`` and the action/time
        embedding block in the expert tower's parameter dtype. The state token
        shares one attention block with the action chunk: leading
        ``att_masks=1`` followed by ``0`` for the action tokens.
        """
        embs = []
        pad_masks = []
        att_masks = []
        modality_mask = []

        # --- State token ----------------------------------------------------
        assert state is not None, "embed_suffix: ``state`` is required."
        expert_dtype = self.dual_tower.expert.model.layers[0].self_attn.q_proj.weight.dtype
        state = state.to(dtype=self.state_proj.weight.dtype)
        state_emb = self.state_proj(state).to(dtype=expert_dtype)
        # (B, D) -> (B, 1, D)
        state_block = state_emb[:, None, :]
        embs.append(state_block)

        bsize = state_block.shape[0]
        T_state = state_block.shape[1]
        device = state_block.device

        state_mask = torch.ones(bsize, T_state, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # All state tokens share one attention block: leading 1, rest 0.
        # Mirrors the action-chunk wiring further down.
        att_masks += [1] + [0] * (T_state - 1)
        modality_mask += [True] * T_state

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.to(dtype=self.action_in_proj.weight.dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(
            noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
        )

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)  # torch.float32

        action_time_emb = self.action_time_mlp_in(
            action_time_emb.to(dtype=self.action_time_mlp_in.weight.dtype)
        )
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(
            action_time_emb.to(dtype=self.action_time_mlp_out.weight.dtype)
        ).to(dtype=expert_dtype)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))
        modality_mask += [True] * (self.config.n_action_steps)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        modality_mask = torch.tensor(modality_mask, dtype=torch.bool, device=pad_masks.device)
        modality_mask = modality_mask[None, :].expand(bsize, len(modality_mask))

        return embs, pad_masks, att_masks, modality_mask

    def _apply_visual_segment_mask(
        self,
        att_2d_masks,
        image_idx_ranges,
        image_full_ranges,
    ):
        """In-place rewrite the visual-segment portion of ``att_2d_masks``.

        Two scopes are selectable via ``self.config.visual_segment_isolation``:

        * ``False`` -- *patch-only* (default, backward-compatible):
          1. collect every image's ``image_idx_ranges`` (image-patch tokens,
             excluding the per-row split tokens) and zero out their pairwise
             visibility;
          2. inside each image's ``image_full_range``, set the image-patch
             tokens to be bidirectionally visible.
          Image-patch / split-row tokens still see segment-external tokens
          via the causal mask, which differs slightly from the VLM-time
          ``_flash_attention_forward_mot`` behaviour.

        * ``True`` -- *full-segment isolation* (matches
          ``_flash_attention_forward_mot``): for each image's
          ``image_full_range`` (image patches + split / newline rows,
          excluding ``vision_start`` / ``vision_end``):
          1. clear all visibility on the rows of those tokens;
          2. enable bidirectional visibility within the segment.
          The released RoboTwin post-train ckpt was trained under this mode,
          so reproducing it requires ``visual_segment_isolation=True`` in
          ``config.json``.

        Args:
            att_2d_masks: ``(B, S, S)`` bool tensor; modified in place.
            image_idx_ranges: per-row image-patch ``[start, end)`` ranges
                (excluding split tokens).
            image_full_ranges: per-image ``[start, end)`` ranges covering
                image patches plus split / newline rows.
        """
        if getattr(self.config, "visual_segment_isolation", False):
            # Full-segment isolation: rewrite each image_full_range as a
            # self-contained bidirectional block.
            for img_full_start, img_full_end in image_full_ranges:
                full_range_idx = torch.arange(
                    img_full_start, img_full_end, device=att_2d_masks.device
                )
                # Clear outward visibility for image-patch + split rows.
                att_2d_masks[:, full_range_idx, :] = False
                # Re-enable visibility within the segment.
                att_2d_masks[:, full_range_idx[:, None], full_range_idx[None, :]] = True
            return

        # Patch-only (default): only adjust image-patch tokens; split rows
        # stay on the causal pathway.
        # Step 1: clear pairwise visibility between every image-patch token
        # (this also drops the causal-pathway visibility between them).
        all_img_indices = []
        for s, e in image_idx_ranges:
            all_img_indices.extend(range(s, e))
        if all_img_indices:
            idx = torch.tensor(all_img_indices, device=att_2d_masks.device)
            att_2d_masks[:, idx[:, None], idx[None, :]] = False

        # Step 2: re-enable bidirectional visibility among image-patch
        # tokens that belong to the same image.
        for img_full_start, img_full_end in image_full_ranges:
            img_indices = []
            for s, e in image_idx_ranges:
                if s >= img_full_start and e <= img_full_end:
                    img_indices.extend(range(s, e))
            if img_indices:
                idx = torch.tensor(img_indices, device=att_2d_masks.device)
                att_2d_masks[:, idx[:, None], idx[None, :]] = True

    # @torch.compile(mode="reduce-overhead")
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, vis_attn=False) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks, modality_mask_prefix, image_idx_ranges, image_full_ranges = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Adjust visual-segment attention according to the configured scope.
        self._apply_visual_segment_mask(prefix_att_2d_masks, image_idx_ranges, image_full_ranges)

        # Compute image and language key value cache
        (prefix_out, _), past_key_values, _, _ = self.dual_tower.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            modality_masks=[modality_mask_prefix, None]
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t, att_vis_output = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt

        if vis_attn:
            # Strip non-patch tokens from att_vis_output, leaving the
            # contiguous (B, H, suffix_len, num_patches * num_views)
            # tensor that downstream visualisation tooling expects.
            all_img_indices = []
            for s, e in image_idx_ranges:
                all_img_indices.extend(range(s, e))
            img_idx_tensor = torch.tensor(all_img_indices, dtype=torch.long, device=device)

            cleaned_att = []
            for layer_att in att_vis_output:
                cleaned_att.append(layer_att[:, :, :, img_idx_tensor])
            return x_t, cleaned_att

        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # IMPORTANT: copy the past_key_values, or its size will increase during n-step denoise.
        past_key_values_vlm = copy.deepcopy(past_key_values)

        suffix_embs, suffix_pad_masks, suffix_att_masks, modality_mask_suffix = self.embed_suffix(
            state, x_t, timestep,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _, att_vis_output, _ = self.dual_tower.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values_vlm,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            modality_masks=[None, modality_mask_suffix]
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(
            suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        )
        return v_t, att_vis_output


__all__ = ["HyVLAConfig", "HyVLA", "HyVLAFlowMatching"]
