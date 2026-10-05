#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
``DexoraPolicy`` — thin runtime wrapper around the Dexora policy
(``models.rdt_runner.RDTRunner``) that exposes the same ``get_action(obs_dict)``
interface used by the real-robot inference loop in
``deploy/dexora_inference_zmq.py``.

The wrapper takes care of:

* Loading policy weights from a local checkpoint file or directory.
* Loading the SigLIP-SO400M vision encoder and the T5-v1.1-XXL text encoder.
* Mapping the 4 raw camera images to per-camera SigLIP token sequences.
* Encoding the (single) language instruction with T5 to ``[lang_len, 4096]``
  token embeddings + attention mask.
* Concatenating the proprioceptive 36-D state into the ``state_tokens``
  expected by ``RDTRunner.predict_action``.

The output is a numpy array of shape ``[chunk_size, 36]`` in the canonical
order ``[left_arm(6) | right_arm(6) | left_hand(12) | right_hand(12)]``,
all in radians — exactly the layout consumed by ``mmk_forwarder`` (arms,
first 12 dims) and ``xhand_forwarder`` (hands, last 24 dims).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

from worldfoundry.core.execution.device import resolve_inference_dtype

from .runner import RDTRunner
from .siglip import SiglipVisionTower
from .t5 import T5Embedder


@dataclass
class DexoraPolicyConfig:
    """Resolved local paths and execution settings for ``DexoraPolicy``."""

    model_config_path: str
    text_encoder_path: str
    vision_encoder_path: str
    dtype: torch.dtype | str
    device: str
    local_files_only: bool
    state_dim: int = field(init=False)
    chunk_size: int = field(init=False)
    cameras: Sequence[str] = field(init=False)
    tokenizer_max_length: int = field(init=False)


class DexoraPolicy:
    """Thin wrapper around ``RDTRunner`` for real-robot inference."""

    def __init__(
        self,
        model_path: str,
        cfg: Optional[DexoraPolicyConfig] = None,
    ) -> None:
        if cfg is None:
            raise ValueError("DexoraPolicy requires resolved data-backed configuration")
        self.cfg = cfg
        self.device = torch.device(self.cfg.device)
        self.cfg.dtype = resolve_inference_dtype(self.device, self.cfg.dtype)

        # ---- Load YAML config the policy was trained with ------------------
        with open(self.cfg.model_config_path, "r") as f:
            self.model_yaml = yaml.safe_load(f)

        common = self.model_yaml["common"]
        self.cfg.state_dim = int(common["state_dim"])
        self.cfg.chunk_size = int(common["action_chunk_size"])
        self.cfg.cameras = tuple(str(value) for value in common["cameras"])
        self.cfg.tokenizer_max_length = int(common["tokenizer_max_length"])

        # ---- Load encoders -------------------------------------------------
        logging.info(f"Loading T5 text encoder from {self.cfg.text_encoder_path} ...")
        tokenizer_max_length = self.cfg.tokenizer_max_length
        text_embedder = T5Embedder(
            from_pretrained=self.cfg.text_encoder_path,
            model_max_length=tokenizer_max_length,
            device=self.device,
            torch_dtype=self.cfg.dtype,
            local_files_only=self.cfg.local_files_only,
        )
        self.tokenizer = text_embedder.tokenizer
        self.text_encoder = text_embedder.model.to(self.device, dtype=self.cfg.dtype).eval()

        logging.info(f"Loading SigLIP vision encoder from {self.cfg.vision_encoder_path} ...")
        self.vision_encoder = SiglipVisionTower(
            vision_tower=self.cfg.vision_encoder_path,
            args=None,
            local_files_only=self.cfg.local_files_only,
        )
        self.vision_encoder.vision_tower.to(self.device, dtype=self.cfg.dtype).eval()
        self.image_processor = self.vision_encoder.image_processor

        # ---- Load policy ---------------------------------------------------
        logging.info(f"Loading Dexora policy from {model_path} ...")
        self.policy = self._load_policy(model_path).to(self.device, dtype=self.cfg.dtype).eval()
        n_params = sum(p.numel() for p in self.policy.parameters())
        logging.info(f"[DexoraPolicy] policy params = {n_params / 1e6:.1f}M")

        # Static action mask (all dims active for the 36-DoF embodiment).
        self._action_mask = torch.ones((1, 1, self.cfg.state_dim), device=self.device, dtype=self.cfg.dtype)

        # Cache last language embedding to avoid re-running T5 every step.
        self._cached_instruction: Optional[str] = None
        self._cached_lang_tokens: Optional[torch.Tensor] = None
        self._cached_lang_mask: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def get_action(self, obs: dict) -> np.ndarray:
        """
        Run one diffusion sampling pass and return ``[chunk_size, state_dim]``.

        ``obs`` must contain:
          * ``state``        : np.ndarray ``[state_dim]`` (radians)
          * ``images``       : dict ``{cam_name -> np.ndarray [H, W, 3] RGB uint8}``
                               with keys matching ``cfg.cameras``
          * ``instruction``  : str  (single English language goal)
          * ``ctrl_freq``    : float, optional (defaults to config['common'] value)
        """
        # ---- 1. Language ---------------------------------------------------
        lang_tokens, lang_mask = self._encode_language(obs["instruction"])

        # ---- 2. Vision -----------------------------------------------------
        img_tokens = self._encode_images(obs["images"])

        # ---- 3. State + control frequency ---------------------------------
        state = torch.from_numpy(np.asarray(obs["state"], dtype=np.float32))[None, None, :]
        # Pad / truncate to expected state_dim (defensive against 39-D feeders).
        if state.shape[-1] > self.cfg.state_dim:
            state = state[..., : self.cfg.state_dim]
        elif state.shape[-1] < self.cfg.state_dim:
            pad = torch.zeros(1, 1, self.cfg.state_dim - state.shape[-1])
            state = torch.cat([state, pad], dim=-1)
        state = state.to(self.device, dtype=self.cfg.dtype)

        if "ctrl_freq" not in obs:
            raise ValueError("Dexora ctrl_freq must be supplied by the data-backed runtime configuration")
        ctrl_freq = float(obs["ctrl_freq"])
        ctrl_freqs = torch.tensor([ctrl_freq], device=self.device, dtype=self.cfg.dtype)

        # ---- 4. Diffusion sampling ----------------------------------------
        action_pred = self.policy.predict_action(
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_mask,
            img_tokens=img_tokens,
            state_tokens=state,
            action_mask=self._action_mask,
            ctrl_freqs=ctrl_freqs,
        )  # [1, chunk_size, state_dim]

        return action_pred[0].float().cpu().numpy()

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    def _load_policy(self, model_path: str) -> RDTRunner:
        """Construct from data-backed architecture and restore a strict local state dict."""
        cfg = self.model_yaml
        img_cond_len = (
            cfg["common"]["img_history_size"] * cfg["common"]["num_cameras"] * self.vision_encoder.num_patches
        )
        with torch.device("meta"):
            policy = RDTRunner(
                action_dim=cfg["common"]["state_dim"],
                pred_horizon=cfg["common"]["action_chunk_size"],
                config=cfg["model"],
                lang_token_dim=cfg["model"]["lang_token_dim"],
                img_token_dim=cfg["model"]["img_token_dim"],
                state_token_dim=cfg["model"]["state_token_dim"],
                max_lang_cond_len=self.cfg.tokenizer_max_length,
                img_cond_len=img_cond_len,
                img_pos_embed_config=[
                    (
                        "image",
                        (
                            cfg["common"]["img_history_size"],
                            cfg["common"]["num_cameras"],
                            -self.vision_encoder.num_patches,
                        ),
                    ),
                ],
                lang_pos_embed_config=[
                    ("lang", -self.cfg.tokenizer_max_length),
                ],
                dtype=self.cfg.dtype,
            )

        root = Path(model_path).expanduser()
        if root.is_file():
            state_path = root
        elif root.is_dir():
            candidates = (root / "model.safetensors", root / "pytorch_model.bin")
            state_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if state_path is None:
                names = ", ".join(candidate.name for candidate in candidates)
                raise FileNotFoundError(f"Dexora checkpoint directory {root} contains neither {names}")
        else:
            raise FileNotFoundError(f"Local Dexora checkpoint does not exist: {root}")

        if state_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(state_path), device="cpu")
        else:
            state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"Dexora checkpoint must contain a state mapping: {state_path}")
        for container_key in ("module", "model_state_dict", "state_dict", "model"):
            nested = state.get(container_key)
            if isinstance(nested, dict):
                state = nested
                break
        if not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
        ):
            raise TypeError(f"Dexora checkpoint must contain a flat tensor state dict: {state_path}")
        if all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        policy.load_state_dict(state, strict=True, assign=True)
        return policy

    def _encode_language(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        if instruction == self._cached_instruction and self._cached_lang_tokens is not None:
            return self._cached_lang_tokens, self._cached_lang_mask

        max_len = self.cfg.tokenizer_max_length
        tokens = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attn_mask = tokens["attention_mask"].to(self.device)
        lang_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)["last_hidden_state"].to(
            self.cfg.dtype
        )

        # Cache one entry (most deployments use a single instruction for a run).
        self._cached_instruction = instruction
        self._cached_lang_tokens = lang_embeds
        self._cached_lang_mask = attn_mask.to(torch.bool)
        return self._cached_lang_tokens, self._cached_lang_mask

    def _encode_images(self, images: dict) -> torch.Tensor:
        """Resize / pad / normalize each of the 4 cameras through SigLIP."""
        # Background colour for any missing camera.
        bg_color = np.array([int(x * 255) for x in self.image_processor.image_mean], dtype=np.uint8).reshape(1, 1, 3)

        pixel_values = []
        for cam in self.cfg.cameras:
            img = images.get(cam)
            if img is None:
                # Use the SigLIP mean-colour background as the "missing camera"
                # placeholder used by the released preprocessing contract.
                H = self.image_processor.size["height"]
                W = self.image_processor.size["width"]
                img = np.ones((H, W, 3), dtype=np.uint8) * bg_color
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            pil = Image.fromarray(img, mode="RGB")
            # Pad-to-square (image_aspect_ratio == 'pad') so wrist views aren't squashed.
            pil = _expand2square(pil, tuple(int(x * 255) for x in self.image_processor.image_mean))
            arr = self.image_processor.preprocess(pil, return_tensors="pt")["pixel_values"][0]
            pixel_values.append(arr)
        batch = torch.stack(pixel_values, dim=0).to(self.device, dtype=self.cfg.dtype)
        # [N_cam, T_patch, hidden]
        img_embeds = self.vision_encoder(batch).detach()
        # Flatten to [1, N_cam * T_patch, hidden]
        return img_embeds.reshape(1, -1, self.vision_encoder.hidden_size)


def _expand2square(pil_img: Image.Image, bg_color):
    """Square-pad an image to its longest side."""
    w, h = pil_img.size
    if w == h:
        return pil_img
    if w > h:
        out = Image.new(pil_img.mode, (w, w), bg_color)
        out.paste(pil_img, (0, (w - h) // 2))
        return out
    out = Image.new(pil_img.mode, (h, h), bg_color)
    out.paste(pil_img, ((h - w) // 2, 0))
    return out
