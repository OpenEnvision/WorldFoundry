# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""WorldCrafter camera/memory extensions to the shared Helios base model.

Adapted from TencentARC/WorldCrafter at 7e57e4d; see THIRD-PARTY-NOTICES (WorldCrafter).
"""
import json
import os
import torch
from diffusers.utils import logging
from worldfoundry.base_models.diffusion_model.models.networks.helios.model import (
    HeliosTransformer3DModel, HeliosTransformerBlock, _context_parallel_runtime,
)
logger = logging.get_logger(__name__)

class WorldCrafterTransformerBlock(HeliosTransformerBlock):
    def _self_attention_residual(
        self, hidden_states, norm_hidden_states, gate_msa, rotary_emb,
        original_context_length, camera_control_ucpe_input=None,
    ):
        history_seq_len = (
            hidden_states.shape[1] - original_context_length
            if original_context_length is not None
            else 0
        )
        current_norm_hidden_states = norm_hidden_states[:, history_seq_len:, :]

        if hasattr(self, "cam_self_attn") and camera_control_ucpe_input is not None:
            if self.cam_self_attn.adaptation_method == "before":
                cam_current = self.cam_self_attn(
                    current_norm_hidden_states, camera_control_ucpe_input
                )
                cam_full = torch.zeros_like(norm_hidden_states)
                cam_full[:, history_seq_len:, :] = cam_current
                norm_hidden_states = norm_hidden_states + cam_full

        attn_output = self.attn1(
            norm_hidden_states, None, None, rotary_emb, original_context_length
        )

        if hasattr(self, "cam_self_attn") and camera_control_ucpe_input is not None:
            if self.cam_self_attn.adaptation_method == "parallel":
                cam_current = self.cam_self_attn(
                    current_norm_hidden_states, camera_control_ucpe_input
                )
                cam_full = torch.zeros_like(attn_output)
                cam_full[:, history_seq_len:, :] = cam_current
                attn_output = attn_output + cam_full

        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        if hasattr(self, "cam_self_attn") and camera_control_ucpe_input is not None:
            if self.cam_self_attn.adaptation_method == "after":
                cam_current = self.cam_self_attn(
                    hidden_states[:, history_seq_len:, :], camera_control_ucpe_input
                )
                hidden_states = hidden_states.clone()
                hidden_states[:, history_seq_len:, :] = (
                    hidden_states[:, history_seq_len:, :] + cam_current
                )

        return hidden_states

class WorldCrafterTransformer3DModel(HeliosTransformer3DModel):
    _block_class = WorldCrafterTransformerBlock
    _history_patch_name = "patch_memory"
    _history_patch_scale = 1
    _no_split_modules = ["WorldCrafterTransformerBlock", "HeliosOutputNorm"]
    _repeated_blocks = ["WorldCrafterTransformerBlock"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q", r"patch_long\..*"]
    _skip_layerwise_casting_patterns = ["patch_embedding", "patch_short", "patch_mid", "patch_memory", "condition_embedder", "norm"]

    def _block_forward_kwargs(self, attention_kwargs):
        return {"camera_control_ucpe_input": (attention_kwargs or {}).get("camera_control_ucpe_input")}

    def _long_history_tokens(self, latents, indices, height, width):
        latents = self.patch_memory(latents)
        _, _, _, height, width = latents.shape
        rotary = self.rope(frame_indices=indices, height=height, width=width, device=latents.device)
        return latents.flatten(2).transpose(1, 2), rotary.flatten(2).transpose(1, 2)

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                indices_hidden_states=None, indices_latents_history_short=None,
                indices_latents_history_mid=None, indices_latents_memory=None,
                latents_history_short=None, latents_history_mid=None, latents_memory=None,
                indices_latents_history_long=None, latents_history_long=None,
                return_dict=True, attention_kwargs=None):
        if _context_parallel_runtime(self) is not None:
            raise ValueError("WorldCrafter camera attention requires single-device inference")
        return super().forward(
            hidden_states, timestep, encoder_hidden_states,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_memory if indices_latents_memory is not None else indices_latents_history_long,
            latents_history_short=latents_history_short, latents_history_mid=latents_history_mid,
            latents_history_long=latents_memory if latents_memory is not None else latents_history_long,
            return_dict=return_dict, attention_kwargs=attention_kwargs,
        )

    @staticmethod
    def _local_checkpoint_keys(checkpoint_dir: str) -> set[str] | None:
        index_names = (
            "diffusion_pytorch_model.safetensors.index.json",
            "model.safetensors.index.json",
            "diffusion_pytorch_model.bin.index.json",
            "pytorch_model.bin.index.json",
        )
        for index_name in index_names:
            index_path = os.path.join(checkpoint_dir, index_name)
            if os.path.exists(index_path):
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
                return set(weight_map.keys())

        for weights_name in (
            "diffusion_pytorch_model.safetensors",
            "model.safetensors",
        ):
            weights_path = os.path.join(checkpoint_dir, weights_name)
            if os.path.exists(weights_path):
                from safetensors import safe_open

                with safe_open(weights_path, framework="pt", device="cpu") as f:
                    return set(f.keys())

        return None


    @staticmethod
    def _resolve_local_checkpoint_dir(
        pretrained_model_name_or_path: str, subfolder: str | None
    ) -> str | None:
        if not isinstance(pretrained_model_name_or_path, (str, os.PathLike)):
            return None
        if not os.path.isdir(pretrained_model_name_or_path):
            return None
        checkpoint_dir = pretrained_model_name_or_path
        if subfolder is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, subfolder)
        if not os.path.isdir(checkpoint_dir):
            return None
        return checkpoint_dir


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        subfolder = kwargs.get("subfolder")
        checkpoint_dir = cls._resolve_local_checkpoint_dir(
            pretrained_model_name_or_path, subfolder
        )
        checkpoint_keys = (
            cls._local_checkpoint_keys(checkpoint_dir)
            if checkpoint_dir is not None
            else None
        )
        init_memory_from_short = (
            checkpoint_keys is not None
            and "patch_memory.weight" not in checkpoint_keys
            and "patch_short.weight" in checkpoint_keys
        )

        loaded = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model = loaded[0] if isinstance(loaded, tuple) else loaded

        if (
            init_memory_from_short
            and hasattr(model, "patch_memory")
            and hasattr(model, "patch_short")
        ):
            with torch.no_grad():
                model.patch_memory.weight.copy_(model.patch_short.weight)
                if (
                    model.patch_memory.bias is not None
                    and model.patch_short.bias is not None
                ):
                    model.patch_memory.bias.copy_(model.patch_short.bias)
            logger.info("Initialized patch_memory weights from patch_short.")

        return loaded
