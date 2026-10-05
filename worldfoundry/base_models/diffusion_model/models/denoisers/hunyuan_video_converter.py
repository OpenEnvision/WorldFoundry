"""Civitai / official-key remapping for the original HunyuanVideo DiT.

The released HunyuanVideo transformer uses ``double_blocks`` /
``single_blocks`` / ``txt_in.individual_token_refiner`` names.  The
native module in :mod:`..networks.hunyuan_video` uses ``component_a`` /
``component_b`` dual-stream names and splits fused ``linear1`` / ``linear2``
weights.  :class:`HunyuanVideoDiTStateDictConverter.from_civitai` is the
``state_dict_converter`` passed to :class:`~...loaders.module.NativeModuleLoader`
by :mod:`.hunyuan_video`; it does not implement :class:`~...contracts.Denoiser`.
"""

from __future__ import annotations

import torch


class HunyuanVideoDiTStateDictConverter:
    """Map official HunyuanVideo DiT keys onto the native module tree."""

    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        """Rewrite one checkpoint dict; unwraps a top-level ``module`` wrapper."""
        if "module" in state_dict:
            state_dict = state_dict["module"]
        direct_dict = {
            "img_in.proj": "img_in.proj",
            "time_in.mlp.0": "time_in.timestep_embedder.0",
            "time_in.mlp.2": "time_in.timestep_embedder.2",
            "vector_in.in_layer": "vector_in.0",
            "vector_in.out_layer": "vector_in.2",
            "guidance_in.mlp.0": "guidance_in.timestep_embedder.0",
            "guidance_in.mlp.2": "guidance_in.timestep_embedder.2",
            "txt_in.input_embedder": "txt_in.input_embedder",
            "txt_in.t_embedder.mlp.0": "txt_in.t_embedder.timestep_embedder.0",
            "txt_in.t_embedder.mlp.2": "txt_in.t_embedder.timestep_embedder.2",
            "txt_in.c_embedder.linear_1": "txt_in.c_embedder.0",
            "txt_in.c_embedder.linear_2": "txt_in.c_embedder.2",
            "final_layer.linear": "final_layer.linear",
            "final_layer.adaLN_modulation.1": "final_layer.adaLN_modulation.1",
        }
        txt_suffix_dict = {
            "norm1": "norm1",
            "self_attn_qkv": "self_attn_qkv",
            "self_attn_proj": "self_attn_proj",
            "norm2": "norm2",
            "mlp.fc1": "mlp.0",
            "mlp.fc2": "mlp.2",
            "adaLN_modulation.1": "adaLN_modulation.1",
        }
        double_suffix_dict = {
            "img_mod.linear": "component_a.mod.linear",
            "img_attn_qkv": "component_a.to_qkv",
            "img_attn_q_norm": "component_a.norm_q",
            "img_attn_k_norm": "component_a.norm_k",
            "img_attn_proj": "component_a.to_out",
            "img_mlp.fc1": "component_a.ff.0",
            "img_mlp.fc2": "component_a.ff.2",
            "txt_mod.linear": "component_b.mod.linear",
            "txt_attn_qkv": "component_b.to_qkv",
            "txt_attn_q_norm": "component_b.norm_q",
            "txt_attn_k_norm": "component_b.norm_k",
            "txt_attn_proj": "component_b.to_out",
            "txt_mlp.fc1": "component_b.ff.0",
            "txt_mlp.fc2": "component_b.ff.2",
        }
        single_suffix_dict = {
            "linear1": ["to_qkv", "ff.0"],
            "linear2": ["to_out", "ff.2"],
            "q_norm": "norm_q",
            "k_norm": "norm_k",
            "modulation.linear": "mod.linear",
        }
        # single_suffix_dict = {
        #     "linear1": "linear1",
        #     "linear2": "linear2",
        #     "q_norm": "q_norm",
        #     "k_norm": "k_norm",
        #     "modulation.linear": "modulation.linear",
        # }
        state_dict_ = {}
        for name, param in state_dict.items():
            names = name.split(".")
            direct_name = ".".join(names[:-1])
            if direct_name in direct_dict:
                name_ = direct_dict[direct_name] + "." + names[-1]
                state_dict_[name_] = param
            elif names[0] == "double_blocks":
                prefix = ".".join(names[:2])
                suffix = ".".join(names[2:-1])
                name_ = prefix + "." + double_suffix_dict[suffix] + "." + names[-1]
                state_dict_[name_] = param
            elif names[0] == "single_blocks":
                prefix = ".".join(names[:2])
                suffix = ".".join(names[2:-1])
                if isinstance(single_suffix_dict[suffix], list):
                    if suffix == "linear1":
                        name_a, name_b = single_suffix_dict[suffix]
                        param_a, param_b = torch.split(param, (3072*3, 3072*4), dim=0)
                        state_dict_[prefix + "." + name_a + "." + names[-1]] = param_a
                        state_dict_[prefix + "." + name_b + "." + names[-1]] = param_b
                    elif suffix == "linear2":
                        if names[-1] == "weight":
                            name_a, name_b = single_suffix_dict[suffix]
                            param_a, param_b = torch.split(param, (3072*1, 3072*4), dim=-1)
                            state_dict_[prefix + "." + name_a + "." + names[-1]] = param_a
                            state_dict_[prefix + "." + name_b + "." + names[-1]] = param_b
                        else:
                            name_a, name_b = single_suffix_dict[suffix]
                            state_dict_[prefix + "." + name_a + "." + names[-1]] = param
                    else:
                        pass
                else:
                    name_ = prefix + "." + single_suffix_dict[suffix] + "." + names[-1]
                    state_dict_[name_] = param
            elif names[0] == "txt_in":
                prefix = ".".join(names[:4]).replace(".individual_token_refiner.", ".")
                suffix = ".".join(names[4:-1])
                name_ = prefix + "." + txt_suffix_dict[suffix] + "." + names[-1]
                state_dict_[name_] = param
            else:
                pass

        return state_dict_

__all__ = ["HunyuanVideoDiTStateDictConverter"]

