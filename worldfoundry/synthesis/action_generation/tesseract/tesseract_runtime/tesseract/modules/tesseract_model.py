# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
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

from typing import Any, Dict, Optional, Tuple, Union
import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from diffusers.models.attention import Attention, AdaLayerNorm, FeedForward
from diffusers import CogVideoXTransformer3DModel
from diffusers.utils import is_torch_version
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers

from .embeddings import TesserActDepthPatchEmbed, TesserActDepthNormalPatchEmbed


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation_fn: str = "relu",
        use_batchnorm: bool = False,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        activation_functions = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
        }
        if activation_fn not in activation_functions:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        self.activation_fn = activation_functions[activation_fn]()

        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation_fn)
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))
            prev_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Conv3DNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        paddings: list[int],
        activation_fn: str = "relu",
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        activation_functions = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
        }

        if activation_fn not in activation_functions:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        self.activation_fn = activation_functions[activation_fn]()

        layers = []
        prev_channels = input_channels

        # Add Conv3d hidden layers
        for i, hidden_dim in enumerate(hidden_channels):
            layers.append(
                nn.Conv3d(
                    in_channels=prev_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                )
            )
            layers.append(self.activation_fn)
            if dropout_prob > 0.0:
                layers.append(nn.Dropout3d(dropout_prob))
            prev_channels = hidden_dim

        # Add final Conv3d output layer
        layers.append(
            nn.Conv3d(
                in_channels=prev_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MapBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        inner_dim = dim
        self.map_norm = nn.LayerNorm(inner_dim)
        self.map_cross = Attention(
            query_dim=inner_dim + 128,
            cross_attention_dim=inner_dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm",
            eps=1e-6,
            bias=True,
            out_bias=True,
            out_dim=inner_dim,
        )
        self.output_norm = nn.LayerNorm(inner_dim)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(self, map_feature, hidden_states, camera_feature):
        map_feature = self.map_norm(map_feature)
        image_feature = torch.cat([hidden_states, camera_feature], dim=-1)
        hidden_states = self.map_cross(image_feature, map_feature) + hidden_states
        norm_hidden_states = self.output_norm(hidden_states)
        hidden_states = self.ff(norm_hidden_states) + hidden_states
        return hidden_states


class TesserActDepth(CogVideoXTransformer3DModel):
    """
    This class is in development. Please use TesserActDepthNormal instead.
    """

    @classmethod
    def from_pretrained_modify(
        self,
        pretrained_model_path,
        subfolder=None,
        transformer_additional_kwargs={},
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
        **kwargs,
    ):
        # from https://github.com/aigc-apps/CogVideoX-Fun/blob/main/cogvideox/models/transformer3d.py#L657
        if not os.path.isdir(pretrained_model_path):
            pretrained_model_path = snapshot_download(
                repo_id=pretrained_model_path,
                revision=kwargs.pop("revision", None),
                cache_dir=kwargs.pop("cache_dir", None),
                token=kwargs.pop("token", None),
                local_dir_use_symlinks=False,
            )
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, "config.json")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME

        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        model = self.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            import glob

            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        m, u = model.load_state_dict(state_dict, strict=False)
        model = model.to(torch_dtype)
        return model


class TesserActDepthNormal(TesserActDepth):
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
    ):
        super().__init__(
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            flip_sin_to_cos,
            freq_shift,
            time_embed_dim,
            ofs_embed_dim,
            text_embed_dim,
            num_layers,
            dropout,
            attention_bias,
            sample_width,
            sample_height,
            sample_frames,
            patch_size,
            patch_size_t,
            temporal_compression_ratio,
            max_text_seq_length,
            activation_fn,
            timestep_activation_fn,
            norm_elementwise_affine,
            norm_eps,
            spatial_interpolation_scale,
            temporal_interpolation_scale,
            use_rotary_positional_embeddings,
            use_learned_positional_embeddings,
            patch_bias,
        )
        # Override the class name for PEFT compatibility
        self.__class__.__name__ = "CogVideoXTransformer3DModel"
        
        inner_dim = num_attention_heads * attention_head_dim

        # Redefine Patch embedding
        self.patch_embed = TesserActDepthNormalPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )

        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.dn_out_proj = Conv3DNet(
            input_channels=output_dim // 4 * 5,
            output_channels=inner_dim // 4,
            hidden_channels=[inner_dim // 4] * 2,
            kernel_sizes=[5] * 2,
            strides=[1] * 2,
            paddings=[2] * 2,
            activation_fn="relu",
            dropout_prob=dropout,
        )
        self.dn_out = MLP(
            input_dim=inner_dim * 2,
            output_dim=output_dim * 2,
            hidden_dims=[inner_dim * 2],
            activation_fn="relu",
            use_batchnorm=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        depth_input_state = hidden_states[:, :, channels // 3 : channels // 3 * 2]
        normal_input_state = hidden_states[:, :, channels // 3 * 2 :]
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        hidden_states = self.norm_out(hidden_states, temb=emb)
        rgb_states = self.proj_out(hidden_states)

        # Unpatchify RGB states
        if p_t is None:
            rgb_output = rgb_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            rgb_output = rgb_output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            rgb_output = rgb_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            rgb_output = rgb_output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        # rgb_output [b, t, c, h, w], depth_input_state [b, t, c * 2, h, w]
        rdn_input_state = torch.cat([rgb_output, depth_input_state, normal_input_state], dim=2).transpose(1, 2)
        rdn_input_state = self.dn_out_proj(rdn_input_state).transpose(1, 2)
        rdn_input_state = rdn_input_state.reshape(batch_size, num_frames, -1, height // p, p, width // p, p)
        rdn_input_state = rdn_input_state.permute(0, 1, 3, 5, 2, 4, 6)  # [b, t, h/p, w/p, c, p, p]
        rdn_input_state = rdn_input_state.flatten(4, 6).flatten(1, 3)  # [b, (t * h/p * w/p), c * p * p]
        depth_states = self.dn_out(torch.cat([hidden_states, rdn_input_state], dim=-1))

        # Unpatchify depth states
        if p_t is None:
            dn_output = depth_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            dn_output = dn_output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            dn_output = depth_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            dn_output = dn_output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        # 5. Output
        output = torch.cat([rgb_output, dn_output], dim=2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
