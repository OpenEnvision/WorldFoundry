# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""Inference-only HY15 Action2V PRoPE transformer; common layers are shared."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from einops import rearrange, repeat

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.minwm.prope import prope_qkv
from worldfoundry.core.attention import apply_nd_rotary_embedding as apply_rotary_emb
from worldfoundry.core.attention import get_nd_rotary_pos_embed
from worldfoundry.core.attention.sequence.multimodal_sequence import parallel_attention
from worldfoundry.core.distributed.model_parallel.sequence_mesh_state import get_parallel_state
from worldfoundry.core.distributed.sequence_parallel.communication_op import sequence_model_parallel_all_gather
from worldfoundry.core.nn import ConditioningProjection as TextProjection
from worldfoundry.core.nn import DiTModulation as ModulateDiT
from worldfoundry.core.nn import MLPEmbedder
from worldfoundry.core.nn import SinusoidalTimestepEmbedder as TimestepEmbedder
from worldfoundry.core.nn import TransformerMLP as MLP
from worldfoundry.core.nn import activation_layer as get_activation_layer
from worldfoundry.core.nn import normalization_layer as get_norm_layer

from .camera_attention import sequence_parallel_attention_txt, sequence_parallel_attention_vision
from .camera_modulation import FinalLayer, apply_gate, modulate
from .modules.embed_layers import ByT5Mapper, PatchEmbed, VisionProjection
from .modules.token_refiner import SingleTokenRefiner


class MMDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.deterministic = False
        self.heads_num = heads_num
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.qkv_bias = qkv_bias
        self.factory_kwargs = factory_kwargs
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )
        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def modulate_txt(self, vec_txt, txt):
        (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.txt_mod(
            vec_txt
        ).chunk(6, dim=-1)
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)
        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)
        return (txt_q, txt_k, txt_v, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate)

    def modulate_img(self, vec, img):
        (img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.img_mod(
            vec
        ).chunk(6, dim=-1)
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)
        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        return (img_q, img_k, img_v, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate)

    def forward_txt(
        self,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
        kv_cache: Optional[dict] = None,
        cache_txt: bool = False,
    ) -> Tuple[torch.Tensor]:
        (txt_q, txt_k, txt_v, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.modulate_txt(
            vec_txt, txt
        )
        attn_mode = "torch_causal"
        (txt_attn, t_kv) = sequence_parallel_attention_txt(
            txt_q,
            txt_k,
            txt_v,
            img_q_len=txt_q.shape[1],
            img_kv_len=txt_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
            kv_cache=kv_cache,
            cache_txt=cache_txt,
        )
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)), gate=txt_mod2_gate
        )
        return (txt, t_kv)

    def forward_vision(
        self,
        img: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        attn_param=None,
        block_idx=None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_vision: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (img_q, img_k, img_v, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.modulate_img(
            vec, img
        )
        (img_q_prope, img_k_prope, img_v_prope, apply_fn_o) = prope_qkv(
            img_q.permute(0, 2, 1, 3), img_k.permute(0, 2, 1, 3), img_v.permute(0, 2, 1, 3), viewmats=viewmats, Ks=Ks
        )
        img_q_prope = img_q_prope.permute(0, 2, 1, 3)
        img_k_prope = img_k_prope.permute(0, 2, 1, 3)
        img_v_prope = img_v_prope.permute(0, 2, 1, 3)
        if freqs_cis is not None:
            (img_qq, img_kk) = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
                f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            )
            (img_q, img_k) = (img_qq, img_kk)
        (img_attn, vision_kv) = sequence_parallel_attention_vision(
            img_q, img_k, img_v, block_idx=block_idx, kv_cache=kv_cache, cache_vision=cache_vision
        )
        (img_attn_prope, _) = sequence_parallel_attention_vision(
            img_q_prope, img_k_prope, img_v_prope, block_idx=block_idx, kv_cache=kv_cache, cache_vision=False
        )
        img_attn_prope = rearrange(img_attn_prope, "B L (H D) -> B H L D", H=self.heads_num)
        img_attn_prope = apply_fn_o(img_attn_prope)
        img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")
        img = img + apply_gate(
            self.img_attn_proj(img_attn) + self.img_attn_prope_proj(img_attn_prope), gate=img_mod1_gate
        )
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)), gate=img_mod2_gate
        )
        return (img, vision_kv)

    def forward_bi(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (txt_q, txt_k, txt_v, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.modulate_txt(
            vec_txt, txt
        )
        (img_q, img_k, img_v, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.modulate_img(
            vec, img
        )
        (img_q_prope, img_k_prope, img_v_prope, apply_fn_o) = prope_qkv(
            img_q.permute(0, 2, 1, 3), img_k.permute(0, 2, 1, 3), img_v.permute(0, 2, 1, 3), viewmats=viewmats, Ks=Ks
        )
        img_q_prope = img_q_prope.permute(0, 2, 1, 3)
        img_k_prope = img_k_prope.permute(0, 2, 1, 3)
        img_v_prope = img_v_prope.permute(0, 2, 1, 3)
        if freqs_cis is not None:
            (img_qq, img_kk) = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
                f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            )
            (img_q, img_k) = (img_qq, img_kk)
        attn_mode = "flash" if is_flash else self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        (img_attn, txt_attn) = (attn[:, : img_q.shape[1]].contiguous(), attn[:, img_q.shape[1] :].contiguous())
        attn_prope = parallel_attention(
            (img_q_prope, txt_q),
            (img_k_prope, txt_k),
            (img_v_prope, txt_v),
            img_q_len=img_q_prope.shape[1],
            img_kv_len=img_k_prope.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        img_attn_prope = attn_prope[:, : img_q_prope.shape[1]].contiguous()
        img_attn_prope = rearrange(img_attn_prope, "B L (H D) -> B H L D", H=self.heads_num)
        img_attn_prope = apply_fn_o(img_attn_prope)
        img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")
        img = img + apply_gate(
            self.img_attn_proj(img_attn) + self.img_attn_prope_proj(img_attn_prope), gate=img_mod1_gate
        )
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)), gate=img_mod2_gate
        )
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)), gate=txt_mod2_gate
        )
        return (img, txt)

    def forward_sr(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (img_q, img_k, img_v, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.modulate_img(
            vec, img
        )
        (txt_q, txt_k, txt_v, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.modulate_txt(
            vec_txt, txt
        )
        if freqs_cis is not None:
            (img_qq, img_kk) = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
                f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            )
            (img_q, img_k) = (img_qq, img_kk)
        attn_mode = "flash" if is_flash else self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        (img_attn, txt_attn) = (attn[:, : img_q.shape[1]].contiguous(), attn[:, img_q.shape[1] :].contiguous())
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)), gate=img_mod2_gate
        )
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)), gate=txt_mod2_gate
        )
        return (img, txt)

    def forward(self, bi_inference=True, ar_txt_inference=False, ar_vision_inference=False, **kwargs):
        if bi_inference:
            return self.forward_bi(**kwargs)
        elif ar_txt_inference:
            return self.forward_txt(**kwargs)
        elif ar_vision_inference:
            return self.forward_vision(**kwargs)
        else:
            return self.forward_sr(**kwargs)


class ARHunyuanVideo_1_5_DiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone.

    Args:
        patch_size (list): The size of the patch.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        hidden_size (int): The hidden size of the transformer backbone.
        heads_num (int): The number of attention heads.
        mlp_width_ratio (float): Width ratio for the transformer MLPs.
        mlp_act_type (str): Activation type for the transformer MLPs.
        mm_double_blocks_depth (int): Number of double-stream transformer blocks.
        mm_single_blocks_depth (int): Number of single-stream transformer blocks.
        rope_dim_list (list): Rotary embedding dim for t, h, w.
        qkv_bias (bool): Use bias in qkv projection.
        qk_norm (bool): Whether to use qk norm.
        qk_norm_type (str): Type of qk norm.
        guidance_embed (bool): Use guidance embedding for distillation.
        text_projection (str): Text input projection. Default is "single_refiner".
        use_attention_mask (bool): If to use attention mask.
        text_states_dim (int): Text encoder output dim.
        text_states_dim_2 (int): Secondary text encoder output dim.
        text_pool_type (str): Type for text pooling.
        rope_theta (int): Rotary embedding theta parameter.
        attn_mode (str): Attention mode identifier.
        attn_param (dict): Attention parameter dictionary.
        glyph_byT5_v2 (bool): Use ByT5 glyph module.
        vision_projection (str): Vision condition embedding mode.
        vision_states_dim (int): Vision encoder states input dim.
        is_reshape_temporal_channels (bool): For video VAE adaptation.
        use_cond_type_embedding (bool): Use condition type embedding.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        concat_condition: bool = True,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: list = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        use_meanflow: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        text_pool_type: str = None,
        rope_theta: int = 256,
        attn_mode: str = "flash",
        attn_param: dict = None,
        glyph_byT5_v2: bool = False,
        vision_projection: str = "none",
        vision_states_dim: int = 1280,
        is_reshape_temporal_channels: bool = False,
        use_cond_type_embedding: bool = False,
        ideal_resolution: str = None,
        ideal_task: str = None,
    ):
        super().__init__()
        factory_kwargs = {}
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.attn_mode = attn_mode
        self.text_pool_type = text_pool_type
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.vision_states_dim = vision_states_dim
        self.glyph_byT5_v2 = glyph_byT5_v2
        if self.glyph_byT5_v2:
            self.byt5_in = ByT5Mapper(
                in_dim=1472, out_dim=2048, hidden_dim=2048, out_dim1=hidden_size, use_residual=False
            )
        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.img_in = PatchEmbed(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            is_reshape_temporal_channels=is_reshape_temporal_channels,
            concat_condition=concat_condition,
            **factory_kwargs,
        )
        if vision_projection == "linear":
            self.vision_in = VisionProjection(input_dim=self.vision_states_dim, output_dim=self.hidden_size)
        else:
            self.vision_in = None
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim, self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")
        self.time_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
        self.vector_in = (
            MLPEmbedder(self.config.text_states_dim_2, self.hidden_size, **factory_kwargs)
            if self.text_pool_type is not None
            else None
        )
        self.guidance_in = None
        self.time_r_in = None
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )
        assert mm_single_blocks_depth == 0, "No single block in HunyuanVideo 1.5 architecture"
        self.final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels, get_activation_layer("silu"), **factory_kwargs
        )
        if attn_param is None:
            self.attn_param = {
                "win_size": [[3, 3, 3]],
                "win_type": "fixed",
                "win_ratio": 10,
                "tile_size": [6, 8, 8],
                "ssta_topk": 64,
                "ssta_threshold": 0.0,
                "ssta_lambda": 0.7,
                "ssta_sampling_type": "importance",
                "ssta_adaptive_pool": None,
                "attn_sparse_type": "ssta",
                "attn_pad_type": "zero",
                "attn_use_text_mask": 1,
                "attn_mask_share_within_head": 0,
            }
        else:
            self.attn_param = attn_param
        if attn_mode == "flex-block-attn":
            self.register_to_config(attn_param=self.attn_param)
        if use_cond_type_embedding:
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size)
            self.cond_type_embedding.weight.data.fill_(0)
            assert self.glyph_byT5_v2, "text type embedding is only used when glyph_byT5_v2 is True"
            assert vision_projection is not None, "text type embedding is only used when vision_projection is not None"
        else:
            self.cond_type_embedding = None

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        target_ndim = 3
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        (freqs_cos, freqs_sin) = get_nd_rotary_pos_embed(
            rope_dim_list, rope_sizes, theta=self.rope_theta, use_real=True, theta_rescale_factor=1
        )
        return (freqs_cos, freqs_sin)

    def reorder_txt_token(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):
        if is_reorder:
            reorder_txt = []
            reorder_mask = []
            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()
                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat:
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat(
                        [byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], pad_byt5, pad_text], dim=0
                    )
                else:
                    reorder_txt_i = torch.cat(
                        [
                            byt5_txt_i[byt5_text_mask_i],
                            txt_i[text_mask_i],
                            byt5_txt_i[~byt5_text_mask_i],
                            txt_i[~text_mask_i],
                        ],
                        dim=0,
                    )
                reorder_mask_i = torch.cat(
                    [
                        byt5_text_mask_i[byt5_text_mask_i],
                        text_mask_i[text_mask_i],
                        byt5_text_mask_i[~byt5_text_mask_i],
                        text_mask_i[~text_mask_i],
                    ],
                    dim=0,
                )
                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)
            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.int64)
        return (reorder_txt, reorder_mask)

    def add_action_parameters(self):
        pass

    def add_prope_parameters(self):
        for block in self.double_blocks:
            if hasattr(block, "img_attn_prope_proj"):
                print("Trying to add prope parameters, but img_attn_prope_proj already exists. Skipping...")
                continue
            ref_param = block.img_attn_proj.weight
            block.img_attn_prope_proj = nn.Linear(
                block.hidden_size,
                block.hidden_size,
                bias=block.qkv_bias,
                device=ref_param.device,
                dtype=ref_param.dtype,
            )
            nn.init.zeros_(block.img_attn_prope_proj.weight)
            if block.img_attn_prope_proj.bias is not None:
                nn.init.zeros_(block.img_attn_prope_proj.bias)

    def add_discrete_action_parameters(self):
        if hasattr(self, "action_in"):
            print("Trying to add action_in param that exists. Skipping...")
            return
        self.action_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"))
        nn.init.zeros_(self.action_in.mlp[2].weight)
        if self.action_in.mlp[2].bias is not None:
            nn.init.zeros_(self.action_in.mlp[2].bias)

    def get_text_and_mask(
        self, encoder_attention_mask, text_states, timestep_txt, extra_kwargs, vision_states, mask_type
    ):
        text_mask = encoder_attention_mask
        txt = text_states
        bs = txt.shape[0]
        vec_txt = self.time_in(timestep_txt)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, timestep_txt, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")
        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding(
                torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long)
            )
            txt = txt + cond_emb
        if self.glyph_byT5_v2:
            byt5_text_states = extra_kwargs["byt5_text_states"]
            byt5_text_mask = extra_kwargs["byt5_text_mask"]
            byt5_txt = self.byt5_in(byt5_text_states)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
                )
                byt5_txt = byt5_txt + cond_emb
            (txt, text_mask) = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True)
        if self.vision_in is not None and vision_states is not None:
            extra_encoder_hidden_states = self.vision_in(vision_states)
            if mask_type == "t2v" and torch.all(vision_states == 0):
                extra_attention_mask = torch.zeros(
                    (bs, extra_encoder_hidden_states.shape[1]), dtype=text_mask.dtype, device=text_mask.device
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
            else:
                extra_attention_mask = torch.ones(
                    (bs, extra_encoder_hidden_states.shape[1]), dtype=text_mask.dtype, device=text_mask.device
                )
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    2
                    * torch.ones_like(
                        extra_encoder_hidden_states[:, :, 0],
                        dtype=torch.long,
                        device=extra_encoder_hidden_states.device,
                    )
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb
            (txt, text_mask) = self.reorder_txt_token(extra_encoder_hidden_states, txt, extra_attention_mask, text_mask)
        return (txt, text_mask, vec_txt)

    def forward_txt(
        self,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        vision_states: torch.Tensor = None,
        mask_type="t2v",
        extra_kwargs=None,
        kv_cache: Optional[dict] = None,
        cache_txt: Optional[bool] = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if cache_txt:
            _kv_cache_new = []
            transformer_num_layers = len(self.double_blocks)
            for _ in range(transformer_num_layers):
                _kv_cache_new.append({"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None})
        (txt, text_mask, vec_txt) = self.get_text_and_mask(
            encoder_attention_mask, text_states, timestep_txt, extra_kwargs, vision_states, mask_type
        )
        txt = txt[text_mask.bool().to(txt.device)].unsqueeze(0)
        for index, block in enumerate(self.double_blocks):
            (txt, t_kv) = block(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                txt=txt,
                vec_txt=vec_txt,
                text_mask=None,
                attn_param=None,
                is_flash=False,
                block_idx=index,
                kv_cache=kv_cache,
                cache_txt=cache_txt,
            )
            if cache_txt:
                _kv_cache_new[index]["k_txt"] = t_kv["k_txt"]
                _kv_cache_new[index]["v_txt"] = t_kv["v_txt"]
        if cache_txt:
            return _kv_cache_new

    def forward_vision(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_r=None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        mask_type="t2v",
        extra_kwargs=None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_vision: bool = False,
        rope_temporal_size=4,
        start_rope_start_idx=0,
        action: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if cache_vision:
            _kv_cache_new = []
            transformer_num_layers = len(self.double_blocks)
            for i in range(transformer_num_layers):
                _kv_cache_new.append(
                    {"k_vision": None, "v_vision": None, "k_txt": kv_cache[i]["k_txt"], "v_txt": kv_cache[i]["v_txt"]}
                )
        img = x = hidden_states
        t = timestep
        (bs, _, ot, oh, ow) = x.shape
        (tt, th, tw) = (ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2])
        self.attn_param["thw"] = [tt, th, tw]
        rope_temporal_size = rope_temporal_size // self.patch_size[0]
        if freqs_cos is None and freqs_sin is None:
            (freqs_cos, freqs_sin) = self.get_rotary_pos_embed((rope_temporal_size, th, tw))
            per_latent_size = th * tw
            start_index = start_rope_start_idx * per_latent_size
            end_index = (start_rope_start_idx + tt) * per_latent_size
            freqs_cos = freqs_cos[start_index:end_index, ...]
            freqs_sin = freqs_sin[start_index:end_index, ...]
        img = self.img_in(img)
        t = t.reshape(-1)
        vec = self.time_in(t)
        if action is not None and hasattr(self, "action_in"):
            vec = vec + self.action_in(action.reshape(-1))
        vec = repeat(vec, "(B T) C->B (T H W) C", B=img.shape[0], H=th, W=tw)
        if viewmats is not None:
            viewmats = repeat(viewmats, "B T M N->B (T H W) M N", H=th, W=tw)
            Ks = repeat(Ks, "B T M N->B (T H W) M N", H=th, W=tw)
        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f"Too short context length for SP {sp_size}"
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]
            vec = torch.chunk(vec, sp_size, dim=1)[sp_rank]
            vec = rearrange(vec, "B S C->(B S) C")
            if viewmats is not None:
                viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
                Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]
        else:
            vec = rearrange(vec, "B S C->(B S) C")
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        for index, block in enumerate(self.double_blocks):
            self.attn_param["layer-name"] = f"double_block_{index + 1}"
            (img, vision_kv) = block(
                bi_inference=False,
                ar_txt_inference=False,
                ar_vision_inference=True,
                img=img,
                vec=vec,
                freqs_cis=freqs_cis,
                attn_param=self.attn_param,
                block_idx=index,
                viewmats=viewmats,
                Ks=Ks,
                kv_cache=kv_cache,
                cache_vision=cache_vision,
            )
            if cache_vision:
                _kv_cache_new[index]["k_vision"] = vision_kv["k_vision"]
                _kv_cache_new[index]["v_vision"] = vision_kv["v_vision"]
        if cache_vision:
            return _kv_cache_new
        img = self.final_layer(img, vec)
        if sp_enabled:
            img = sequence_model_parallel_all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        features_list = None
        return (img, features_list)

    def forward_bi(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r=None,
        vision_states: torch.Tensor = None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if guidance is None:
            guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)
        img = x = hidden_states
        t = timestep
        (bs, _, ot, oh, ow) = x.shape
        (tt, th, tw) = (ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2])
        self.attn_param["thw"] = [tt, th, tw]
        if freqs_cos is None and freqs_sin is None:
            (freqs_cos, freqs_sin) = self.get_rotary_pos_embed((tt, th, tw))
        img = self.img_in(img)
        vec = self.time_in(t)
        if action is not None and hasattr(self, "action_in"):
            vec = vec + self.action_in(action.reshape(-1))
        vec = repeat(vec, "(B T) C->B (T H W) C", B=img.shape[0], H=th, W=tw)
        if viewmats is not None:
            viewmats = repeat(viewmats, "B T M N->B (T H W) M N", H=th, W=tw)
            Ks = repeat(Ks, "B T M N->B (T H W) M N", H=th, W=tw)
        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f"Too short context length for SP {sp_size}"
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]
            vec = torch.chunk(vec, sp_size, dim=1)[sp_rank]
            vec = rearrange(vec, "B S C->(B S) C")
            if viewmats is not None:
                viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
                Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]
        else:
            vec = rearrange(vec, "B S C->(B S) C")
        assert text_states_2 is None, (
            "text_states_2 is not None, but handling of text_states_2 is not implemented in forward_bi yet."
        )
        assert self.guidance_embed is False, (
            "guidance_embed is True, but handling of guidance is not implemented in forward_bi yet."
        )
        assert timestep_r is None, (
            "timestep_r is not None, but handling of timestep_r is not implemented in forward_bi yet."
        )
        (txt, text_mask, vec_txt) = self.get_text_and_mask(
            encoder_attention_mask, text_states, timestep_txt, extra_kwargs, vision_states, mask_type
        )
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        for index, block in enumerate(self.double_blocks):
            force_full_attn = (
                self.attn_mode in ["flex-block-attn"]
                and self.attn_param["win_type"] == "hybrid"
                and (self.attn_param["win_ratio"] > 0)
                and ((index + 1) % self.attn_param["win_ratio"] == 0 or index + 1 == len(self.double_blocks))
            )
            self.attn_param["layer-name"] = f"double_block_{index + 1}"
            (img, txt) = block(
                bi_inference=True,
                ar_txt_inference=False,
                ar_vision_inference=False,
                img=img,
                txt=txt,
                vec_txt=vec_txt,
                vec=vec,
                freqs_cis=freqs_cis,
                text_mask=text_mask,
                attn_param=self.attn_param,
                is_flash=force_full_attn,
                block_idx=index,
                viewmats=viewmats,
                Ks=Ks,
            )
        img = self.final_layer(img, vec)
        if sp_enabled:
            img = sequence_model_parallel_all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        assert output_features is False, "output_features is not supported in bi-inference mode currently."
        features_list = None
        return (img, features_list)

    def forward(self, bi_inference=True, ar_txt_inference=False, ar_vision_inference=False, **kwargs):
        if bi_inference:
            return self.forward_bi(**kwargs)
        elif ar_txt_inference:
            return self.forward_txt(**kwargs)
        elif ar_vision_inference:
            return self.forward_vision(**kwargs)
        else:
            raise NotImplementedError

    def unpatchify(self, x, t, h, w):
        """
        Unpatchify a tensorized input back to frame format.

        Args:
            x (Tensor): Input tensor of shape (N, T, patch_size**2 * C)
            t (int): Number of time steps
            h (int): Height in patch units
            w (int): Width in patch units

        Returns:
            Tensor: Output tensor of shape (N, C, t * pt, h * ph, w * pw)
        """
        c = self.unpatchify_channels
        (pt, ph, pw) = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def set_attn_mode(self, attn_mode: str):
        self.attn_mode = attn_mode
        for block in self.double_blocks:
            block.attn_mode = attn_mode
