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

"""Native checkpoint-compatible HunyuanVideo 1.5 MM-DiT.

1.5 keeps the dual-then-single stream layout.  Latents are ``B C T H W``
(optionally channel-concatenated with condition/mask).  :class:`PatchEmbed`
is Conv3d → ``B S C``.  Text may include ByT5 / vision projections.
Double-stream blocks jointly attend video+text QKV; single-stream blocks
run on the concatenated sequence.  Sequence-parallel helpers live in core.
"""

from typing import Any

import torch
import torch.nn as nn

from worldfoundry.core.attention import apply_nd_rotary_embedding as apply_rotary_emb
from worldfoundry.core.attention import get_nd_rotary_pos_embed
from worldfoundry.core.attention.backends.probe import normalize_attention_backend, resolve_attention_backend
from worldfoundry.core.attention.sequence.multimodal_sequence import parallel_attention
from worldfoundry.core.distributed.model_parallel.sequence_mesh_state import get_parallel_state
from worldfoundry.core.distributed.sequence_parallel.runtime import all_gather
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config
from worldfoundry.core.nn import (
    ConcatenatedLinear as LinearWarpforSingle,
)
from worldfoundry.core.nn import ConditioningProjection as TextProjection
from worldfoundry.core.nn import (
    DiTFinalLayer as FinalLayer,
)
from worldfoundry.core.nn import (
    DiTModulation as ModulateDiT,
)
from worldfoundry.core.nn import (
    MLPEmbedder,
    apply_gate,
)
from worldfoundry.core.nn import SinusoidalTimestepEmbedder as TimestepEmbedder
from worldfoundry.core.nn import (
    TransformerMLP as MLP,
)
from worldfoundry.core.nn import (
    activation_layer as get_activation_layer,
)
from worldfoundry.core.nn import (
    modulate_sequence as modulate,
)
from worldfoundry.core.nn import (
    normalization_layer as get_norm_layer,
)

from .camera_attention import sequence_parallel_attention_bi_fa3, sequence_parallel_attention_bi_fp8
from .camera_modulation import fused_layernorm_modulate, fused_residual_gate
from .modules.embed_layers import ByT5Mapper, PatchEmbed, VisionProjection
from .modules.token_refiner import SingleTokenRefiner


def _run_hy15_mlp(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Execute TransformerMLP with optional fused GELU + rowwise FP8 quantization."""
    fc1 = getattr(mlp, "fc1", None)
    act = getattr(mlp, "act", None)
    fc2 = getattr(mlp, "fc2", None)
    drop1 = getattr(mlp, "drop1", None)
    norm = getattr(mlp, "norm", None)
    if (
        fc1 is not None
        and fc2 is not None
        and isinstance(act, nn.GELU)
        and (drop1 is None or getattr(drop1, "p", 0.0) == 0.0)
        and (norm is None or isinstance(norm, nn.Identity))
        and getattr(fc2, "low_precision_enabled", False)
        and getattr(fc2, "scaling", None) == "rowwise"
        and hasattr(fc2, "_forward_quantized")
    ):
        from worldfoundry.core.acceleration.quantization.linear import _fp8_linear_eligible

        hidden = fc1(x)
        if _fp8_linear_eligible(
            hidden,
            fc2.out_features,
            fc2._hardware_eligible,
            min_gemm_work=0 if fc2.weight is None else None,
        ):
            try:
                from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton
            except ImportError:
                return mlp(x)
            mode = "gelu_tanh" if act.approximate == "tanh" else "gelu"
            codes, scales = quantize_rowwise_fp8_triton(
                hidden.reshape(-1, fc2.in_features), fc2.fp8_dtype, activation=mode
            )
            return fc2._forward_quantized(codes, scales, hidden.shape, hidden.dtype)
    return mlp(x)


def _fuse_three_linears(q: nn.Linear, k: nn.Linear, v: nn.Linear) -> nn.Linear:
    """Merge three identical-shape Linear layers into one ``[3*out, in]`` Linear."""
    has_bias = q.bias is not None and k.bias is not None and v.bias is not None
    fused = nn.Linear(
        q.in_features,
        q.out_features * 3,
        bias=has_bias,
        device=q.weight.device,
        dtype=q.weight.dtype,
    )
    with torch.no_grad():
        fused.weight.copy_(torch.cat((q.weight.data, k.weight.data, v.weight.data), dim=0))
        if has_bias:
            fused.bias.copy_(torch.cat((q.bias.data, k.bias.data, v.bias.data), dim=0))
    return fused


class MMDoubleStreamBlock(nn.Module):
    """HY 1.5 dual-stream block: separate AdaLN/QKV, joint attention, separate FFN."""

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str | None = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """Separate AdaLN/QKV/FFN for video and text; attention is joint."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        self.attn_mode = attn_mode

        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )

        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )

        self.hybrid_seq_parallel_attn = None
        self.attention_backend: str | None = attn_mode or "torch"

    def set_attention_backend(self, backend: str | None) -> None:
        """Bind the request-scoped attention backend on this dual-stream block."""
        resolved = "torch" if backend is None else normalize_attention_backend(str(backend))
        self.attention_backend = resolved
        self.attn_mode = resolved

    def fuse_qkv(self) -> int:
        """Merge separate image and text Q/K/V linear projections in place."""
        fused_count = 0
        if not hasattr(self, "img_attn_qkv") and all(
            isinstance(getattr(self, name, None), nn.Linear) for name in ("img_attn_q", "img_attn_k", "img_attn_v")
        ):
            self.img_attn_qkv = _fuse_three_linears(self.img_attn_q, self.img_attn_k, self.img_attn_v)
            del self.img_attn_q, self.img_attn_k, self.img_attn_v
            fused_count += 1
        if not hasattr(self, "txt_attn_qkv") and all(
            isinstance(getattr(self, name, None), nn.Linear) for name in ("txt_attn_q", "txt_attn_k", "txt_attn_v")
        ):
            self.txt_attn_qkv = _fuse_three_linears(self.txt_attn_q, self.txt_attn_k, self.txt_attn_v)
            del self.txt_attn_q, self.txt_attn_k, self.txt_attn_v
            fused_count += 1
        return fused_count

    def enable_deterministic(self):
        """Prefer a bitwise-stable attention kernel when the backend allows it."""
        self.deterministic = True

    def disable_deterministic(self):
        """Allow the faster non-deterministic attention kernel."""
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple | None = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dual-stream block: separate AdaLN/QKV, joint attention, separate FFN.

        ``img`` / ``txt`` are ``B S C``; RoPE is applied to the video stream only.
        """
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        img_modulated = fused_layernorm_modulate(
            img, shift=img_mod1_shift, scale=img_mod1_scale, eps=float(self.img_norm1.eps)
        )

        if hasattr(self, "img_attn_qkv"):
            img_q, img_k, img_v = self.img_attn_qkv(img_modulated).chunk(3, dim=-1)
        else:
            img_q = self.img_attn_q(img_modulated)
            img_k = self.img_attn_k(img_modulated)
            img_v = self.img_attn_v(img_modulated)
        head_dim = img_q.shape[-1] // self.heads_num
        img_q = img_q.view(img_q.shape[0], img_q.shape[1], self.heads_num, head_dim)
        img_k = img_k.view(img_k.shape[0], img_k.shape[1], self.heads_num, head_dim)
        img_v = img_v.view(img_v.shape[0], img_v.shape[1], self.heads_num, head_dim)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
                f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            )
            img_q, img_k = img_qq, img_kk

        txt_modulated = fused_layernorm_modulate(
            txt, shift=txt_mod1_shift, scale=txt_mod1_scale, eps=float(self.txt_norm1.eps)
        )
        if hasattr(self, "txt_attn_qkv"):
            txt_q, txt_k, txt_v = self.txt_attn_qkv(txt_modulated).chunk(3, dim=-1)
        else:
            txt_q = self.txt_attn_q(txt_modulated)
            txt_k = self.txt_attn_k(txt_modulated)
            txt_v = self.txt_attn_v(txt_modulated)
        txt_q = txt_q.view(txt_q.shape[0], txt_q.shape[1], self.heads_num, head_dim)
        txt_k = txt_k.view(txt_k.shape[0], txt_k.shape[1], self.heads_num, head_dim)
        txt_v = txt_v.view(txt_v.shape[0], txt_v.shape[1], self.heads_num, head_dim)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        attn_mode = "flash" if is_flash else (self.attention_backend or self.attn_mode)
        if attn_mode == "cudnn_fp8" and text_mask is None:
            attn = sequence_parallel_attention_bi_fp8(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
                block_idx=block_idx,
                stream_tag="h15_double",
            )
        elif attn_mode == "flash_attention_3" and text_mask is None:
            attn = sequence_parallel_attention_bi_fa3(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
            )
        else:
            eff_attn_mode = "torch" if attn_mode == "cudnn_fp8" else attn_mode
            attn = parallel_attention(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                text_mask=text_mask,
                attn_mode=eff_attn_mode,
                attn_param=attn_param,
                block_idx=block_idx,
            )

        img_attn, txt_attn = attn[:, : img_q.shape[1]].contiguous(), attn[:, img_q.shape[1] :].contiguous()

        img = fused_residual_gate(img, self.img_attn_proj(img_attn), img_mod1_gate)
        img = fused_residual_gate(
            img,
            _run_hy15_mlp(
                self.img_mlp,
                fused_layernorm_modulate(
                    img, shift=img_mod2_shift, scale=img_mod2_scale, eps=float(self.img_norm2.eps)
                ),
            ),
            img_mod2_gate,
        )

        txt = fused_residual_gate(txt, self.txt_attn_proj(txt_attn), txt_mod1_gate)
        txt = fused_residual_gate(
            txt,
            _run_hy15_mlp(
                self.txt_mlp,
                fused_layernorm_modulate(
                    txt, shift=txt_mod2_shift, scale=txt_mod2_scale, eps=float(self.txt_norm2.eps)
                ),
            ),
            txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """HY 1.5 concatenated video+text block; RoPE is applied to the video prefix only."""

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str | None = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.attn_mode = attn_mode

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_mlp = nn.Linear(hidden_size, mlp_hidden_dim, **factory_kwargs)
        self.linear2 = LinearWarpforSingle(hidden_size + mlp_hidden_dim, hidden_size, bias=True, **factory_kwargs)
        self.mlp_act = get_activation_layer(mlp_act_type)()

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.hybrid_seq_parallel_attn = None
        self.attention_backend: str | None = attn_mode or "torch"

    def set_attention_backend(self, backend: str | None) -> None:
        """Bind the request-scoped attention backend on this single-stream block."""
        resolved = "torch" if backend is None else normalize_attention_backend(str(backend))
        self.attention_backend = resolved
        self.attn_mode = resolved

    def fuse_qkv(self) -> int:
        """Merge separate single-stream Q/K/V linear projections in place."""
        if not hasattr(self, "linear1_qkv") and all(
            isinstance(getattr(self, name, None), nn.Linear) for name in ("linear1_q", "linear1_k", "linear1_v")
        ):
            self.linear1_qkv = _fuse_three_linears(self.linear1_q, self.linear1_k, self.linear1_v)
            del self.linear1_q, self.linear1_k, self.linear1_v
            return 1
        return 0

    def enable_deterministic(self):
        """Force the attention backend onto a deterministic path when available."""
        self.deterministic = True

    def disable_deterministic(self):
        """Restore the default (possibly non-deterministic) attention backend."""
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
    ) -> torch.Tensor:
        """Gated self-attn + fused MLP on joint ``B (S_img+S_txt) C`` tokens."""
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

        if hasattr(self, "linear1_qkv"):
            q, k, v = self.linear1_qkv(x_mod).chunk(3, dim=-1)
        else:
            q = self.linear1_q(x_mod)
            k = self.linear1_k(x_mod)
            v = self.linear1_v(x_mod)

        head_dim = q.shape[-1] // self.heads_num
        q = q.view(q.shape[0], q.shape[1], self.heads_num, head_dim)
        k = k.view(k.shape[0], k.shape[1], self.heads_num, head_dim)
        v = v.view(v.shape[0], v.shape[1], self.heads_num, head_dim)

        mlp = self.linear1_mlp(x_mod)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
            f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        )
        img_q, img_k = img_qq, img_kk

        attn_mode = "flash" if is_flash else (self.attention_backend or self.attn_mode)
        if attn_mode == "cudnn_fp8" and text_mask is None:
            from .camera_attention import sequence_parallel_attention_bi_fp8

            attn = sequence_parallel_attention_bi_fp8(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
                stream_tag="h15_single",
            )
        else:
            eff_attn_mode = "torch" if attn_mode == "cudnn_fp8" else attn_mode
            attn = parallel_attention(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                text_mask=text_mask,
                attn_mode=eff_attn_mode,
                attn_param=attn_param,
            )
        output = self.linear2(attn, self.mlp_act(mlp))

        return x + apply_gate(output, gate=mod_gate)


class HunyuanVideo_1_5_DiffusionTransformer(nn.Module, NativeConfigMixin):
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
        patch_size: list = [1, 2, 2],  # noqa: B006 - Keeps the upstream default object and signature.
        in_channels: int = 4,
        concat_condition: bool = True,
        out_channels: int | None = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: list = [16, 56, 56],  # noqa: B006 - Keeps the upstream default object and signature.
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        use_meanflow: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        text_pool_type: str | None = None,
        rope_theta: int = 256,
        attn_mode: str = "flash",
        attn_param: dict | None = None,
        glyph_byT5_v2: bool = False,
        vision_projection: str = "none",
        vision_states_dim: int = 1280,
        is_reshape_temporal_channels: bool = False,
        use_cond_type_embedding: bool = False,
        ideal_resolution: str | None = None,
        ideal_task: str | None = None,
    ):
        """Patch-embed ``B C T H W``, run dual- then single-stream MM-DiT blocks, unpatchify."""
        super().__init__()
        factory_kwargs = {}
        attn_mode = normalize_attention_backend(attn_mode)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
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

        # Vision projection
        if vision_projection == "linear":
            self.vision_in = VisionProjection(input_dim=self.vision_states_dim, output_dim=self.hidden_size)
        else:
            self.vision_in = None

        # Text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                text_states_dim,
                hidden_size,
                heads_num,
                depth=2,
                **factory_kwargs,
            )
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        # time modulation
        self.time_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
        self.vector_in = (
            MLPEmbedder(self.config.text_states_dim_2, self.hidden_size, **factory_kwargs)
            if self.text_pool_type is not None
            else None
        )
        self.guidance_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
            if guidance_embed
            else None
        )

        self.time_r_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs) if use_meanflow else None
        )

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

        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        # STA
        if attn_param is None:
            self.attn_param = {
                # STA
                "win_size": [[3, 3, 3]],
                "win_type": "fixed",
                "win_ratio": 10,
                "tile_size": [6, 8, 8],
                # SSTA
                "ssta_topk": 64,
                "ssta_threshold": 0.0,
                "ssta_lambda": 0.7,
                "ssta_sampling_type": "importance",
                "ssta_adaptive_pool": None,
                # FlexBlockAttention:
                "attn_sparse_type": "ssta",
                "attn_pad_type": "zero",
                "attn_use_text_mask": 1,
                "attn_mask_share_within_head": 0,
            }
        else:
            self.attn_param = attn_param

        if attn_mode == "flex_block_attention":
            self.register_to_config(attn_param=self.attn_param)

        if use_cond_type_embedding:
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size)
            self.cond_type_embedding.weight.data.fill_(0)
            assert self.glyph_byT5_v2, "text type embedding is only used when glyph_byT5_v2 is True"
            assert vision_projection is not None, "text type embedding is only used when vision_projection is not None"
            # 0: text_encoder feature
            # 1: byt5 feature
            # 2: vision_encoder feature
        else:
            self.cond_type_embedding = None
        self.attention_backend: str | None = attn_mode
        self._rope_cache: dict[tuple[int, int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def fuse_qkv(self) -> int:
        """Fuse Q/K/V projections across all double and single stream blocks."""
        total = 0
        for block in self.double_blocks:
            total += block.fuse_qkv()
        for block in self.single_blocks:
            total += block.fuse_qkv()
        return total

    def enable_fp8(
        self,
        *,
        exclude_edge_blocks: bool = True,
        scaling: str = "rowwise",
        use_fast_accum: bool = True,
        min_features: int = 1024,
        keep_dense_fallback: bool = False,
    ) -> int:
        """Apply Hopper/Blackwell FP8 W8A8 quantization to eligible transformer linears."""
        from worldfoundry.core.acceleration.quantization.linear import replace_linear_with_float8

        excludes = [
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "time_r_in",
            "byt5_in",
            "vision_in",
            "final_layer",
            "img_mod",
            "txt_mod",
            "modulation",
        ]
        if exclude_edge_blocks:
            if len(self.double_blocks) > 0:
                excludes.append("double_blocks.0.")
            if len(self.single_blocks) > 0:
                excludes.append(f"single_blocks.{len(self.single_blocks) - 1}.")
            elif len(self.double_blocks) >= 2:
                excludes.append(f"double_blocks.{len(self.double_blocks) - 1}.")
        return replace_linear_with_float8(
            self,
            exclude=tuple(excludes),
            min_features=min_features,
            scaling=scaling,
            use_fast_accum=use_fast_accum,
            keep_dense_fallback=keep_dense_fallback,
        )

    def set_attention_backend(self, backend: str | None) -> None:
        """Bind the request-scoped attention provider across all double and single blocks."""
        self.set_attn_mode("torch" if backend is None else str(backend))

    def enable_deterministic(self):
        """Propagate deterministic attention to every dual/single-stream block."""
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        """Propagate the fast attention kernel to every dual/single-stream block."""
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        """Build 3D RoPE ``(cos, sin)`` for the post-patch lattice ``rope_sizes``."""
        target_ndim = 3
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            tuple(rope_dim_list),
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

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
                    # When using block mask with approximate computation, set pad to zero to reduce error
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

        return reorder_txt, reorder_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r=None,
        vision_states: torch.Tensor = None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: dict[str, Any] | None = None,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Denoise ``B C T H W`` latents through double- then single-stream blocks.

        Optional ``vision_states`` / ByT5 glyph tokens are projected into the
        text stream.  ``mask_type`` selects t2v vs i2v channel packing.
        """

        if guidance is None:
            guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)

        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = text_states
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        self.attn_param["thw"] = [tt, th, tw]
        if freqs_cos is None and freqs_sin is None:
            rope_key = (tt, th, tw, hidden_states.device)
            cached_rope = self._rope_cache.get(rope_key)
            if cached_rope is None:
                c_cos, c_sin = self.get_rotary_pos_embed((tt, th, tw))
                cached_rope = (c_cos.to(device=hidden_states.device), c_sin.to(device=hidden_states.device))
                if len(self._rope_cache) >= 16:
                    self._rope_cache.clear()
                self._rope_cache[rope_key] = cached_rope
            freqs_cos, freqs_sin = cached_rope

        img = self.img_in(img)
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

        # Prepare modulation vectors
        vec = self.time_in(t)

        if text_states_2 is not None:
            vec_2 = self.vector_in(text_states_2)
            vec = vec + vec_2

        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(guidance)

        if timestep_r is not None:
            vec = vec + self.time_r_in(timestep_r)

        # Embed text tokens
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
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
            txt, text_mask = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True)

        if self.vision_in is not None and vision_states is not None:
            extra_encoder_hidden_states = self.vision_in(vision_states)
            # If t2v, set extra_attention_mask to 0 to avoid attention to semantic tokens
            if mask_type == "t2v" and torch.all(vision_states == 0):
                extra_attention_mask = torch.zeros(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
                # Set vision tokens to zero to mitigate potential block mask error in SSTA
                extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
            else:
                extra_attention_mask = torch.ones(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
            # Ensure valid tokens precede padding tokens
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

            txt, text_mask = self.reorder_txt_token(extra_encoder_hidden_states, txt, extra_attention_mask, text_mask)

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        block_text_mask = text_mask
        if block_text_mask is not None and self.attn_mode in {
            "torch",
            "flash_attention_3",
            "sol_attn",
            "cudnn_fp8",
            "triton_tma",
            "scaled_int8_fp8",
        }:
            # Validate once outside the block stack and pass torch.bool (or None if all-valid)
            # so parallel_attention skips per-block .max()/.min()/.all() host syncs
            # and captures cleanly inside piecewise CUDA Graphs.
            if not torch.cuda.is_current_stream_capturing():
                if block_text_mask.dtype in {torch.int64, torch.int32}:
                    assert block_text_mask.max() <= 1 and block_text_mask.min() >= 0, (
                        "Integer attention mask must be between 0 and 1 for torch attention."
                    )
                    block_text_mask = block_text_mask.to(torch.bool)
                if block_text_mask.dtype == torch.bool:
                    if bool(block_text_mask.all()):
                        block_text_mask = None
                    elif (
                        block_text_mask.ndim == 2
                        and block_text_mask.shape[1] == txt.shape[1]
                        and (block_text_mask.shape[0] == 1 or bool((block_text_mask == block_text_mask[:1]).all()))
                    ):
                        valid_idx = block_text_mask[0].nonzero(as_tuple=False).squeeze(-1)
                        if valid_idx.numel() > 0:
                            txt = txt.index_select(1, valid_idx).contiguous()
                            block_text_mask = None
            elif block_text_mask.dtype in {torch.int64, torch.int32}:
                block_text_mask = block_text_mask.to(torch.bool)
        piecewise_graph = getattr(self, "_active_piecewise_graph", None)
        if piecewise_graph is not None and not output_features and not self.training:

            def _run_double_stack(img_in, txt_in, vec_in, cos_in, sin_in, mask_in):
                f_cis = (cos_in, sin_in) if cos_in is not None else None
                for idx, blk in enumerate(self.double_blocks):
                    force_full = (
                        self.attn_mode in ["flex_block_attention"]
                        and self.attn_param["win_type"] == "hybrid"
                        and self.attn_param["win_ratio"] > 0
                        and ((idx + 1) % self.attn_param["win_ratio"] == 0 or (idx + 1) == len(self.double_blocks))
                    )
                    img_in, txt_in = blk(
                        img=img_in,
                        txt=txt_in,
                        vec=vec_in,
                        freqs_cis=f_cis,
                        text_mask=mask_in,
                        attn_param=self.attn_param,
                        is_flash=force_full,
                        block_idx=idx,
                    )
                return img_in, txt_in

            piecewise_graph.get_or_register_segment("double_blocks", _run_double_stack)
            img, txt = piecewise_graph.run_segment(
                "double_blocks", img, txt, vec, freqs_cos, freqs_sin, block_text_mask
            )
            txt_seq_len = txt.shape[1]
            img_seq_len = img.shape[1]
            x = torch.cat((img, txt), 1)
            features_list = None
            if len(self.single_blocks) > 0:

                def _run_single_stack(x_in, vec_in, cos_in, sin_in, mask_in, txt_seq_len_val):
                    for idx, blk in enumerate(self.single_blocks):
                        force_full = (
                            self.attn_mode in ["flex_block_attention"]
                            and self.attn_param["win_type"] == "hybrid"
                            and self.attn_param["win_ratio"] > 0
                            and ((idx + 1) % self.attn_param["win_ratio"] == 0 or (idx + 1) == len(self.single_blocks))
                        )
                        x_in = blk(
                            x=x_in,
                            vec=vec_in,
                            txt_len=txt_seq_len_val,
                            freqs_cis=(cos_in, sin_in),
                            text_mask=mask_in,
                            attn_param=self.attn_param,
                            is_flash=force_full,
                        )
                    return x_in

                piecewise_graph.get_or_register_segment("single_blocks", _run_single_stack)
                x = piecewise_graph.run_segment(
                    "single_blocks", x, vec, freqs_cos, freqs_sin, block_text_mask, txt_seq_len
                )
        else:
            # Pass through double-stream blocks
            for index, block in enumerate(self.double_blocks):
                force_full_attn = (
                    self.attn_mode in ["flex_block_attention"]
                    and self.attn_param["win_type"] == "hybrid"
                    and self.attn_param["win_ratio"] > 0
                    and ((index + 1) % self.attn_param["win_ratio"] == 0 or (index + 1) == len(self.double_blocks))
                )
                self.attn_param["layer-name"] = f"double_block_{index + 1}"
                img, txt = block(
                    img=img,
                    txt=txt,
                    vec=vec,
                    freqs_cis=freqs_cis,
                    text_mask=block_text_mask,
                    attn_param=self.attn_param,
                    is_flash=force_full_attn,
                    block_idx=index,
                )

            txt_seq_len = txt.shape[1]
            img_seq_len = img.shape[1]

            # Merge image and text for single-stream blocks
            x = torch.cat((img, txt), 1)
            features_list = [] if output_features else None
            if len(self.single_blocks) > 0:
                for index, block in enumerate(self.single_blocks):
                    force_full_attn = (
                        self.attn_mode in ["flex_block_attention"]
                        and self.attn_param["win_type"] == "hybrid"
                        and self.attn_param["win_ratio"] > 0
                        and ((index + 1) % self.attn_param["win_ratio"] == 0 or (index + 1) == len(self.single_blocks))
                    )
                    self.attn_param["layer-name"] = f"single_block_{index + 1}"
                    x = block(
                        x=x,
                        vec=vec,
                        txt_len=txt_seq_len,
                        freqs_cis=(freqs_cos, freqs_sin),
                        text_mask=block_text_mask,
                        attn_param=self.attn_param,
                        is_flash=force_full_attn,
                    )
                    if output_features and index % output_features_stride == 0:
                        features_list.append(x[:, :img_seq_len, ...])
        img = x[:, :img_seq_len, ...]

        # Final Layer
        img = self.final_layer(img, vec)
        if sp_enabled:
            img = all_gather(img, dim=1, group=parallel_dims.sp_group)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        if output_features:
            features_list = torch.stack(features_list, dim=0)
            if sp_enabled:
                features_list = all_gather(features_list, dim=2, group=parallel_dims.sp_group)
        else:
            features_list = None
        return (img, features_list)

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
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def set_attn_mode(self, attn_mode: str):
        normalized_attn_mode = normalize_attention_backend(attn_mode)
        device = next(self.parameters()).device
        attn_mode = resolve_attention_backend(
            normalized_attn_mode,
            device,
            # Hunyuan's multimodal attention facade owns the SSTA/FlexBlock
            # mask and block-geometry planner. Other model-specific names must
            # retain generic fail-fast behavior.
            allow_model_specific=normalized_attn_mode == "flex_block_attention",
        )
        self.attn_mode = attn_mode
        self.attention_backend = attn_mode
        for block in self.double_blocks:
            block.attn_mode = attn_mode
            block.attention_backend = attn_mode
        for block in self.single_blocks:
            block.attn_mode = attn_mode
            block.attention_backend = attn_mode
