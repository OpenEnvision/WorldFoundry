"""Wan 2.2 action-conditioned DiT (per-token timestep + optional INT8).

Official-layout Wan 2.2 packing with action residuals and optional
Triton INT8 Q projections.  ``rope_apply_with_indices`` supports
non-contiguous frame indices for action-aligned RoPE.

Adds action conditioning.  No VACE, camera ControlNet, or linear attn.
"""
import math
import re
import torch
import time
import torch.nn as nn
import torch.nn.functional as torch_F
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config
from einops import rearrange
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.action_attention_22 import flash_attention
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.action_conditioning_22 import ActionModule

# --- Int8 Quantization Support ---
_CACHED_TRITON_KERNELS = None

def _get_triton_kernels():
    """Helper function to get triton kernels."""
    global _CACHED_TRITON_KERNELS
    if _CACHED_TRITON_KERNELS is not None:
        return _CACHED_TRITON_KERNELS
    try:
        from worldfoundry.core.kernels import quantized_gemm as triton_kernels

        _CACHED_TRITON_KERNELS = triton_kernels
        return _CACHED_TRITON_KERNELS
    except (ImportError, RuntimeError):
        return None

class Int8Linear(nn.Module):
    """INT8-weight linear used for optional quantized Q projections."""
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """Init.

        Args:
            in_features: The in features.
            out_features: The out features.
            bias: The bias.
            device: The device.
            dtype: The dtype.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_dtype = torch.int8
        
        self.register_buffer('weight_int8', torch.zeros((out_features, in_features), device=device, dtype=torch.int8))
        self.register_buffer('weight_scales', torch.zeros((out_features,), device=device, dtype=torch.float32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=torch.bfloat16))
        else:
            self.register_parameter('bias', None)

    def from_float(self, weight_fp):
        """From float.

        Args:
            weight_fp: The weight fp.
        """
        with torch.no_grad():
            w = weight_fp.float()
            scales = w.abs().max(dim=1)[0] / 127.0
            scales = scales.clamp(min=1e-12)
            w_int8 = (w / scales.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)
            
            self.weight_int8.copy_(w_int8)
            self.weight_scales.copy_(scales)

    @property
    def weight(self):
        """Weight."""
        class DummyWeight:
            """Placeholder parameter holder for INT8 scale / zero-point metadata."""
            def __init__(self, dtype):
                """Init.

                Args:
                    dtype: The dtype.
                """
                self.dtype = dtype
        return DummyWeight(self.weight_dtype)

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        triton_kernels = _get_triton_kernels()
        orig_dtype = x.dtype
        verify_mode = getattr(self, 'verify_mode', False)

        if triton_kernels is None:
            if not hasattr(self, '_warned_no_triton'):
                print("CRITICAL WARNING: Triton kernels NOT found! Falling back to slow BF16 emulation.")
                self._warned_no_triton = True
            target_dtype = x.dtype if x.dtype != torch.int8 else torch.bfloat16
            x_bf16 = x.to(target_dtype)
            w_bf16 = (self.weight_int8.to(target_dtype) * 
                      self.weight_scales.view(-1, 1).to(target_dtype))
            b_bf16 = self.bias.to(target_dtype) if self.bias is not None else None
            return torch_F.linear(x_bf16, w_bf16, b_bf16).to(orig_dtype)

        if self.in_features < 16:
            target_dtype = x.dtype if x.dtype != torch.int8 else torch.bfloat16
            x_bf16 = x.to(target_dtype)
            w_bf16 = (self.weight_int8.to(target_dtype) * 
                      self.weight_scales.view(-1, 1).to(target_dtype))
            b_bf16 = self.bias.to(target_dtype) if self.bias is not None else None
            return torch_F.linear(x_bf16, w_bf16, b_bf16).to(orig_dtype)

        x_shape = x.shape
        x_flat = x.reshape(-1, x_shape[-1]).clone()
        M, K = x_flat.shape
        N = self.out_features
        
        if not hasattr(Int8Linear, '_global_last_shapes'):
            Int8Linear._global_last_shapes = set()
        
        shape_key = (M, N, K)
        if shape_key not in Int8Linear._global_last_shapes:
            rank = 0
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            if rank == 0:
                print(f"DEBUG: [Int8Linear] New Shape detected: M={M}, N={N}, K={K}. JIT Compiling/Switching...", flush=True)
            Int8Linear._global_last_shapes.add(shape_key)

        try:
            x_int8, x_scales = triton_kernels.int8_quantize_triton(x_flat)
            
            out = triton_kernels.int8_gemm_triton(
                x_int8, 
                self.weight_int8, 
                x_scales.view(-1), 
                self.weight_scales.view(-1), 
                output_dtype=torch.bfloat16
            )
        except RuntimeError as e:
            if "Triton Error" not in str(e) and "device kernel image is invalid" not in str(e):
                raise
            if not hasattr(self, '_warned_triton_runtime'):
                print(f"CRITICAL WARNING: Triton int8 runtime failed ({e}). Falling back to slow BF16 emulation.")
                self._warned_triton_runtime = True
            target_dtype = x.dtype if x.dtype != torch.int8 else torch.bfloat16
            x_bf16 = x.to(target_dtype)
            w_bf16 = (
                self.weight_int8.to(target_dtype)
                * self.weight_scales.view(-1, 1).to(target_dtype)
            )
            b_bf16 = self.bias.to(target_dtype) if self.bias is not None else None
            return torch_F.linear(x_bf16, w_bf16, b_bf16).to(orig_dtype)
        
        if verify_mode:
            with torch.no_grad():
                x_int8_f = x_int8.float()
                w_int8_f = self.weight_int8.t().float()
                acc_int32 = torch.matmul(x_int8_f, w_int8_f)
                out_simulated = acc_int32 * x_scales.view(-1, 1).float() * self.weight_scales.view(1, -1).float()
                out_simulated = out_simulated.to(torch.bfloat16)
                cos_sim_kernel = torch_F.cosine_similarity(out.flatten().float(), out_simulated.flatten().float(), dim=0, eps=1e-8).item()
                
                w_bf16 = self.weight_int8.to(torch.bfloat16) * self.weight_scales.view(-1, 1).to(torch.bfloat16)
                out_bf16 = torch_F.linear(x_flat.to(torch.bfloat16), w_bf16)
                cos_sim_bf16 = torch_F.cosine_similarity(out.flatten().float(), out_bf16.flatten().float(), dim=0, eps=1e-8).item()
                
                if not hasattr(Int8Linear, '_global_stats'):
                    Int8Linear._global_stats = {
                        'kernel_sims': [], 
                        'bf16_sims': [],
                        'max_w': 0.0,
                        'max_x': 0.0,
                        'max_out': 0.0,
                        'low_sim_count': 0  
                    }
                Int8Linear._global_stats['kernel_sims'].append(cos_sim_kernel)
                Int8Linear._global_stats['bf16_sims'].append(cos_sim_bf16)
                
                if cos_sim_bf16 < 0.99:
                    Int8Linear._global_stats['low_sim_count'] += 1
                
                w_max = self.weight_int8.abs().max().item()
                x_max = x_flat.abs().max().item()
                out_max = out.abs().max().item()
                Int8Linear._global_stats['max_w'] = max(Int8Linear._global_stats['max_w'], w_max)
                Int8Linear._global_stats['max_x'] = max(Int8Linear._global_stats['max_x'], x_max)
                Int8Linear._global_stats['max_out'] = max(Int8Linear._global_stats['max_out'], out_max)

                threshold = 0.96
                if cos_sim_kernel < 0.99 or cos_sim_bf16 < threshold:
                    print(f"⚠️ [Quant Warning] Low Similarity! M={M} | KernelSim: {cos_sim_kernel:.4f} | BF16Sim: {cos_sim_bf16:.4f} | OutMax: {out_max:.4f} | WMax: {w_max:.4f} | XMax: {x_max:.4f}")
        
        out = out.view(*x_shape[:-1], self.out_features)
        
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
            
        return out.to(orig_dtype)

def convert_model_to_int8(model, target_layers=["q"]):
    """Convert model to int8.

    Args:
        model: The model.
        target_layers: The target layers.
    """

    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    if rank == 0:
        print(f"Starting Int8 quantization for layers: {target_layers}")
    
    quantized_types = set()
    def _recursive_convert(module, prefix=""):
        """Helper function to recursive convert.

        Args:
            module: The module.
            prefix: The prefix.
        """
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            is_target = False
            if isinstance(child, nn.Linear):
                for target in target_layers:
                    if target in ["q", "k", "v", "o"]:
                        if name == target: 
                            is_target = True
                            break
                    elif target in name: 
                        is_target = True
                        break

            if is_target:
                simplified_name = re.sub(r'\.\d+\.', '.', full_name)
                if rank == 0 and simplified_name not in quantized_types:
                    # print(f"  Quantizing {simplified_name} ({child.in_features}x{child.out_features})")
                    quantized_types.add(simplified_name)
                
                new_layer = Int8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device
                )
                new_layer.from_float(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data.to(torch.bfloat16))
                
                setattr(module, name, new_layer)
            else:
                _recursive_convert(child, full_name)

    _recursive_convert(model)
    if rank == 0:
        print("Quantization conversion finished.")
    return model

def quantize_model_to_int8(model):
    """
    Deprecated: Use convert_model_to_int8 instead.
    """
    return convert_model_to_int8(model, target_layers=["q", "k", "v", "o", "ffn"])

# --- End Int8 Quantization Support ---

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    """Classic sinusoidal features for 1-D timestep / position ids."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """Build complex cis frequencies for one RoPE axis."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """Apply per-sample 3D RoPE to packed tokens; leave pad tokens unrotated."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []    
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i.to(x_i.dtype)).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()

@torch.amp.autocast('cuda', enabled=False)
def rope_apply_with_indices(x, grid_sizes, freqs, t_indices=None):
    """
    Apply RoPE to input tensor using precomputed freqs and optional time indices.
    freqs shape: [max_len, C/2] or [num_heads, max_len, C/2]
    """
    n, c = x.size(2), x.size(3) // 2

    if freqs.dim() == 3:
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=2)
    else:
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))

        if t_indices is None:
            t_idx = torch.arange(f, device=freqs[0].device)
        else:
            t_idx = t_indices
            if torch.is_tensor(t_idx) and t_idx.dim() > 1:
                t_idx = t_idx[i]
            if not torch.is_tensor(t_idx):
                t_idx = torch.tensor(t_idx, device=freqs[0].device)
            elif t_idx.device != freqs[0].device:
                t_idx = t_idx.to(freqs[0].device)
            t_idx = t_idx.to(dtype=torch.long)

        if freqs[0].dim() == 3:
            t_freqs = freqs[0][:, t_idx, :]  # [n, f, c_t]
            h_freqs = freqs[1][:, :h, :]     # [n, h, c_h]
            w_freqs = freqs[2][:, :w, :]     # [n, w, c_w]

            freqs_i = torch.cat([
                t_freqs.permute(1, 0, 2).view(f, 1, 1, n, -1).expand(f, h, w, n, -1),
                h_freqs.permute(1, 0, 2).view(1, h, 1, n, -1).expand(f, h, w, n, -1),
                w_freqs.permute(1, 0, 2).view(1, 1, w, n, -1).expand(f, h, w, n, -1),
            ], dim=-1).reshape(seq_len, n, -1)
        else:
            t_freqs = freqs[0][t_idx]
            h_freqs = freqs[1][:h]
            w_freqs = freqs[2][:w]

            freqs_i = torch.cat([
                t_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
                h_freqs.view(1, h, 1, -1).expand(f, h, w, -1),
                w_freqs.view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i.to(x_i.dtype)).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)

    return torch.stack(output).float()

class WanRMSNorm(nn.Module):
    """RMSNorm with a learned scale; compute in fp32 when the impl upcasts."""

    def __init__(self, dim, eps=1e-5):
        """Init.

        Args:
            dim: The dim.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).to(x.dtype) * self.weight

    def _norm(self, x):
        """RMS-normalize the last dim of ``x``."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm that typically upcasts to fp32, then restores the activation dtype."""

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        """Init.

        Args:
            dim: The dim.
            eps: The eps.
            elementwise_affine: The elementwise affine.
        """
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return torch.nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        ).type_as(x)


class WanSelfAttention(nn.Module):
    """Wan self-attention: RMSNorm Q/K, 3D RoPE, flash or variant kernel."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 use_memory=False):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            eps: The eps.
            use_memory: The use memory.
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_memory = use_memory

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, memory_length=0,
                memory_latent_idx=None, predict_latent_idx=None, fa_version=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            memory_length: Number of memory frames
            memory_latent_idx: Actual latent indices for memory frames (list/array)
            predict_latent_idx: Tuple of (start_idx, end_idx) for predict frames
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            """Project and reshape QKV to ``[B, L, H, D]``."""
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        x_orig = x 
        q = self.norm_q(self.q(x_orig)).view(b, s, n, d)
        k = self.norm_k(self.k(x_orig)).view(b, s, n, d)
        v = self.v(x_orig).view(b, s, n, d)

        if self.use_memory:
            if memory_length > 0:
                '''
                memory_length: the length of the memory
                grid_sizes: the grid sizes of the input
                add rope to memory and pred frames separately.
                '''
                hw = grid_sizes[0][1]*grid_sizes[0][2]
                q_pred = q[:, memory_length*hw:, :]
                k_pred = k[:, memory_length*hw:, :]
                grid_sizes_pred = grid_sizes.clone()
                grid_sizes_pred[:,0] = grid_sizes_pred[:,0] - memory_length
                
                if predict_latent_idx is not None:
                    if isinstance(predict_latent_idx, tuple) and len(predict_latent_idx) == 2:
                        start_idx, end_idx = predict_latent_idx
                        pred_indices = list(range(start_idx, end_idx))
                    else:
                        pred_indices = predict_latent_idx
                else:
                    pred_indices = list(range(grid_sizes_pred[0][0].item()))

                q_pred = rope_apply_with_indices(q_pred, grid_sizes_pred, freqs, pred_indices)
                k_pred = rope_apply_with_indices(k_pred, grid_sizes_pred, freqs, pred_indices)

                q_memory = q[:, :memory_length*hw, :]
                k_memory = k[:, :memory_length*hw, :]
                grid_sizes_mem = grid_sizes.clone()
                grid_sizes_mem[:,0] = memory_length
                
                if memory_latent_idx is not None:
                    mem_indices = memory_latent_idx
                else:
                    mem_indices = list(range(memory_length))

                q_memory = rope_apply_with_indices(q_memory, grid_sizes_mem, freqs, mem_indices)
                k_memory = rope_apply_with_indices(k_memory, grid_sizes_mem, freqs, mem_indices)

                q = torch.cat([q_memory, q_pred], dim=1)
                k = torch.cat([k_memory, k_pred], dim=1)
            else:
                if predict_latent_idx is not None:
                    if isinstance(predict_latent_idx, tuple) and len(predict_latent_idx) == 2:
                        start_idx, end_idx = predict_latent_idx
                        pred_indices = list(range(start_idx, end_idx))
                    else:
                        pred_indices = predict_latent_idx
                else:
                    pred_indices = list(range(grid_sizes[0][0].item()))

                q = rope_apply_with_indices(q, grid_sizes, freqs, pred_indices)
                k = rope_apply_with_indices(k, grid_sizes, freqs, pred_indices)
        else:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
            version=fa_version)

        x = x.flatten(2)
        x = self.o(x)

        return x


class WanCrossAttention(WanSelfAttention):
    """Wan text cross-attention (optional KV cache)."""

    def forward(self, x, context, context_lens, fa_version=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x.to(torch.bfloat16))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(torch.bfloat16))).view(b, -1, n, d)
        v = self.v(context.to(torch.bfloat16)).view(b, -1, n, d)
        x = flash_attention(q, k, v, k_lens=context_lens, version=fa_version)

        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    """Wan transformer block: AdaLN self-attn, cross-attn, then AdaLN FFN."""

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 action_config = {},block_idx=0, use_memory=False):
        """Init.

        Args:
            dim: The dim.
            ffn_dim: The ffn dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            cross_attn_norm: The cross attn norm.
            eps: The eps.
            action_config: The action config.
            block_idx: The block idx.
            use_memory: The use memory.
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_memory = use_memory
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, use_memory=use_memory)
        if len(action_config) != 0 and block_idx in action_config['blocks']:
            self.action_model = ActionModule(**action_config)
        else:
            self.action_model = None
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        if use_memory:
            self.cam_injector_layer1 = nn.Linear(dim, dim)
            self.cam_injector_layer2 = nn.Linear(dim, dim)
            self.cam_scale_layer = nn.Linear(dim, dim)
            self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        mouse_cond=None,
        keyboard_cond=None,
        plucker_emb=None,
        mouse_cond_memory=None,
        keyboard_cond_memory=None,
        memory_length=0,
        memory_latent_idx=None,
        predict_latent_idx=None,
        fa_version=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        if self.use_memory:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                y = self.self_attn(
                    (self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)).to(x.dtype),
                    seq_lens, grid_sizes, freqs, memory_length,
                    memory_latent_idx=memory_latent_idx,
                    predict_latent_idx=predict_latent_idx, fa_version=fa_version)
        else:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                y = self.self_attn(
                    (self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)).to(x.dtype),
                    seq_lens, grid_sizes, freqs, fa_version=fa_version)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        if plucker_emb is not None:
            c2ws_hidden_states = self.cam_injector_layer2(
                torch_F.silu(self.cam_injector_layer1(plucker_emb))
            )
            c2ws_hidden_states = c2ws_hidden_states + plucker_emb
            cam_scale = self.cam_scale_layer(c2ws_hidden_states)
            cam_shift = self.cam_shift_layer(c2ws_hidden_states)
            x = (1.0 + cam_scale) * x + cam_shift

        def cross_attn_ffn(x, context, context_lens, e, mouse_cond=None, keyboard_cond=None, mouse_cond_memory=None, keyboard_cond_memory=None, fa_version=None):
            """Cross attn ffn.

            Args:
                x: The x.
                context: The context.
                context_lens: The context lens.
                e: The e.
                mouse_cond: The mouse cond.
                keyboard_cond: The keyboard cond.
                mouse_cond_memory: The mouse cond memory.
                keyboard_cond_memory: The keyboard cond memory.
                fa_version: The fa version.
            """
            if mouse_cond is not None or self.use_memory:
                x = self.norm3(x)
                x = x + self.cross_attn(x, context, context_lens, fa_version=fa_version)
            else:
                x = x + self.cross_attn(self.norm3(x), context, context_lens, fa_version=fa_version)

            if self.action_model is not None:
                from worldfoundry.core.distributed.sequence_parallel.ops import get_world_size, get_rank, gather_forward
                dtype = self.ffn[0].weight.dtype
                
                valid_len = int(grid_sizes[0].prod().item())

                is_dist = torch.distributed.is_initialized()
                world_size = get_world_size() if is_dist else 1

                if self.use_memory:
                    if world_size > 1:
                        x_full = gather_forward(x.contiguous(), dim=1)
                        
                        x_valid = x_full[:, :valid_len, :]
                        x_valid = self.action_model(x_valid.to(dtype), grid_sizes[0][0], grid_sizes[0][1], grid_sizes[0][2], mouse_cond, keyboard_cond, mouse_cond_memory, keyboard_cond_memory)

                        if x_valid.shape[1] < x_full.shape[1]:
                            x_full = torch.cat([x_valid, x_full[:, valid_len:, :]], dim=1)
                        else:
                            x_full = x_valid

                        x = torch.chunk(x_full.contiguous(), world_size, dim=1)[get_rank() if is_dist else 0].contiguous()
                    else:
                        x_valid = x[:, :valid_len, :]
                        x_valid = self.action_model(x_valid.to(dtype), grid_sizes[0][0], grid_sizes[0][1], grid_sizes[0][2], mouse_cond, keyboard_cond, mouse_cond_memory, keyboard_cond_memory)
                        
                        if x_valid.shape[1] < x.shape[1]:
                            x = torch.cat([x_valid, x[:, valid_len:, :]], dim=1)
                        else:
                            x = x_valid
                else:
                    if world_size > 1:
                        x_full = gather_forward(x.contiguous(), dim=1)
                        x_valid = x_full[:, :valid_len, :]

                        x_valid = self.action_model(x_valid.to(dtype), grid_sizes[0][0], grid_sizes[0][1], grid_sizes[0][2], mouse_cond, keyboard_cond)

                        if x_valid.shape[1] < x_full.shape[1]:
                            x_full = torch.cat([x_valid, x_full[:, valid_len:, :]], dim=1)
                        else:
                            x_full = x_valid

                        x = torch.chunk(x_full.contiguous(), world_size, dim=1)[get_rank() if is_dist else 0].contiguous()
                    else:
                        x_valid = x[:, :valid_len, :]
                        x_valid = self.action_model(x_valid.to(dtype), grid_sizes[0][0], grid_sizes[0][1], grid_sizes[0][2], mouse_cond, keyboard_cond)
                        
                        if x_valid.shape[1] < x.shape[1]:
                            x = torch.cat([x_valid, x[:, valid_len:, :]], dim=1)
                        else:
                            x = x_valid

            y = self.ffn(
                (self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(self.ffn[0].weight.dtype))

            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)

            return x

        if self.use_memory:
            x = cross_attn_ffn(x, context, context_lens, e, mouse_cond=mouse_cond, keyboard_cond=keyboard_cond, mouse_cond_memory=mouse_cond_memory, keyboard_cond_memory=keyboard_cond_memory, fa_version=fa_version)
        else:
            x = cross_attn_ffn(x, context, context_lens, e, mouse_cond=mouse_cond, keyboard_cond=keyboard_cond, fa_version=fa_version)
        return x


class Head(nn.Module):
    """AdaLN-modulated output projection from tokens to packed patch voxels."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        """Init.

        Args:
            dim: The dim.
            out_dim: The out dim.
            patch_size: The patch size.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e, profiler=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            
            # --- Linear Layer Profiling (Head) ---
            if profiler is not None and 'linear_layers' in profiler:
                norm_x = self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
                torch.cuda.synchronize()
                l_start = time.time()
                x = self.head(norm_x)
                torch.cuda.synchronize()
                l_dur = time.time() - l_start
                key = ("Head.head", self.head.in_features, self.head.out_features, "Int8Linear" if hasattr(self.head, "weight_int8") else "Linear")
                profiler['linear_layers'][key] = profiler['linear_layers'].get(key, 0.0) + l_dur
            else:
                x = (
                    self.head(
                        self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
            # -------------------------------------
        return x

class WanModel(nn.Module, NativeConfigMixin):
    r"""Wan 2.2 action-conditioned DiT (per-token timestep, optional INT8 Q)."""

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 action_config = {},
                 use_memory=True,
                 sigma_theta=0.0,
                 use_text_crossattn=True):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_memory = use_memory
        self.sigma_theta = sigma_theta

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        if use_memory:
            self.patch_embedding_wancamctrl = nn.Linear(
                6 * 256 * patch_size[0] * patch_size[1] * patch_size[2], dim)
            self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
            self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps, action_config, _, use_memory) for _ in range(num_layers)
        ])

        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        if use_memory:
            max_seq_len = 2048
            if sigma_theta > 0:
                c = d // 2
                c_t = c - 2 * (c // 3)
                c_h = c // 3
                c_w = c // 3
                rope_epsilon = torch.linspace(-1, 1, num_heads, dtype=torch.float64)
                theta_base = 10000.0
                theta_hat = theta_base * (1 + sigma_theta * rope_epsilon)

                def build_freqs(seq_len, c_part):
                    """Build freqs.

                    Args:
                        seq_len: The seq len.
                        c_part: The c part.
                    """
                    exp = torch.arange(c_part, dtype=torch.float64) / c_part
                    omega = 1.0 / torch.pow(theta_hat.unsqueeze(1), exp.unsqueeze(0))
                    pos = torch.arange(seq_len, dtype=torch.float64)
                    angles = pos.view(1, -1, 1) * omega.unsqueeze(1)
                    return torch.polar(torch.ones_like(angles), angles)

                freqs_t = build_freqs(max_seq_len, c_t)
                freqs_h = build_freqs(max_seq_len, c_h)
                freqs_w = build_freqs(max_seq_len, c_w)
                self.freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=2)
            else:
                self.freqs = torch.cat([
                    rope_params(max_seq_len, d - 4 * (d // 6)),
                    rope_params(max_seq_len, 2 * (d // 6)),
                    rope_params(max_seq_len, 2 * (d // 6))
                ], dim=1)
        else:
            self.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
                               dim=1)

        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        mouse_cond=None,
        keyboard_cond=None,
        x_memory=None,
        timestep_memory=None,
        mouse_cond_memory=None,
        keyboard_cond_memory=None,
        plucker_emb=None,
        memory_latent_idx=None,
        predict_latent_idx=None,
        return_memory=False,
        fa_version=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        memory_length = 0
        if x_memory is not None:
            memory_length = x_memory.shape[2]
            x = torch.cat([x_memory, x], dim=2)
            t = torch.cat([timestep_memory, t], dim=1)
        
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        elif t.dim() == 2 and t.size(1) < seq_len:
            t = torch.cat([
                t, t.new_zeros(t.size(0), seq_len - t.size(1))
            ], dim=1)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            e = e.float()
            e0 = e0.float()
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        context_lens = None
        # plucker embeddings
        if plucker_emb is not None:
            # Accept batched tensor [B, C, F, H, W] or list of [1, C, F, H, W].
            if torch.is_tensor(plucker_emb):
                plucker_items = [u.unsqueeze(0) for u in plucker_emb]
            else:
                plucker_items = [u.unsqueeze(0) if u.dim() == 4 else u for u in plucker_emb]
            plucker_emb = [
                rearrange(
                    i,
                    '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                    c1=self.patch_size[0],
                    c2=self.patch_size[1],
                    c3=self.patch_size[2],
                ) for i in plucker_items
            ]
            plucker_emb = torch.cat(plucker_emb, dim=1)
            if plucker_emb.size(1) < seq_len:
                plucker_emb = torch.cat([
                    plucker_emb, 
                    plucker_emb.new_zeros(plucker_emb.size(0), seq_len - plucker_emb.size(1), plucker_emb.size(2))
                ], dim=1)
            plucker_emb = self.patch_embedding_wancamctrl(plucker_emb)
            plucker_hidden = self.c2ws_hidden_states_layer2(
                torch_F.silu(self.c2ws_hidden_states_layer1(plucker_emb))
            )
            plucker_emb = plucker_emb + plucker_hidden

        if self.use_memory:
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                mouse_cond=mouse_cond,
                keyboard_cond=keyboard_cond,
                plucker_emb=plucker_emb,
                mouse_cond_memory=mouse_cond_memory,
                keyboard_cond_memory=keyboard_cond_memory,
                memory_length=memory_length,
                memory_latent_idx=memory_latent_idx,
                predict_latent_idx=predict_latent_idx,
                fa_version=fa_version,
            )
        else:
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                mouse_cond=mouse_cond,
                keyboard_cond=keyboard_cond,
                fa_version=fa_version,
                )

        for block in self.blocks:
            x = block(x, **kwargs)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)

        if self.use_memory:
            if return_memory:
                return [u[:, :memory_length] for u in x], [u[:, memory_length:] for u in x]
            if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
                return torch.stack([u[:, memory_length:] for u in x])
            return [u[:, memory_length:] for u in x]
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, Int8Linear):
                temp_weight = torch.empty(m.out_features, m.in_features, device=m.weight_int8.device)
                nn.init.xavier_uniform_(temp_weight)
                m.from_float(temp_weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        nn.init.zeros_(self.head.head.weight)

        for m in self.blocks:
            try:
                nn.init.zeros_(m.action_model.proj_mouse.weight)
                if m.action_model.proj_mouse.bias is not None:
                    nn.init.zeros_(m.action_model.proj_mouse.bias)
                nn.init.zeros_(m.action_model.proj_keyboard.weight)
                if m.action_model.proj_keyboard.bias is not None:
                    nn.init.zeros_(m.action_model.proj_keyboard.bias)

            except:
                pass
        if self.use_memory:
            nn.init.xavier_uniform_(self.patch_embedding_wancamctrl.weight)
            nn.init.zeros_(self.patch_embedding_wancamctrl.bias)
            nn.init.xavier_uniform_(self.c2ws_hidden_states_layer1.weight)
            nn.init.zeros_(self.c2ws_hidden_states_layer1.bias)
            nn.init.xavier_uniform_(self.c2ws_hidden_states_layer2.weight)
            nn.init.zeros_(self.c2ws_hidden_states_layer2.bias)

            for block in self.blocks:
                nn.init.xavier_uniform_(block.cam_injector_layer1.weight)
                nn.init.zeros_(block.cam_injector_layer1.bias)
                nn.init.xavier_uniform_(block.cam_injector_layer2.weight)
                nn.init.zeros_(block.cam_injector_layer2.bias)
                nn.init.zeros_(block.cam_scale_layer.weight)
                nn.init.zeros_(block.cam_scale_layer.bias)
                nn.init.zeros_(block.cam_shift_layer.weight)
                nn.init.zeros_(block.cam_shift_layer.bias)
