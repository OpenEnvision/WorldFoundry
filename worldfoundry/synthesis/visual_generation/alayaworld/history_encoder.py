"""Inference-only temporal history encoder used by AlayaWorld."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy


class CausalConv3d(nn.Conv3d):
    """Conv3d with left-only temporal padding and symmetric spatial padding."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._causal_padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(value, self._causal_padding))


class _Conv3dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1, 1)) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=(1, 1, 1),
        )
        self.act = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(value))


class _SelfAttention3D(nn.Module):
    """Spatially bidirectional, frame-causal attention at compressed scale."""

    def __init__(self, dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = value.shape
        spatial = height * width
        sequence = value.permute(0, 2, 3, 4, 1).reshape(batch, frames * spatial, channels)
        residual = sequence
        qkv = self.qkv(self.norm(sequence)).view(
            batch,
            frames * spatial,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        frame_ids = torch.arange(frames * spatial, device=value.device) // spatial
        allowed = frame_ids[None] <= frame_ids[:, None]
        mask = torch.zeros(frames * spatial, frames * spatial, device=value.device, dtype=q.dtype)
        mask.masked_fill_(~allowed, torch.finfo(q.dtype).min)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(batch, frames * spatial, channels)
        output = residual + self.proj(attended)
        return output.view(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()


class AlayaHistoryEncoder(nn.Module):
    """Compress a sliding LTX latent history into transformer-width tokens."""

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 4096,
        compress_t: int = 1,
        compress_h: int = 2,
        compress_w: int = 2,
        lr_compress_t: int = 1,
        lr_compress_h: int = 2,
        lr_compress_w: int = 2,
        gate_init: float = 0.5,
        use_self_attn: bool = True,
        use_lr_branch: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.compress_t = int(compress_t)
        self.compress_h = int(compress_h)
        self.compress_w = int(compress_w)
        self.lr_compress_t = int(lr_compress_t)
        self.lr_compress_h = int(lr_compress_h)
        self.lr_compress_w = int(lr_compress_w)
        self.use_self_attn = bool(use_self_attn)
        self.use_lr_branch = bool(use_lr_branch)

        for name in (
            "compress_t",
            "compress_h",
            "compress_w",
            "lr_compress_t",
            "lr_compress_h",
            "lr_compress_w",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        self.hr_stage1 = _Conv3dBlock(self.in_channels, 64)
        self.hr_stage2 = _Conv3dBlock(64, 128, stride=(self.compress_t, 1, 1))
        self.hr_stage3 = _Conv3dBlock(128, 256, stride=(1, self.compress_h, self.compress_w))
        self.hr_stage4 = _Conv3dBlock(256, 256)
        self.hr_stage5 = _Conv3dBlock(256, 512)
        self.hr_stage6 = _Conv3dBlock(512, 512)
        if self.use_self_attn:
            self.hr_attn = _SelfAttention3D(dim=512, num_heads=8)
        self.hr_proj = CausalConv3d(512, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.output_gate = nn.Parameter(torch.full((1,), float(gate_init)))

        if self.use_lr_branch:
            self.register_buffer(
                "lr_proj_weight",
                torch.zeros(self.out_channels, self.in_channels),
                persistent=False,
            )
            self.register_buffer("lr_proj_bias", torch.zeros(self.out_channels), persistent=False)
            self._lr_proj_initialized = False

    @torch.no_grad()
    def setup_lr_proj_from_patchify(self, patchify_proj: nn.Linear) -> None:
        """Snapshot the frozen canonical LTX patch projection for the LR path."""

        if not self.use_lr_branch:
            return
        if tuple(patchify_proj.weight.shape) != tuple(self.lr_proj_weight.shape):
            raise ValueError(
                f"patchify projection shape {tuple(patchify_proj.weight.shape)} does not match "
                f"history LR projection {tuple(self.lr_proj_weight.shape)}"
            )
        self.lr_proj_weight.copy_(patchify_proj.weight.detach().to(self.lr_proj_weight))
        if patchify_proj.bias is None:
            self.lr_proj_bias.zero_()
        else:
            self.lr_proj_bias.copy_(patchify_proj.bias.detach().to(self.lr_proj_bias))
        self._lr_proj_initialized = True

    @staticmethod
    def _indices(
        batch: int,
        frames: int,
        height: int,
        width: int,
        scales: tuple[int, int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        temporal = torch.arange(frames, device=device, dtype=torch.float32) * scales[0]
        vertical = torch.arange(height, device=device, dtype=torch.float32) * scales[1]
        horizontal = torch.arange(width, device=device, dtype=torch.float32) * scales[2]
        starts = torch.meshgrid(temporal, vertical, horizontal, indexing="ij")
        starts = torch.stack(tuple(value.flatten() for value in starts), dim=0)
        scale = torch.tensor(scales, device=device, dtype=torch.float32)[:, None]
        bounds = torch.stack((starts, starts + scale), dim=-1)
        return bounds.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 5 or latent.shape[1] != self.in_channels:
            raise ValueError(f"history latent must be [B,{self.in_channels},T,H,W], got {tuple(latent.shape)}")
        batch, _, frames, height, width = latent.shape
        output = self.hr_stage6(self.hr_stage5(self.hr_stage4(self.hr_stage3(self.hr_stage2(self.hr_stage1(latent))))))
        if self.use_self_attn:
            output = self.hr_attn(output)
        output = self.hr_proj(output)
        _, _, out_frames, out_height, out_width = output.shape
        hr_tokens = output.permute(0, 2, 3, 4, 1).reshape(batch, -1, self.out_channels)
        indices = self._indices(
            batch,
            out_frames,
            out_height,
            out_width,
            (self.compress_t, self.compress_h, self.compress_w),
            device=latent.device,
        )
        if not self.use_lr_branch:
            return hr_tokens * self.output_gate, indices
        if not self._lr_proj_initialized:
            raise RuntimeError("call setup_lr_proj_from_patchify() before using the history LR branch")

        target_size = (
            (frames + self.lr_compress_t - 1) // self.lr_compress_t,
            (height + self.lr_compress_h - 1) // self.lr_compress_h,
            (width + self.lr_compress_w - 1) // self.lr_compress_w,
        )
        low_resolution = F.interpolate(
            latent.float(),
            size=target_size,
            mode="trilinear",
            align_corners=False,
        ).to(latent.dtype)
        lr_tokens = low_resolution.permute(0, 2, 3, 4, 1).reshape(batch, -1, self.in_channels)
        lr_tokens = F.linear(
            lr_tokens.to(self.lr_proj_weight.dtype),
            self.lr_proj_weight,
            self.lr_proj_bias,
        ).to(latent.dtype)
        if lr_tokens.shape != hr_tokens.shape:
            raise RuntimeError(
                f"history HR/LR token shapes differ: {tuple(hr_tokens.shape)} vs {tuple(lr_tokens.shape)}"
            )
        return (hr_tokens + lr_tokens) * self.output_gate, indices

    def output_token_count(self, frames: int, height: int, width: int) -> int:
        def ceil_div(value: int, divisor: int) -> int:
            return (value + divisor - 1) // divisor

        return ceil_div(frames, self.compress_t) * ceil_div(height, self.compress_h) * ceil_div(width, self.compress_w)


def convert_alaya_history_state_dict(state_dict):
    """Select the history encoder from Alaya's merged checkpoint."""

    prefix = "history_encoder."
    return {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}


def load_alaya_history_encoder(
    checkpoint: CheckpointSpec,
    patchify_proj: nn.Linear,
    *,
    policy: RuntimePolicy,
    compress_t: int = 1,
    compress_h: int = 2,
    compress_w: int = 2,
    lr_compress_t: int = 1,
    lr_compress_h: int = 2,
    lr_compress_w: int = 2,
    gate_init: float = 0.5,
    use_self_attn: bool = True,
    use_lr_branch: bool = True,
) -> AlayaHistoryEncoder:
    """Load only the history subset from Alaya's merged safetensors file."""

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=AlayaHistoryEncoder,
            config={
                "in_channels": patchify_proj.in_features,
                "out_channels": patchify_proj.out_features,
                "compress_t": compress_t,
                "compress_h": compress_h,
                "compress_w": compress_w,
                "lr_compress_t": lr_compress_t,
                "lr_compress_h": lr_compress_h,
                "lr_compress_w": lr_compress_w,
                "gate_init": gate_init,
                "use_self_attn": use_self_attn,
                "use_lr_branch": use_lr_branch,
            },
            state_dict_converter=convert_alaya_history_state_dict,
        ),
        checkpoint,
        policy,
    )
    if not isinstance(model, AlayaHistoryEncoder):
        raise TypeError(f"expected AlayaHistoryEncoder, got {type(model).__name__}")
    model.setup_lr_proj_from_patchify(patchify_proj)
    return model.eval()


__all__ = [
    "AlayaHistoryEncoder",
    "CausalConv3d",
    "convert_alaya_history_state_dict",
    "load_alaya_history_encoder",
]
