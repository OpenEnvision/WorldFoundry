import itertools
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import CausalConv3d, RMS_norm, Upsample
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.action_21 import (
    AttentionBlock,
    ResidualBlock,
)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.cache_t = 2

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006 - Keeps the upstream default object and signature.
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -self.cache_t :, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                x = self.time_conv(x) if feat_cache[idx] == "Rep" else self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":  # noqa: SIM102 - Branches carry per-case comments.
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[: c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2 :, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class VAEDecoderWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = VAEDecoder3d()
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.z_dim = 16
        self.conv2 = CausalConv3d(self.z_dim, self.z_dim, 1)
        self._scale_cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
        self._cuda_graph_enabled: bool = False
        self._steady_graph_runner: Any = None
        self._steady_graph_key: tuple[Any, ...] | None = None
        self._steady_compiled_fn: Any = None
        self._steady_compile_mode: str | None = None

    def enable_steady_compile(self, *, mode: str = "default") -> None:
        """Compile only the steady-state tensor-cached decode path without chunk-0 ``None``/``"Rep"`` guards."""
        import torch._dynamo as _dynamo

        if _dynamo.config.cache_size_limit < 64:
            _dynamo.config.cache_size_limit = 64

        def _steady_decode(z_in: torch.Tensor, *cache_in: torch.Tensor):
            out_t, next_cache = self._decode_eager(z_in, list(cache_in))
            return (out_t, *next_cache)

        self._steady_compiled_fn = torch.compile(_steady_decode, mode=mode, dynamic=False)
        self._steady_compile_mode = mode

    def enable_cuda_graph(self, *, warmup: int = 1) -> None:
        """Capture and replay steady-state causal VAE decode chunks via CUDA Graph."""
        self._cuda_graph_enabled = True
        self._cuda_graph_warmup = int(warmup)

    def disable_cuda_graph(self) -> None:
        """Disable VAE CUDA Graph capture and release cached graphs."""
        self._cuda_graph_enabled = False
        self.invalidate_cuda_graphs()

    def invalidate_cuda_graphs(self) -> None:
        if self._steady_graph_runner is not None:
            self._steady_graph_runner.invalidate()
        self._steady_graph_runner = None
        self._steady_graph_key = None

    def graph_report(self) -> dict[str, Any] | None:
        if not self._cuda_graph_enabled and self._steady_graph_runner is None:
            return None
        if self._steady_graph_runner is None:
            return {"enabled": self._cuda_graph_enabled, "capture": 0, "replay": 0, "eager": 0}
        return self._steady_graph_runner.report()

    def _get_scale_tensors(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        cached = self._scale_cache.get(key)
        if cached is None:
            mean_t = self.mean.to(device=device, dtype=dtype).view(1, self.z_dim, 1, 1, 1)
            std_t = self.std.to(device=device, dtype=dtype).view(1, self.z_dim, 1, 1, 1)
            cached = (mean_t, std_t)
            self._scale_cache[key] = cached
        return cached

    def _decode_eager(self, z: torch.Tensor, feat_cache: list[torch.Tensor | None]):
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        z = z.permute(0, 2, 1, 3, 4)
        mean_t, std_t = self._get_scale_tensors(z.device, z.dtype)
        z = z * std_t + mean_t
        iter_ = z.shape[2]
        x = self.conv2(z)
        outs: list[torch.Tensor] = []
        for i in range(iter_):
            out_i, feat_cache = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=feat_cache)
            outs.append(out_i)
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        out = out.float().clamp_(-1, 1)
        out = out.permute(0, 2, 1, 3, 4)
        return out, feat_cache

    def forward(self, z: torch.Tensor, *feat_cache: list[torch.Tensor]):
        cache_list = list(feat_cache)
        all_tensors = len(cache_list) > 0 and all(isinstance(item, torch.Tensor) for item in cache_list)
        if self._steady_compiled_fn is not None and z.is_cuda and not self.training and all_tensors:
            packed = self._steady_compiled_fn(z, *cache_list)
            return packed[0], list(packed[1:])
        if self._cuda_graph_enabled and z.is_cuda and not self.training and all_tensors:
            z_contig = z.contiguous()
            cache_contig = [t.contiguous() for t in cache_list]
            sig = (tuple(z_contig.shape), z_contig.dtype, z_contig.device, tuple(tuple(t.shape) for t in cache_contig))
            if self._steady_graph_runner is None or self._steady_graph_key != sig:
                if self._steady_graph_runner is not None:
                    self._steady_graph_runner.invalidate()
                from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

                def _steady_decode(z_in: torch.Tensor, *cache_in: torch.Tensor):
                    out_t, next_cache = self._decode_eager(z_in, list(cache_in))
                    return (out_t, *next_cache)

                self._steady_graph_runner = InferenceCUDAGraphRunner(
                    _steady_decode,
                    warmup=getattr(self, "_cuda_graph_warmup", 1),
                    max_graphs=4,
                    clone_outputs=True,
                    extra_key=f"mg2_vae_steady:{sig[:3]}",
                )
                self._steady_graph_key = sig
            packed = self._steady_graph_runner(z_contig, *cache_contig)
            return packed[0], list(packed[1:])
        return self._decode_eager(z, cache_list)


class VAEDecoder3d(nn.Module):
    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],  # noqa: B006 - Keeps the upstream default object and signature.
        num_res_blocks=2,
        attn_scales=[],  # noqa: B006 - Keeps the upstream default object and signature.
        temperal_upsample=[True, True, False],  # noqa: B006 - Keeps the upstream default object and signature.
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.cache_t = 2
        self.decoder_conv_num = 32

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout)
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x: torch.Tensor, feat_cache: list[torch.Tensor]):
        feat_idx = [0]

        # conv1
        idx = feat_idx[0]
        cache_x = x[:, :, -self.cache_t :, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -self.cache_t :, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x, feat_cache
