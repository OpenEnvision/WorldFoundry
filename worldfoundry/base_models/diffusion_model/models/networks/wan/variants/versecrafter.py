"""VerseCrafter GeoAda Wan (VACE-like geometry hint trunk + TeaCache).

Built on VideoX-Fun blocks.  A parallel GeoAda stack emits hints that
selected main blocks add (same stack-of-skips pattern as VACE).
``process_teacache_skip_logic`` can skip the main stack.

Adds geometry/VACE-like hints and TeaCache.  No action or linear attn.
"""

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from worldfoundry.core.model_loading.model_configuration import register_to_config

from .video_x_fun.transformer import (
    WanAttentionBlock,
    WanTransformer3DModel,
    prepare_sequence_tensors,
    process_teacache_skip_logic,
    sinusoidal_embedding_1d,
)
from worldfoundry.base_models.diffusion_model.optimizations.wan.cfg_skip import (
    cfg_skip,
)


def _process_blocks_forward(blocks, x, kwargs):
    """Run VerseCrafter's GeoAda transformer blocks during inference."""
    for block in blocks:
        x = block(x, **kwargs)
    return x


class VerseCrafterWanAttentionBlock(WanAttentionBlock):
    """GeoAda condition block that stacks skip hints (VACE-style)."""

    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        """Advance the stacked GeoAda tensor; layer 0 mixes in the main stream."""
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)

        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class BaseWanAttentionBlock(WanAttentionBlock):
    """Main Fun block with an optional GeoAda hint injection point."""

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        """Run the Fun block, then add the mapped GeoAda hint when selected."""
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x


class VerseCrafterWanTransformer3DModel(WanTransformer3DModel):
    """VideoX-Fun Wan plus a GeoAda hint trunk and TeaCache skip hook."""

    @register_to_config
    def __init__(self,
                 geoada_layers=None,
                 geoada_in_dim=None,
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
                 eps=1e-6):
        model_type = "t2v"   # TODO: Hard code for both preview and official versions.
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        self.geoada_layers = [i for i in range(0, self.num_layers, 2)] if geoada_layers is None else geoada_layers
        self.geoada_in_dim = self.in_dim if geoada_in_dim is None else geoada_in_dim

        assert 0 in self.geoada_layers
        self.geoada_layers_mapping = {i: n for n, i in enumerate(self.geoada_layers)}

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim,
                                    self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps,
                                  block_id=self.geoada_layers_mapping[i] if i in self.geoada_layers else None)
            for i in range(self.num_layers)
        ])

        # geoada blocks
        self.geoada_blocks = nn.ModuleList([
            VerseCrafterWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim,
                                        self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in self.geoada_layers
        ])

        # geoada patch embeddings
        self.geoada_patch_embedding = nn.Conv3d(
            self.geoada_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward_geoada(
        self,
        x,
        geoada_context,
        seq_len,
        kwargs
    ):
        """Patchify GeoAda context, run the hint trunk, return skip tensors."""
        # embeddings
        c = [self.geoada_patch_embedding(u.unsqueeze(0)) for u in geoada_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        # Context Parallel
        if self.sp_world_size > 1:
            c = torch.chunk(c, self.sp_world_size, dim=1)[self.sp_world_rank]

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        for block in self.geoada_blocks:
            c = block(c, **new_kwargs)
        hints = torch.unbind(c)[:-1]
        return hints

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        geoada_context,
        context,
        seq_len,
        geoada_context_scale=1.0,
        clip_fea=None,
        y=None,
        cond_flag=True
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
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        dtype = x.dtype
        device = self.patch_embedding.weight.device
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        x, seq_lens = prepare_sequence_tensors(x, seq_len, self.sp_world_size)

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

            e0 = e0.to(dtype)
            e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dtype=dtype,
            t=t)
        hints = self.forward_geoada(x, geoada_context, seq_len, kwargs)

        kwargs['hints'] = hints
        kwargs['context_scale'] = geoada_context_scale

        # TeaCache
        if self.teacache is not None:
            self.should_calc = process_teacache_skip_logic(self.teacache, e0, t, cond_flag)

        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = (
                    self.teacache.previous_residual_cond
                    if cond_flag
                    else self.teacache.previous_residual_uncond
                )
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                x = _process_blocks_forward(self.blocks, x, kwargs)

                # Update TeaCache residual after block processing
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            x = _process_blocks_forward(self.blocks, x, kwargs)

        # head
        x = self.head(x, e)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x
