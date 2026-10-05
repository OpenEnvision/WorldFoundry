"""DualCamCtrl (Wan 2.x): camera / RGB / depth ControlNet on native Wan.

:class:`WanControlNet` reuses the packed :class:`WanModel` stem and adds
a parallel control stack.  Selected host layers receive fused residuals
from camera, RGB, and/or depth inject lists.  Fusion is either gated 3D
conv (:class:`ResidualFusion3D_Gating`) or a stacked frame-gated block
(:class:`ResidualFusion3D_Stack`).

Adds camera (and optional RGB/depth) control.  No VACE, action encoder,
causal attention, or linear-attention processors.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from worldfoundry.core.nn import sinusoidal_embedding_1d

from ..model import DiTBlock, WanModel
from .dualcamctrl_fusion import ResidualFusion3D_Gating
from .dualcamctrl_fusion_stack import ResidualFusion3D_Stack


class WanControlNet(WanModel):
    """Packed Wan DiT plus a parallel control stack fused at selected layers."""

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        control_block_index,
        camera_lora_rank: int = 64,
        cover_base: int = 5,
        pre_camera_control: bool = False,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        use_gate_3d_linear: bool = False,
        update: bool = False,
        camera_inject_blocks=None,
        rgb_inject_blocks=None,
        depth_inject_blocks=None,
        **kwargs,
    ):
        super().__init__(
            dim,
            in_dim,
            ffn_dim,
            out_dim,
            text_dim,
            freq_dim,
            eps,
            patch_size,
            num_heads,
            num_layers,
            has_image_input,
            has_image_pos_emb,
            has_ref_conv,
            add_control_adapter,
            in_dim_control_adapter,
        )
        # This model is initialized with extra kwargs: {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
        camera_inject_blocks = list(camera_inject_blocks or ())
        rgb_inject_blocks = list(rgb_inject_blocks or ())
        depth_inject_blocks = list(depth_inject_blocks or ())
        self.use_gate_3d_linear = use_gate_3d_linear
        self.rgb_inject_blocks = rgb_inject_blocks
        self.depth_inject_blocks = depth_inject_blocks
        if self.control_adapter is not None:
            print(f"Adding camera into input using addition.")

        self.control_block_index = control_block_index
        num_control_block = len(control_block_index)
        self.num_control_block = num_control_block
        num_rgb_inject_blocks = len(rgb_inject_blocks)
        self.num_rgb_inject_blocks = num_rgb_inject_blocks
        num_depth_inject_blocks = len(depth_inject_blocks)
        self.num_depth_inject_blocks = num_depth_inject_blocks
        if num_control_block >= 0:
            print(
                f"Using {num_control_block} control blocks, for layers {self.control_block_index}"
            )
            # Control blocks
            self.control_blocks = nn.ModuleList(
                [
                    DiTBlock(
                        has_image_input=has_image_input,
                        dim=dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        eps=eps,
                    )
                    for _ in range(num_control_block)
                ]
            )

            if use_gate_3d_linear:
                if update:
                    self.control_zero_inits = nn.ModuleList(
                        [
                            ResidualFusion3D_Stack(C=dim)
                            for _ in range(num_depth_inject_blocks)
                        ]
                    )
                    self.rgb_zero_inits = nn.ModuleList(
                        [
                            ResidualFusion3D_Stack(C=dim)
                            for _ in range(num_rgb_inject_blocks)
                        ]
                    )
                else:
                    self.control_zero_inits = nn.ModuleList(
                        [
                            ResidualFusion3D_Gating(C=dim)
                            for _ in range(num_depth_inject_blocks)
                        ]
                    )
                    self.rgb_zero_inits = nn.ModuleList(
                        [
                            ResidualFusion3D_Gating(C=dim)
                            for _ in range(num_rgb_inject_blocks)
                        ]
                    )

            else:
                self.control_zero_inits = nn.ModuleList(
                    [
                        nn.Linear(dim, dim, bias=False)
                        for _ in range(num_depth_inject_blocks)
                    ]
                )
                self.rgb_zero_inits = nn.ModuleList(
                    [
                        nn.Linear(dim, dim, bias=False)
                        for _ in range(num_rgb_inject_blocks)
                    ]
                )

                for idx in range(num_depth_inject_blocks):
                    self.control_zero_inits[idx].weight.data.zero_()
                    if self.control_zero_inits[idx].bias is not None:
                        self.control_zero_inits[idx].bias.data.zero_()
                for idx in range(num_rgb_inject_blocks):
                    self.rgb_zero_inits[idx].weight.data.zero_()
                    if self.rgb_zero_inits[idx].bias is not None:
                        self.rgb_zero_inits[idx].bias.data.zero_()

        else:
            print(
                f"No control blocks used, control_block_index is empty: {control_block_index}"
            )
            self.control_blocks = None

    def copy_weights_from_main_branch(self):
        """Copy host DiT block weights into the matching control blocks (init helper)."""
        # print(f"Start copying weights from main branch to control blocks.")
        if self.control_blocks is None:
            return
        miss_keys = []
        unexpected_keys = []
        for idx, block in enumerate(self.blocks):
            if idx in self.control_block_index:
                c_block_id = self.control_block_index.index(idx)
                # print(f"copy main block {idx} to control block {c_block_id}")
                state = self.control_blocks[c_block_id].load_state_dict(
                    block.state_dict(), strict=True
                )
        return

    # def zero_init_linear(self):
    #     if self.control_blocks is not None:
    #         for idx in range(self.num_depth_inject_blocks):
    #             # print(f"Zero init control block {idx}")
    #             self.control_zero_inits[idx].weight.data.zero_()
    #             if self.control_zero_inits[idx].bias is not None:
    #                 self.control_zero_inits[idx].bias.data.zero_()

    def patchify(
        self, x: torch.Tensor, control_camera_latents_input: torch.Tensor = None
    ):
        """Patchify latents and optionally add the camera-adapter residual."""
        x = self.patch_embedding(x)

        if (
            self.control_adapter is not None
            and control_camera_latents_input is not None
        ):
            # print(
            #     f"Camera control input shape: {control_camera_latents_input.shape}")

            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = torch.stack(x, dim=0)
        grid_size = x.shape[2:]
        # print(f"Grid size after patch embedding: {grid_size}")

        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        control_video: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        camera_pose_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Denoise ``x`` while running the control video through fused inject layers."""
        if control_video is None:
            control_video = torch.zeros_like(x)
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            control_latents = torch.cat([control_video, y.clone()], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)
        control_latents, _ = self.patchify(control_latents, None)

        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for idx, block in enumerate(self.blocks):
            # Control branch
            if idx < self.num_control_block:
                control_latents = control_latents + x
                if self.training and use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            control_latents = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(self.control_blocks[idx]),
                                control_latents,
                                context,
                                t_mod,
                                freqs,
                                use_reentrant=False,
                            )
                    else:
                        print(f"using gradient checkpointing for control block {idx}")
                        control_video = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(self.control_blocks[idx]),
                            control_latents,
                            context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                else:
                    control_latents = self.control_blocks[idx](
                        control_latents, context, t_mod, freqs
                    )

                x = x + self.control_zero_inits[idx](control_video)

            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            camera_pose_embedding=_pose_embeddings,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        camera_pose_embedding=_pose_embeddings,
                        use_reentrant=False,
                    )
            else:
                x = block(
                    x, context, t_mod, freqs, camera_pose_embedding=_pose_embeddings
                )

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        from ....denoisers.wan import WanModelStateDictConverter

        return WanModelStateDictConverter()


__all__ = ["WanControlNet"]
