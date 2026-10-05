"""DiffSynth-shaped Wan VACE factory (hint stack, Civitai key filter).

This is not a standalone Wan version.  Callers pass the host DiT block
class and a state-dict hasher; the factory returns checkpoint-compatible
``VaceWanAttentionBlock`` / ``VaceWanModel`` / ``VaceWanModelDictConverter``
with ``__module__`` rewritten so loaders see the historical DiffSynth path.

Compared with :mod:`.vace`:
- Condition tokens are a stacked tensor: previous hints plus the live
  condition stream (``[..., c_skip, c]``).
- ``forward`` returns only the hint sequence (everything except the last
  stacked plane).  The host DiT injects those hints.
- Optional gradient-checkpoint / CPU-offload wrappers match DiffSynth.

No action, camera, causal, linear-attention, or TeaCache logic lives here.
"""

import torch


def build_vace_wan_classes(dit_block_cls, hash_state_dict_keys, *, module_name: str):
    """Build DiffSynth-layout VACE classes bound to ``dit_block_cls``.

    Args:
        dit_block_cls: Host Wan DiT block used as the VACE condition block.
        hash_state_dict_keys: Fingerprint helper used to recover 14B configs.
        module_name: Historical module path written onto the generated classes.

    Returns:
        ``(VaceWanAttentionBlock, VaceWanModel, VaceWanModelDictConverter)``.
    """
    class VaceWanAttentionBlock(dit_block_cls):
        """VACE condition block that stacks skip hints for later main-graph injection."""

        def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
            """Build QKV/FFN from the host block, plus VACE before/after projections.

            Args:
                has_image_input: Whether the host block also has CLIP image KV.
                dim: Transformer hidden size.
                num_heads: Attention heads (must divide ``dim``).
                ffn_dim: Feed-forward inner width.
                eps: RMS / LayerNorm epsilon.
                block_id: Official VACE layer id; ``0`` owns ``before_proj``.
            """
            super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
            self.block_id = block_id
            if block_id == 0:
                self.before_proj = torch.nn.Linear(self.dim, self.dim)
            self.after_proj = torch.nn.Linear(self.dim, self.dim)

        def forward(self, c, x, context, t_mod, freqs):
            """Advance the stacked condition tensor and append a new hint plane.

            Args:
                c: Condition tokens.  Layer 0 is a single stream; later layers
                    are ``stack([hint_0, ..., hint_{n-1}, live_c])``.
                x: Main-graph hidden states mixed in only at layer 0.
                context: Text (and optional image) cross-attention context.
                t_mod: Timestep AdaLN modulation, shape ``[B, 6, C]``.
                freqs: 3D RoPE frequencies for the patch grid.
            """
            if self.block_id == 0:
                c = self.before_proj(c) + x
                all_c = []
            else:
                all_c = list(torch.unbind(c))
                c = all_c.pop(-1)
            c = super().forward(c, context, t_mod, freqs)
            c_skip = self.after_proj(c)
            all_c += [c_skip, c]
            c = torch.stack(all_c)
            return c

    class VaceWanModel(torch.nn.Module):
        """Standalone VACE condition trunk that emits hints for a host Wan DiT."""

        def __init__(
            self,
            vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
            vace_in_dim=96,
            patch_size=(1, 2, 2),
            has_image_input=False,
            dim=1536,
            num_heads=12,
            ffn_dim=8960,
            eps=1e-6,
        ):
            """Build the VACE patch embed and selected condition blocks.

            Args:
                vace_layers: Official layer ids that emit hints (must include 0).
                vace_in_dim: Channels of the VACE context video (often 96).
                patch_size: Same 3D patch as the host Wan DiT.
                has_image_input: Forwarded to each condition block.
                dim: Hidden size; must match the host DiT.
                num_heads: Attention heads; must match the host DiT.
                ffn_dim: Feed-forward inner width.
                eps: Normalization epsilon.
            """
            super().__init__()
            self.vace_layers = vace_layers
            self.vace_in_dim = vace_in_dim
            self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}
            self.vace_blocks = torch.nn.ModuleList(
                [
                    VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
                    for i in self.vace_layers
                ]
            )
            self.vace_patch_embedding = torch.nn.Conv3d(
                vace_in_dim, dim, kernel_size=patch_size, stride=patch_size
            )

        def forward(
            self,
            x,
            vace_context,
            context,
            t_mod,
            freqs,
            use_gradient_checkpointing: bool = False,
            use_gradient_checkpointing_offload: bool = False,
        ):
            """Patchify VACE context, run the condition trunk, and return hint tensors.

            Per-sample context videos are padded to the host token length so
            hints line up with the main packed sequence.

            Args:
                x: Host hidden states used only to size padding (``[B, L, C]``).
                vace_context: Per-sample VACE videos (list of ``[C, F, H, W]``).
                context: Shared text / image cross-attention tokens.
                t_mod: Host timestep modulation, ``[B, 6, C]``.
                freqs: Host 3D RoPE frequencies.
                use_gradient_checkpointing: Recompute condition blocks in backward.
                use_gradient_checkpointing_offload: Same, but stash activations on CPU.
            """
            c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
            c = [u.flatten(2).transpose(1, 2) for u in c]
            c = torch.cat(
                [
                    torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))], dim=1)
                    for u in c
                ]
            )

            def create_custom_forward(module):
                """Wrap ``module`` so ``torch.utils.checkpoint`` can call it."""

                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            for block in self.vace_blocks:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        c = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            c,
                            x,
                            context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c,
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
                else:
                    c = block(c, x, context, t_mod, freqs)
            hints = torch.unbind(c)[:-1]
            return hints

        @staticmethod
        def state_dict_converter():
            """Return the Civitai / DiffSynth key filter for this VACE trunk."""
            return VaceWanModelDictConverter()

    class VaceWanModelDictConverter:
        """Keep only ``vace*`` tensors and recover the 14B VACE layout from a hash."""

        def from_civitai(self, state_dict):
            """Filter a full checkpoint down to VACE weights and optional config.

            Args:
                state_dict: Combined Wan + VACE (or Civitai) state dict.

            Returns:
                ``(vace_only_state_dict, config)``.  Config is filled when the
                filtered keys match the known 14B fingerprint.
            """
            state_dict_ = {name: param for name, param in state_dict.items() if name.startswith("vace")}
            if hash_state_dict_keys(state_dict_) == "3b2726384e4f64837bdf216eea3f310d":
                config = {
                    "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35),
                    "vace_in_dim": 96,
                    "patch_size": (1, 2, 2),
                    "has_image_input": False,
                    "dim": 5120,
                    "num_heads": 40,
                    "ffn_dim": 13824,
                    "eps": 1e-06,
                }
            else:
                config = {}
            return state_dict_, config

    for cls in (VaceWanAttentionBlock, VaceWanModel, VaceWanModelDictConverter):
        cls.__module__ = module_name

    return VaceWanAttentionBlock, VaceWanModel, VaceWanModelDictConverter


__all__ = ["build_vace_wan_classes"]
