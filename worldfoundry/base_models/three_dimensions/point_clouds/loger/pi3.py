"""Module for base_models -> three_dimensions -> point_clouds -> loger -> pi3.py functionality."""

import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from typing import Optional, Union, List

from worldfoundry.core.nn.blocks.layers import Mlp
from .utils.geometry import homogenize_points, robust_scale_estimation
from worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.layers.transformer_head import (
    ContextOnlyTransformerDecoder,
    LinearPts3d,
    TransformerDecoder,
)
from .layers.camera_head import CameraHead
from worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.layers.conv_head import ConvHead
from worldfoundry.base_models.perception_core.general_perception.dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
from .ttt import FastWeightGluMLPMultihead, TTTOperator

class Pi3(nn.Module, PyTorchModelHubMixin):
    """Pi implementation."""
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            ttt_insert_after: Union[int, List[int]] = None,
            ttt_head_dim: int = 512,
            ttt_inter_multi: int = 2,
            num_muon_update_steps: int = 5,
            use_momentum: bool = False,
            ttt_update_steps: int = 1,
            conf: bool = True,
            attn_insert_after: Union[int, List[int], None] = None,
            ttt_pre_norm: bool = False,
            pi3x: bool = False,
            pi3x_metric: bool = True,
        ):
        """Init.

        Args:
            pos_type: The pos type.
            decoder_size: The decoder size.
            ttt_insert_after: The ttt insert after.
            ttt_head_dim: The ttt head dim.
            ttt_inter_multi: The ttt inter multi.
            num_muon_update_steps: The num muon update steps.
            use_momentum: The use momentum.
            ttt_update_steps: The ttt update steps.
            conf: The conf.
            attn_insert_after: The attn insert after.
            ttt_pre_norm: The ttt pre norm.
            pi3x: The pi3x.
            pi3x_metric: The pi3x metric.
        """
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        def _normalize_insert_positions(value: Union[int, List[int], None]) -> List[int]:
            """Helper function to normalize insert positions.

            Args:
                value: The value.

            Returns:
                The return value.
            """
            if isinstance(value, (int, float)):
                return [int(value)]
            if isinstance(value, (list, tuple)):
                return [int(x) for x in value]
            return []

        parsed_ttt_insert_after = _normalize_insert_positions(ttt_insert_after)
        parsed_attn_insert_after = _normalize_insert_positions(attn_insert_after)

        if not parsed_attn_insert_after:
            parsed_attn_insert_after = parsed_ttt_insert_after.copy()

        self.ttt_insert_after = parsed_ttt_insert_after
        self.attn_insert_after = parsed_attn_insert_after
        self.detach_swa_history = False
        self.initialize_swa_from_global = True
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        self.num_muon_update_steps = int(num_muon_update_steps)
        self.num_pe_tokens = 3
        self.use_momentum = use_momentum
        self.ttt_update_steps = int(ttt_update_steps)
        self.use_conf = bool(conf)
        self.ttt_pre_norm = ttt_pre_norm
        self.pi3x = pi3x
        self.pi3x_metric = pi3x_metric
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        for i in range(3):
            pe_token = nn.Parameter(torch.randn(1, 1, 1, self.dec_embed_dim))
            nn.init.normal_(pe_token, std=1e-6)
            self.register_parameter(f'pe_token_{i}', pe_token)
        self.patch_start_idx += 1

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        if self.pi3x:
            self.point_head = ConvHead(
                num_features=4, 
                dim_in=1024,
                projects=nn.Identity(),
                dim_out=[2, 1], 
                dim_proj=1024,
                dim_upsample=[256, 128, 64],
                dim_times_res_block_hidden=2,
                num_res_blocks=2,
                res_block_norm='group_norm',
                last_res_blocks=0,
                last_conv_channels=32,
                last_conv_size=1,
                using_uv=True
            )
        else:
            self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        if self.use_conf:
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)
        else:
            self.conf_decoder = None
            self.conf_head = None

        # ----------------------
        #     Metric Decoder
        # ----------------------
        if self.pi3x and self.pi3x_metric:
            self.metric_token = nn.Parameter(torch.randn(1, 1, 2*self.dec_embed_dim))
            self.metric_decoder = ContextOnlyTransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=512,
                dec_num_heads=8,                # 8
                out_dim=512,
                rope=self.rope,
            )
            self.metric_head = nn.Linear(512, 1)
            nn.init.normal_(self.metric_token, std=1e-6)
        else:
            self.metric_token = None
            self.metric_decoder = None
            self.metric_head = None

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512, output_quat=False)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # ----------------------
        #            TTT
        # ----------------------

        self.ttt_layers = None
        self.ttt_gate_projs = None
        self.ttt_op_order = None

        self.ttt_layers = nn.ModuleList([
            FastWeightGluMLPMultihead(
                dim=dec_embed_dim,
                head_dim=ttt_head_dim,
                inter_multi=ttt_inter_multi,
                bias=False,
                base_lr=0.01,
                muon_update_steps=self.num_muon_update_steps,
                use_momentum=self.use_momentum,
                ttt_update_steps=self.ttt_update_steps,
                ttt_pre_norm=self.ttt_pre_norm,
            )
            for _ in self.ttt_insert_after
        ])
        self.ttt_gate_projs = nn.ModuleList([
            nn.Linear(dec_embed_dim, 1)
            for _ in self.ttt_insert_after
        ])

        for gate_proj in self.ttt_gate_projs:
            torch.nn.init.zeros_(gate_proj.weight)
            if gate_proj.bias is not None:
                torch.nn.init.zeros_(gate_proj.bias)

        self.ttt_op_order = [
            TTTOperator(start=0, end=None, update=False, apply=True),
            TTTOperator(start=0, end=None, update=True, apply=False),
        ]

        # ----------------------
        #   Attention Adapters
        # ----------------------
        self.swa_layers = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=ttt_inter_multi,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope,
            )
            for _ in self.attn_insert_after
        ])
        self.swa_gate_projs = nn.ModuleList([
            nn.Linear(dec_embed_dim, 1)
            for _ in self.attn_insert_after
        ])

        for gate_proj in self.swa_gate_projs:
            torch.nn.init.zeros_(gate_proj.weight)
            if gate_proj.bias is not None:
                torch.nn.init.zeros_(gate_proj.bias)
    
    def _initialize_ttt_layers_from_global(
        self,
        layers: Optional[nn.ModuleList],
        kind: str,
        insert_after: Optional[List[int]] = None,
    ) -> None:
        """Helper for initializing adapter layers from decoder global attention weights."""
        if layers is None or len(layers) == 0:
            print(f"{kind} initialization skipped: no target layers defined.")
            return

        insert_positions = insert_after if insert_after is not None else self.ttt_insert_after
        if not insert_positions:
            print(f"{kind} initialization skipped: no insert positions defined.")
            return

        num_decoder_layers = len(self.decoder)
        print(f"Initializing {len(layers)} {kind} layers from decoder attention blocks")
        print(f"  Insert positions: {insert_positions}")


        for layer_idx, insert_idx in enumerate(insert_positions):
            decoder_idx = int(insert_idx)
            if decoder_idx % 2 == 0:
                decoder_idx += 1  # move to the subsequent global-attention layer

            if decoder_idx >= num_decoder_layers:
                raise IndexError(
                    f"Decoder index {decoder_idx} out of range for {kind} initialization (decoder has {num_decoder_layers} layers)."
                )

            if decoder_idx % 2 == 0:
                raise AssertionError(
                    f"Decoder index {decoder_idx} is not a global-attention layer after adjustment."
                )

            source_layer = self.decoder[decoder_idx]
            target_layer = layers[layer_idx]
            target_layer.load_state_dict(source_layer.state_dict())

            print(f"  Initialized {kind}_layer[{layer_idx}] from decoder[{decoder_idx}]")

    def _initialize_swa_from_global(self):
        """Helper function to initialize swa from global."""
        if self.swa_layers is None:
            return
        self._initialize_ttt_layers_from_global(self.swa_layers, "swa", self.attn_insert_after)

    def decode(self, hidden, N, H, W, ttt_dict: Optional[dict] = None, window_size: Optional[int] = None, overlap_size: Optional[int] = None, is_first_window: bool = False,
               turn_off_ttt=False, turn_off_swa=False) -> torch.Tensor:
        """Decode.

        Args:
            hidden: The hidden.
            N: The n.
            H: The h.
            W: The w.
            ttt_dict: The ttt dict.
            window_size: The window size.
            overlap_size: The overlap size.
            is_first_window: The is first window.
            turn_off_ttt: The turn off ttt.
            turn_off_swa: The turn off swa.

        Returns:
            The return value.
        """
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        pe_token_0 = getattr(self, 'pe_token_0')  # (1, 1, 1, dim)
        pe_token_1 = getattr(self, 'pe_token_1')  # (1, 1, 1, dim)
        pe_token_2 = getattr(self, 'pe_token_2')  # (1, 1, 1, dim)
        if overlap_size is None or window_size is None:
            raise ValueError("overlap_size and window_size must be provided when num_pe_tokens > 0")
        num_overlap_with_previous = min(overlap_size, N)
        num_other_frames = min(max(window_size - 2 * overlap_size, 0), N - num_overlap_with_previous)
        num_overlap_with_later = max(min(overlap_size, N, N - num_overlap_with_previous - num_other_frames), 0)
        pe_tokens = torch.cat([
            pe_token_0.repeat(B, num_overlap_with_previous, 1, 1),
            pe_token_1.repeat(B, num_other_frames, 1, 1),
            pe_token_2.repeat(B, num_overlap_with_later, 1, 1)
        ], dim=1).to(hidden.device).to(hidden.dtype).reshape(B*N, *pe_token_0.shape[-2:])  # (B*N, 1, dim)
        hidden = torch.cat([pe_tokens, hidden], dim=1)

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + torch.ones_like(pos)
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        ttt_output_info = None
        ttt_state = ttt_dict.get("ttt") if ttt_dict is not None else None
        attn_state = ttt_dict.get("attn") if ttt_dict is not None else None
        gate_scales: List[torch.Tensor] = []
        attn_gate_scales: List[torch.Tensor] = []
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                # frame attention
                pos_reshaped = pos.reshape(B*N, hw, -1) if pos is not None else None
                hidden = hidden.reshape(B*N, hw, -1)
                hidden_for_block = hidden
                pos_for_block = pos_reshaped
            else:
                # global attention
                pos_reshaped = pos.reshape(B, N*hw, -1) if pos is not None else None
                hidden = hidden.reshape(B, N*hw, -1)
                hidden_for_block = hidden
                pos_for_block = pos_reshaped

            # Save pre-block hidden for the fixed no-skip-residual path.
            # With skip0 config removed, default behavior is skip0=False.
            layer_skip0 = (
                len(self.ttt_insert_after) == 36
                and i in self.ttt_insert_after
                and self.ttt_insert_after.index(i) % 2 == 0
            )
            
            if i % 2 == 1 and not layer_skip0:
                hidden_before_block = hidden_for_block
            elif i % 2 == 0 and layer_skip0:
                hidden_before_block = hidden_for_block
            else:
                hidden_before_block = hidden_for_block # dummy

            hidden = blk(hidden_for_block, xpos=pos_for_block)

            if ttt_state is not None and i in ttt_state.get("insert_after", []):
                # Help static analyzers: ensure non-None
                assert self.ttt_gate_projs is not None and self.ttt_layers is not None
                insert_after_list = ttt_state.get("insert_after", [])
                layer_idx = insert_after_list.index(i)

                x_for_residual = hidden.view(B, N, hw, -1)
                tokens_post = x_for_residual
                tokens_in = tokens_post

                gate_scale = torch.nn.functional.silu(self.ttt_gate_projs[layer_idx](tokens_in))
                # keep the gate scale to be always 0
                # if i <= 19: gate_scale = torch.zeros_like(gate_scale)  # turn off ttt
                if turn_off_ttt: gate_scale = torch.zeros_like(gate_scale)  # turn off ttt
                gate_scales.append(gate_scale)
                info = {
                    "ttt_op_order": ttt_state.get("ttt_op_order", []),
                    "w0": ttt_state["w0"][layer_idx],
                    "w1": ttt_state["w1"][layer_idx],
                    "w2": ttt_state["w2"][layer_idx],
                }
                ttt_output, output = self.ttt_layers[layer_idx](tokens_in, info)
                
                update_term = ttt_output * gate_scale

                tokens_out = update_term + tokens_post

                hidden = tokens_out

                if ttt_output_info is None:
                    ttt_output_info = {
                        "w0": [None] * len(insert_after_list),
                        "w1": [None] * len(insert_after_list),
                        "w2": [None] * len(insert_after_list),
                    }
                ttt_output_info["w0"][layer_idx] = output["w0"]
                ttt_output_info["w1"][layer_idx] = output["w1"]
                ttt_output_info["w2"][layer_idx] = output["w2"]

            # Sliding Window Attention (SWA)
            if attn_state is not None and i in attn_state.get("insert_after", []):
                assert self.swa_gate_projs is not None and self.swa_layers is not None
                insert_after_list = attn_state.get("insert_after", [])
                layer_idx = insert_after_list.index(i)

                patch_tokens_post_block = hidden
                x_for_residual = patch_tokens_post_block.view(B, N, hw, -1)
                x_in = x_for_residual

                history_list = attn_state.get("history", [None] * len(insert_after_list))
                history = history_list[layer_idx]
                x_in_for_layer = x_in

                # Prepare position embeddings for current tokens
                if pos is not None:
                    pos_current = pos.reshape(B, N, hw, -1).reshape(B, N * hw, -1)
                else:
                    pos_current = None

                # Check if we have KV cache from history
                use_kv_cache = (
                    history is not None 
                    and isinstance(history, dict) 
                    and "k" in history
                )

                if use_kv_cache:
                    # Use KV cache path
                    k_cache = history["k"]  # [B, num_heads, N_hist * hw, head_dim]
                    v_cache = history["v"]  # [B, num_heads, N_hist * hw, head_dim]
                    # Forward with KV cache
                    x_curr_flat = x_in_for_layer.reshape(B, N * hw, -1)
                    swa_output_flat = self.swa_layers[layer_idx].forward_with_kv_cache(
                        x_curr_flat, k_cache, v_cache,
                        xpos=pos_current,
                    )
                    swa_output = swa_output_flat.reshape(B, N, hw, -1)
                else:
                    # Original path (no history or legacy format)
                    # Handle legacy history format (raw tensor instead of dict)
                    history_raw = history if history is not None and not isinstance(history, dict) else None

                    if history_raw is not None:
                        x_with_history = torch.cat([history_raw, x_in_for_layer], dim=1)
                    else:
                        x_with_history = x_in_for_layer

                    N_total = x_with_history.shape[1]
                    x_swa = x_with_history.reshape(B, N_total * hw, -1)

                    if pos is not None:
                        pos_swa = pos.reshape(B, N, hw, -1)
                        if history_raw is not None:
                            N_hist = history_raw.shape[1]
                            pos_hist = pos_swa[:, :1].repeat(1, N_hist, 1, 1)
                            pos_swa = torch.cat([pos_hist, pos_swa], dim=1)
                        pos_swa = pos_swa.reshape(B, N_total * hw, -1)
                    else:
                        pos_swa = None

                    swa_output_full = self.swa_layers[layer_idx](
                        x_swa, 
                        xpos=pos_swa, 
                    )
                    swa_output_full = swa_output_full.reshape(B, N_total, hw, x_in.shape[-1])
                    if history_raw is not None:
                        N_hist = history_raw.shape[1]
                        swa_output = swa_output_full[:, N_hist:, :, :]
                    else:
                        swa_output = swa_output_full

                gate_scale = torch.nn.functional.silu(self.swa_gate_projs[layer_idx](swa_output))
                if turn_off_swa: gate_scale = torch.zeros_like(gate_scale)
                attn_gate_scales.append(gate_scale)

                update_term = swa_output * gate_scale
                x_out_patch = update_term + x_for_residual
                x_out_patch_flat = x_out_patch.reshape(B, N * hw, -1)
                hidden = x_out_patch_flat.reshape(B * N, hw, -1)

                # Store KV cache for next window
                # Compute KV for current x_in with history_pe (since it will be history next time)
                if ttt_output_info is None:
                    ttt_output_info = {"history": [None] * len(insert_after_list)}
                elif "history" not in ttt_output_info:
                    ttt_output_info["history"] = [None] * len(insert_after_list)

                x_for_cache = x_in
                x_for_cache_flat = x_for_cache.reshape(B, N * hw, -1)
                
                # Position for cache: use first frame's position repeated (same as original logic)
                if pos is not None:
                    pos_for_cache = pos.reshape(B, N, hw, -1)[:, :1].repeat(1, N, 1, 1).reshape(B, N * hw, -1)
                else:
                    pos_for_cache = None

                k_new, v_new = self.swa_layers[layer_idx].compute_kv_cache(x_for_cache_flat, xpos=pos_for_cache)
                
                if getattr(self, "detach_swa_history", False):
                    k_new = k_new.detach()
                    v_new = v_new.detach()
                
                ttt_output_info["history"][layer_idx] = {"k": k_new, "v": v_new}

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        avg_gate_scale = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)
        avg_attn_gate_scale: Optional[torch.Tensor] = None
        if gate_scales:
            all_gate_scales = torch.cat([g.flatten() for g in gate_scales])
            if all_gate_scales.numel() > 0:
                avg_gate_scale = all_gate_scales.abs().mean()
        if attn_gate_scales:
            all_attn_gate_scales = torch.cat([g.flatten() for g in attn_gate_scales])
            if all_attn_gate_scales.numel() > 0:
                avg_attn_gate_scale = all_attn_gate_scales.abs().mean()

        if len(final_output) < 2:
            raise RuntimeError(
                f"Decoder expected to collect two final outputs but got {len(final_output)}."
            )

        return (
            torch.cat([final_output[0], final_output[1]], dim=-1),
            (pos.reshape(B*N, hw, -1) if pos is not None else None),
            ttt_output_info,
            avg_gate_scale,
            avg_attn_gate_scale,
            gate_scales,
        )
    
    def forward(self, imgs, *args, **kwargs):
        """Forward.

        Args:
            imgs: The imgs.
        """
        # Windowing controls (optional)
        window_size = kwargs.pop('window_size', -1)
        overlap_size = kwargs.pop('overlap_size', 1)
        num_iterations = kwargs.pop('num_iterations', 1)
        no_detach = kwargs.pop('no_detach', False)
        sim3 = kwargs.pop('sim3', False)
        se3 = kwargs.pop('se3', False)
        reset_every = kwargs.pop('reset_every', 0)  # reset TTT / adapter state every N windows (0 disables)
        turn_off_ttt = kwargs.pop('turn_off_ttt', False)
        turn_off_swa = kwargs.pop('turn_off_swa', False)
        sim3_scale_mode = kwargs.pop('sim3_scale_mode', 'median')

        if sim3 and se3:
            raise ValueError("'sim3' and 'se3' alignments are mutually exclusive; enable only one.")

        # Ensure at least one decode iteration so that 'hidden' is always defined
        try:
            num_iterations = int(num_iterations)
        except Exception:
            num_iterations = 1
        if num_iterations < 1:
            num_iterations = 1
        try:
            reset_every = int(reset_every)
        except Exception:
            reset_every = 0
        if reset_every < 0:
            reset_every = 0

        # Ensure batch dimension
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(0)

        # Normalize
        # imgs = (imgs - self.image_mean) / self.image_std

        B, N, C, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # --- Unified Windowed Inference ---
        if window_size <= 0 or window_size >= N:
            windows = [(0, N)]
            eff_overlap = 0
            eff_window_size = N
        else:
            windows = []
            step = max(window_size - overlap_size, 1)
            for start_idx in range(0, N, step):
                end_idx = min(start_idx + window_size, N)
                if end_idx - start_idx >= overlap_size or (end_idx == N and start_idx < N):
                    windows.append((start_idx, end_idx))
                if end_idx == N:
                    break
            eff_overlap = overlap_size
            eff_window_size = window_size

        # Cache the effective window and overlap sizes for downstream merging utilities
        self._last_window_size = eff_window_size
        self._last_overlap_size = eff_overlap

        # Prepare TTT states across windows
        if self.ttt_layers is not None:
            w0 = [None] * len(self.ttt_insert_after)
            w1 = [None] * len(self.ttt_insert_after)
            w2 = [None] * len(self.ttt_insert_after)
        else:
            w0 = w1 = w2 = None

        # Prepare SWA history states across windows
        swa_history = [None] * len(self.attn_insert_after) if self.swa_layers is not None else None

        def reset_adaptive_states():
            """Reset fast-weight TTT states only; SWA history is preserved across resets."""
            nonlocal w0, w1, w2
            if self.ttt_layers is not None:
                w0 = [None] * len(self.ttt_insert_after)
                w1 = [None] * len(self.ttt_insert_after)
                w2 = [None] * len(self.ttt_insert_after)

        all_predictions = []
        all_gate_scales: List[torch.Tensor] = []
        all_attn_gate_scales: List[torch.Tensor] = []
        
        windows_iter = windows
        for window_idx, (start_idx, end_idx) in enumerate(windows_iter):
            if reset_every > 0 and window_idx > 0 and window_idx % reset_every == 0:
                reset_adaptive_states()
            imgs_w = imgs[:, start_idx:end_idx]  # (B, Nw, C, H, W)
            imgs_w = imgs_w.to(self.image_mean.device)
            imgs_w = (imgs_w - self.image_mean) / self.image_std
            Nw = imgs_w.shape[1]

            # Initialize to satisfy static analyzers; will be set inside decode loop
            hidden = None  # type: ignore[assignment]
            pos = None     # type: ignore[assignment]

            for _ in range(num_iterations):
                if self.ttt_layers is not None and w0 is None:
                    w0 = [None] * len(self.ttt_insert_after)
                    w1 = [None] * len(self.ttt_insert_after)
                    w2 = [None] * len(self.ttt_insert_after)

                if self.swa_layers is not None and swa_history is None:
                    swa_history = [None] * len(self.attn_insert_after)

                imgs_flat = imgs_w.reshape(B * Nw, C, H, W)
                hidden_input = self.encoder(imgs_flat, is_training=True)
                if isinstance(hidden_input, dict):
                    hidden_input = hidden_input["x_norm_patchtokens"]

                # Prepare adapter control dictionaries for decode
                ttt_state = None
                attn_state = None

                if self.ttt_layers is not None:
                    ttt_state = {
                        "ttt_op_order": self.ttt_op_order if self.ttt_op_order is not None else [],
                        "insert_after": self.ttt_insert_after,
                        "w0": w0,
                        "w1": w1,
                        "w2": w2,
                    }

                if self.swa_layers is not None:
                    attn_state = {
                        "insert_after": self.attn_insert_after,
                        "history": swa_history,
                    }

                if ttt_state is None and attn_state is None:
                    ttt_dict = None
                else:
                    ttt_dict = {
                        "ttt": ttt_state,
                        "attn": attn_state,
                    }
                hidden, pos, ttt_output_info, decode_avg_gate_scale, decode_avg_attn_gate_scale, _decode_gate_scales = self.decode(
                    hidden_input, Nw, H, W,
                    ttt_dict=ttt_dict,
                    window_size=window_size,
                    overlap_size=overlap_size,
                    is_first_window=(start_idx == 0),
                    turn_off_ttt=turn_off_ttt,
                    turn_off_swa=turn_off_swa,
                )
                if decode_avg_gate_scale is not None:
                    all_gate_scales.append(decode_avg_gate_scale.detach().cpu())
                if decode_avg_attn_gate_scale is not None:
                    all_attn_gate_scales.append(decode_avg_attn_gate_scale.detach().cpu())

                # TODO: get the updated state from the ttt layer
                if self.ttt_layers is not None and ttt_output_info is not None:
                    w0, w1, w2 = ttt_output_info["w0"], ttt_output_info["w1"], ttt_output_info["w2"]
                
                # TODO: get the updated history from the swa layer
                if ttt_output_info is not None:
                    swa_history = ttt_output_info.get("history", swa_history)

            # If for some reason decoding didn't produce hidden (e.g., empty window), skip this window
            if hidden is None:
                continue

            point_hidden = self.point_decoder(hidden, xpos=pos)
            if self.use_conf and self.conf_decoder is not None:
                conf_hidden = self.conf_decoder(hidden, xpos=pos)
            else:
                conf_hidden = None
            
            if self.pi3x and self.pi3x_metric:
                hw = hidden.shape[1]
                pos_hw = pos.reshape(B, Nw*hw, -1)
                metric_hidden = self.metric_decoder(self.metric_token.repeat(B, 1, 1), hidden.reshape(B, Nw*hw, -1), xpos=pos_hw[:, 0:1], ypos=pos_hw)
            else:
                metric_hidden = None

            camera_hidden = self.camera_decoder(hidden, xpos=pos)

            global_camera_hidden = camera_hidden

            with torch.autocast(device_type='cuda', enabled=False):
                # local points
                point_hidden = point_hidden.float()
                if self.pi3x:
                    xy, z = self.point_head(point_hidden[:, self.patch_start_idx:], patch_h=patch_h, patch_w=patch_w)
                    xy = xy.permute(0, 2, 3, 1).reshape(B, Nw, H, W, -1)
                    z = z.permute(0, 2, 3, 1).reshape(B, Nw, H, W, -1)
                    z = torch.exp(z.clamp(max=15.0))
                    local_points = torch.cat([xy * z, z], dim=-1)
                else:
                    ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, Nw, H, W, -1)
                    xy, z = ret.split([2, 1], dim=-1)
                    z = torch.exp(z)
                    local_points = torch.cat([xy * z, z], dim=-1)

                # confidence
                if conf_hidden is not None and self.conf_head is not None:
                    conf_hidden = conf_hidden.float()
                    conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, Nw, H, W, -1)
                else:
                    conf = None

                # camera
                global_camera_hidden = global_camera_hidden.float()
                camera_poses = self.camera_head(global_camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, Nw, 4, 4)
                camera_qvec = None
                local_camera_poses = None
                local_camera_qvec = None

                # metric
                if self.pi3x and self.pi3x_metric and metric_hidden is not None:
                    metric = self.metric_head(metric_hidden.float()).reshape(B).exp()
                    
                    # apply metric to points and camera poses
                    # points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3] * metric.view(B, 1, 1, 1, 1)
                    camera_poses[..., :3, 3] = camera_poses[..., :3, 3] * metric.view(B, 1, 1)
                    local_points = local_points * metric.view(B, 1, 1, 1, 1)
                    if local_camera_poses is not None:
                        local_camera_poses[..., :3, 3] = local_camera_poses[..., :3, 3] * metric.view(B, 1, 1)
                else:
                    metric = None


            # unproject local points using camera poses
            with torch.autocast(device_type='cuda', enabled=False):
                points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]


            def maybe_detach(t, no_detach=no_detach):
                """Maybe detach.

                Args:
                    t: The t.
                    no_detach: The no detach.
                """
                if t is None:
                    return None
                return t if self.training or no_detach else t.detach().cpu()

            pred_dict = dict(
                points=maybe_detach(points, no_detach=no_detach),
                local_points=maybe_detach(local_points, no_detach=no_detach),
                conf=maybe_detach(conf, no_detach=no_detach),
                camera_poses=maybe_detach(camera_poses, no_detach=no_detach),
                local_camera_poses=maybe_detach(local_camera_poses, no_detach=no_detach),
                camera_qvec=maybe_detach(camera_qvec, no_detach=no_detach),
                local_camera_qvec=maybe_detach(local_camera_qvec, no_detach=no_detach),
                metric=maybe_detach(metric, no_detach=no_detach),
            )
            all_predictions.append(pred_dict)

        # Merge windowed predictions
        # When reset is enabled but explicit Sim3/SE3 alignment is off, keep each reset block
        # in a stable rigid frame by applying one estimated transform per block.
        align_on_resets_without_explicit_pose = reset_every > 0 and not sim3 and not se3
        if sim3:
            merged = self._merge_windowed_predictions_sim3(
                all_predictions, 
                allow_scale=True, 
                scale_mode=sim3_scale_mode,
            )
        elif se3 or align_on_resets_without_explicit_pose:
            merged = self._merge_windowed_predictions_sim3(
                all_predictions, 
                allow_scale=False,
                reset_every=reset_every,
                reuse_transform_within_reset_block=align_on_resets_without_explicit_pose,
            )
        else:
            merged = self._merge_windowed_predictions(all_predictions, eff_window_size, eff_overlap)
        if all_gate_scales:
            merged["avg_gate_scale"] = torch.stack(all_gate_scales).mean()
        if all_attn_gate_scales:
            merged["attn_gate_scale"] = torch.stack(all_attn_gate_scales).mean()
        
        return merged

    def _merge_windowed_predictions(self, all_predictions, window_size, overlap_size):
        """
        Merge predictions from multiple windows by concatenating along the time dimension
        while removing overlapping frames.
        """
        if not all_predictions:
            return {}
        if len(all_predictions) == 1:
            return all_predictions[0]

        merged_predictions = {}
        keys = list(all_predictions[0].keys())
        sequence_keys = {"points", "local_points", "conf", "camera_poses", "local_camera_poses", "camera_qvec", "local_camera_qvec"}
        for key in keys:
            # Collect window tensors
            window_tensors = [pred.get(key, None) for pred in all_predictions]

            # Skip if all windows have None for this key
            if all(t is None for t in window_tensors):
                continue

            # Only perform overlap-aware concatenation for known sequence-shaped tensors
            if key in sequence_keys:
                # Filter out None windows safely while preserving positions for slicing
                result_parts = []

                # First window: drop last overlap_size frames
                first = window_tensors[0]
                if first is not None:
                    if overlap_size > 0 and first.shape[1] > overlap_size:
                        result_parts.append(first[:, :-overlap_size])
                    elif overlap_size > 0 and first.shape[1] <= overlap_size:
                        # If window shorter or equal to overlap, drop completely
                        pass
                    else:
                        result_parts.append(first)

                # Middle windows: drop last overlap_size frames
                for tensor in window_tensors[1:-1]:
                    if tensor is None:
                        continue
                    if overlap_size > 0 and tensor.shape[1] > overlap_size:
                        result_parts.append(tensor[:, :-overlap_size])
                    elif overlap_size > 0 and tensor.shape[1] <= overlap_size:
                        # If window shorter or equal to overlap, drop completely
                        continue
                    else:
                        result_parts.append(tensor)

                # Last window: keep all frames
                last_tensor = window_tensors[-1]
                if last_tensor is not None:
                    result_parts.append(last_tensor)

                if result_parts:
                    merged_predictions[key] = torch.cat(result_parts, dim=1)
                else:
                    # Fallback: if everything was dropped due to tiny windows, keep last non-None
                    for t in reversed(window_tensors):
                        if t is not None:
                            merged_predictions[key] = t
                            break
            else:
                # Non-sequence keys: keep the last non-None
                for t in reversed(window_tensors):
                    if t is not None:
                        merged_predictions[key] = t
                        break

        # Instead of computing overlap losses here, export overlap prev/next tensors for trainer-side chunk losses
        if overlap_size > 0 and len(all_predictions) > 1:
            prev_cam_chunks = []
            next_cam_chunks = []
            prev_pcd_chunks = []
            next_pcd_chunks = []
            next_conf_chunks = []

            for i in range(len(all_predictions) - 1):
                pred_a = all_predictions[i]
                pred_b = all_predictions[i + 1]

                cam_a = pred_a.get("camera_poses", None)
                cam_b = pred_b.get("camera_poses", None)
                lpts_a = pred_a.get("local_points", None)
                lpts_b = pred_b.get("local_points", None)
                conf_a = pred_a.get("conf", None)
                conf_b = pred_b.get("conf", None)

                # Only collect when both sides have enough frames for a full overlap window
                if cam_a is not None and cam_b is not None and cam_a.shape[1] >= overlap_size and cam_b.shape[1] >= overlap_size:
                    S_a = cam_a.shape[1]
                    # Take last overlap_size from A and first overlap_size from B
                    prev_cam_chunks.append(cam_a[:, S_a - overlap_size: S_a])  # (B, O, 4, 4)
                    next_cam_chunks.append(cam_b[:, 0: overlap_size])         # (B, O, 4, 4)

                if lpts_a is not None and lpts_b is not None and lpts_a.shape[1] >= overlap_size and lpts_b.shape[1] >= overlap_size:
                    S_a = lpts_a.shape[1]
                    prev_pcd_chunks.append(lpts_a[:, S_a - overlap_size: S_a])  # (B, O, H, W, 3)
                    next_pcd_chunks.append(lpts_b[:, 0: overlap_size])          # (B, O, H, W, 3)
                    if conf_b is not None and conf_b.shape[1] >= overlap_size:
                        next_conf_chunks.append(conf_b[:, 0: overlap_size].squeeze(-1))  # (B, O, H, W)

            # Stack along a new chunk dimension if any collected
            if prev_cam_chunks and next_cam_chunks:
                merged_predictions["overlap_prev_cam"] = torch.stack(prev_cam_chunks, dim=1)  # (B, K, O, 4, 4)
                merged_predictions["overlap_next_cam"] = torch.stack(next_cam_chunks, dim=1)  # (B, K, O, 4, 4)
            if prev_pcd_chunks and next_pcd_chunks:
                merged_predictions["overlap_prev_pcd"] = torch.stack(prev_pcd_chunks, dim=1)  # (B, K, O, H, W, 3)
                merged_predictions["overlap_next_pcd"] = torch.stack(next_pcd_chunks, dim=1)  # (B, K, O, H, W, 3)
                if next_conf_chunks:
                    merged_predictions["overlap_next_conf"] = torch.stack(next_conf_chunks, dim=1)  # (B, K, O, H, W)

        return merged_predictions

    def _merge_windowed_predictions_sim3(
        self,
        all_predictions,
        allow_scale: bool = True,
        scale_mode: str = 'median',
        reset_every: int = 0,
        reuse_transform_within_reset_block: bool = False,
    ):
        """
        Merge windowed predictions by estimating relative poses between overlaps.
        When ``allow_scale`` is True this performs Sim(3) alignment (scale+SE(3));
        when False it reduces to SE(3) alignment by keeping the scale fixed to 1.
        If ``reuse_transform_within_reset_block`` is enabled with ``reset_every > 0``,
        one transform is estimated at each reset boundary and reused for the rest of
        that reset block.
        """
        # print("allow_scale -----------------------------", allow_scale)
        if not all_predictions:
            return {}
        if len(all_predictions) == 1:
            return all_predictions[0]

        # Locate a reference tensor to determine batch/device/dtype information
        sample_tensor = None
        for pred in all_predictions:
            for key in ("points", "camera_poses", "local_points", "conf"):
                tensor = pred.get(key, None)
                if tensor is not None:
                    sample_tensor = tensor
                    break
            if sample_tensor is not None:
                break
        if sample_tensor is None:
            raise ValueError("Sim3 merge requires at least one tensor prediction")

        device = sample_tensor.device
        dtype = sample_tensor.dtype
        batch_size = sample_tensor.shape[0]

        identity_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        zero_trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        one_scale = torch.ones(batch_size, device=device, dtype=dtype)

        aligned_predictions: List[dict] = []
        sim3_scales: Optional[List[torch.Tensor]] = [] if allow_scale else None
        sim3_poses: List[torch.Tensor] = []

        window_size = getattr(self, "_last_window_size", -1)
        overlap_size = getattr(self, "_last_overlap_size", 0)

        def _estimate_relative_sim3(prev_aligned: dict, curr_raw: dict, overlap: int, current_allow_scale: bool, forced_scale: Optional[torch.Tensor] = None):
            """Helper function to estimate relative sim3.

            Args:
                prev_aligned: The prev aligned.
                curr_raw: The curr raw.
                overlap: The overlap.
                current_allow_scale: The current allow scale.
                forced_scale: The forced scale.
            """
            if overlap <= 0:
                return torch.ones_like(one_scale), identity_rot, zero_trans

            prev_cam = prev_aligned.get("camera_poses", None)
            curr_cam = curr_raw.get("camera_poses", None)
            if prev_cam is None or curr_cam is None or prev_cam.shape[1] == 0 or curr_cam.shape[1] == 0:
                return torch.ones_like(one_scale), identity_rot, zero_trans

            prev_frames = prev_cam.shape[1]
            prev_idx = max(prev_frames - overlap, 0)

            prev_pose = prev_cam[:, prev_idx]
            curr_pose = curr_cam[:, 0]

            R_prev = prev_pose[:, :3, :3]
            t_prev = prev_pose[:, :3, 3]
            R_curr = curr_pose[:, :3, :3]
            t_curr = curr_pose[:, :3, 3]

            relative_rot = torch.matmul(R_prev, R_curr.transpose(-1, -2))

            relative_scale = torch.ones_like(one_scale)
            if forced_scale is not None:
                relative_scale = forced_scale
            elif current_allow_scale:
                prev_local_raw = prev_aligned.get("local_points", None)
                if prev_local_raw is None:
                    prev_local_raw = prev_aligned.get("_local_points_raw", None)
                curr_local_raw = curr_raw.get("local_points", None)

                if (
                    prev_local_raw is not None
                    and curr_local_raw is not None
                    and prev_local_raw.shape[1] > prev_idx
                    and curr_local_raw.shape[1] > 0
                ):
                    if scale_mode in ['median_all', 'trimmed_mean_all']:
                        # Use all overlapping frames
                        actual_overlap = min(overlap, prev_local_raw.shape[1] - prev_idx, curr_local_raw.shape[1])
                        if actual_overlap > 0:
                            prev_depth = prev_local_raw[:, prev_idx : prev_idx + actual_overlap, ..., 2]
                            curr_depth = curr_local_raw[:, :actual_overlap, ..., 2]
                        else:
                            # Fallback to single frame if overlap calculation fails (should not happen given checks above)
                            prev_depth = prev_local_raw[:, prev_idx, ..., 2]
                            curr_depth = curr_local_raw[:, 0, ..., 2]
                    else:
                        # Use only the first overlapping frame (standard behavior)
                        prev_depth = prev_local_raw[:, prev_idx, ..., 2]
                        curr_depth = curr_local_raw[:, 0, ..., 2]

                    prev_depth_f32 = prev_depth.to(torch.float32)
                    curr_depth_f32 = curr_depth.to(torch.float32)
                    eps_depth = torch.finfo(torch.float32).eps
                    valid = (
                        torch.isfinite(prev_depth_f32)
                        & torch.isfinite(curr_depth_f32)
                        & (curr_depth_f32.abs() > eps_depth)
                    )

                    prev_depth_flat = prev_depth_f32.reshape(batch_size, -1)
                    curr_depth_flat = curr_depth_f32.reshape(batch_size, -1)
                    valid_flat = valid.reshape(batch_size, -1)
                    
                    if scale_mode in ['median', 'median_all']:
                        scale_values = []
                        for b in range(batch_size):
                            valid_idx = valid_flat[b]
                            if valid_idx.any():
                                ratios = prev_depth_flat[b, valid_idx] / curr_depth_flat[b, valid_idx]
                                scale_values.append(ratios.median())
                            else:
                                scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                        relative_scale = torch.stack(scale_values).to(dtype)
                    elif scale_mode in ['trimmed_mean', 'trimmed_mean_all']:
                        # Vectorized implementation for trimmed mean
                        # Mask invalid entries with NaN or filter before passing?
                        # robust_scale_estimation expects (B, N)
                        # Since N varies per batch due to validity, we might still need a loop or careful padding.
                        # However, valid_flat is (B, N_pixels).
                        
                        # To keep it simple and consistent with the median loop structure for now (which handles varying valid counts per batch):
                        scale_values = []
                        for b in range(batch_size):
                            valid_idx = valid_flat[b]
                            if valid_idx.any():
                                ratios = prev_depth_flat[b, valid_idx] / curr_depth_flat[b, valid_idx]
                                # ratios is 1D tensor of valid pixels
                                # We need to pass (1, N) to robust_scale_estimation to reuse it, or just use it directly if we modify it to handle 1D
                                # robust_scale_estimation expects (B, N). Let's reshape.
                                scale_val = robust_scale_estimation(ratios.unsqueeze(0), trim_ratio=0.25).squeeze(0)
                                scale_values.append(scale_val)
                            else:
                                scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                        relative_scale = torch.stack(scale_values).to(dtype)
                    elif scale_mode in ['sim3_avg1']:
                        scale_values = []
                        for b in range(batch_size):
                            valid_idx = valid_flat[b]
                            if valid_idx.any():
                                ratios = prev_depth_flat[b, valid_idx] / curr_depth_flat[b, valid_idx]
                                scale_values.append(ratios.median())
                            else:
                                scale_values.append(torch.tensor(1.0, device=device, dtype=torch.float32))
                        relative_scale = torch.stack(scale_values).to(dtype)
                        relative_scale = (relative_scale + 1.0) / 2.0
                    else:
                        raise ValueError(f"Unknown scale_mode: {scale_mode}")

                    relative_scale = torch.clamp(relative_scale, min=1e-3, max=1e3)

            rotated_curr_centers = torch.matmul(relative_rot, t_curr.unsqueeze(-1)).squeeze(-1)
            relative_trans = t_prev - relative_scale.unsqueeze(-1) * rotated_curr_centers

            return relative_scale, relative_rot.to(dtype), relative_trans.to(dtype)

        block_scale: Optional[torch.Tensor] = None
        block_rot: Optional[torch.Tensor] = None
        block_trans: Optional[torch.Tensor] = None

        for window_idx, pred in enumerate(all_predictions):
            if window_idx == 0:
                current_scale = torch.ones_like(one_scale)
                current_rot = identity_rot.clone()
                current_trans = zero_trans.clone()
                if reuse_transform_within_reset_block and reset_every > 0:
                    block_scale = current_scale.clone()
                    block_rot = current_rot.clone()
                    block_trans = current_trans.clone()
            else:
                prev_aligned = aligned_predictions[-1]
                reuse_block_transform = (
                    reuse_transform_within_reset_block
                    and reset_every > 0
                    and window_idx % reset_every != 0
                    and block_rot is not None
                    and block_trans is not None
                )
                if reuse_block_transform:
                    current_rot = block_rot.clone()
                    current_trans = block_trans.clone()
                    if allow_scale and block_scale is not None:
                        current_scale = block_scale.clone()
                    else:
                        current_scale = torch.ones_like(one_scale)
                else:
                    current_scale, current_rot, current_trans = _estimate_relative_sim3(
                        prev_aligned, pred, overlap_size, allow_scale
                    )
                    if reuse_transform_within_reset_block and reset_every > 0:
                        block_scale = current_scale.clone()
                        block_rot = current_rot.clone()
                        block_trans = current_trans.clone()

            if allow_scale and sim3_scales is not None:
                sim3_scales.append(current_scale.clone())
                # print(current_scale, 'current_scale-----------------')
            pose_mat = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            pose_mat[:, :3, :3] = current_rot
            pose_mat[:, :3, 3] = current_trans
            sim3_poses.append(pose_mat)

            aligned_pred: dict = {}

            original_local_points = pred.get("local_points", None)
            aligned_pred["_local_points_raw"] = original_local_points

            if original_local_points is not None:
                if allow_scale: # Keep using global allow_scale for applying scale if we have it, or maybe we should track per-window scale application?
                    # Actually, current_scale will be 1.0 if current_allow_scale was False.
                    # So we can just always apply current_scale.
                    scale_factor = current_scale.view(batch_size, 1, 1, 1, 1)
                    aligned_local_points = original_local_points * scale_factor
                else:
                    aligned_local_points = original_local_points
            else:
                aligned_local_points = None
            aligned_pred["local_points"] = aligned_local_points

            def _transform_camera(cam_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                """Helper function to transform camera.

                Args:
                    cam_tensor: The cam tensor.

                Returns:
                    The return value.
                """
                if cam_tensor is None:
                    return None
                frames = cam_tensor.shape[1]
                rot_local = cam_tensor[..., :3, :3]
                trans_local = cam_tensor[..., :3, 3]
                rot_global = torch.matmul(
                    current_rot.unsqueeze(1).expand(-1, frames, -1, -1),
                    rot_local
                )
                rotated_trans = torch.matmul(
                    current_rot.unsqueeze(1).expand(-1, frames, -1, -1),
                    trans_local.unsqueeze(-1)
                ).squeeze(-1)
                if allow_scale:
                    rotated_trans = rotated_trans * current_scale.view(batch_size, 1, 1)
                trans_global = rotated_trans + current_trans.unsqueeze(1)
                cam_out = cam_tensor.clone()
                cam_out[..., :3, :3] = rot_global
                cam_out[..., :3, 3] = trans_global
                return cam_out

            camera_global = _transform_camera(pred.get("camera_poses", None))
            aligned_pred["camera_poses"] = camera_global

            local_camera_global = _transform_camera(pred.get("local_camera_poses", None))
            aligned_pred["local_camera_poses"] = local_camera_global

            if camera_global is not None and aligned_local_points is not None:
                aligned_points = torch.einsum(
                    'bnij, bnhwj -> bnhwi',
                    camera_global,
                    homogenize_points(aligned_local_points)
                )[..., :3]
            else:
                points = pred.get("points", None)
                if points is not None:
                    rotated_points = torch.einsum('bij, bnhwj -> bnhwi', current_rot, points)
                    if allow_scale:
                        rotated_points = rotated_points * current_scale.view(batch_size, 1, 1, 1, 1)
                    aligned_points = rotated_points + current_trans.view(batch_size, 1, 1, 1, 3)
                else:
                    aligned_points = None
            aligned_pred["points"] = aligned_points

            aligned_pred["conf"] = pred.get("conf", None)

            for key, value in pred.items():
                if key in aligned_pred:
                    continue
                aligned_pred[key] = value

            aligned_predictions.append(aligned_pred)

        aligned_predictions_clean = []
        for pred in aligned_predictions:
            cleaned = pred.copy()
            cleaned.pop("_local_points_raw", None)
            aligned_predictions_clean.append(cleaned)

        merged = self._merge_windowed_predictions(aligned_predictions_clean, window_size, overlap_size)

        pose_key = "chunk_sim3_poses" if allow_scale else "chunk_se3_poses"
        if allow_scale and sim3_scales:
            merged["chunk_sim3_scales"] = torch.stack(sim3_scales, dim=1)
        if sim3_poses:
            merged[pose_key] = torch.stack(sim3_poses, dim=1)
        merged["alignment_mode"] = "sim3" if allow_scale else "se3"

        return merged
