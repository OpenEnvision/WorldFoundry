# Inference-only Being-H0.5 runtime retained in-tree.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch.nn.functional as F
import os
import torch
import torch.utils.checkpoint
from typing import List, Optional, Tuple, Union
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, or_masks, and_masks
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from ..preprocessing.conversation import get_conv_template
from ..preprocessing.constants import LLM_MODEL_ARCH, VIT_MODEL_ARCH, CONNECTOR_ARCH
from .internvit import has_flash_attn
from .layers import *


logger = logging.get_logger(__name__)


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    """
    Create sparse attention mask
    - causal: Standard causal attention (q_idx >= kv_idx)
    - full: Tokens within the same split can attend to each other
    - Different samples cannot attend to each other
    """
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_mask(b, h, q_idx, kv_idx):
        # full attention: tokens within the same split_id can attend to each other
        return (split_seq_id[q_idx] == split_seq_id[kv_idx]) & (split_seq_id[q_idx] >= 0)

    def sample_mask(b, h, q_idx, kv_idx):
        # Can only attend to tokens within the same sample
        return document_id[q_idx] == document_id[kv_idx]

    # Assign split_id to each token
    split_seq_id_list = []
    for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
        # Splits in full mode get unique id, causal mode gets -1
        value = i if mode == 'full' else -1
        split_seq_id_list.extend([value] * length)

    split_seq_id = torch.tensor(split_seq_id_list, dtype=torch.float32, device=device)

    # Assign document_id to each sample
    document_id = torch.cat([
        torch.full((l,), i, dtype=torch.float32)
        for i, l in enumerate(document_lens, start=1)
    ]).to(device)

    # Combine masks: (causal OR full) AND sample
    return and_masks(or_masks(causal_mask, full_mask), sample_mask)


def create_dense_attention_masks(document_lens, split_lens, attn_modes, device):
    split_ranges = []
    split_start = 0
    for length, mode in zip(split_lens, attn_modes):
        split_end = split_start + int(length)
        split_ranges.append((split_start, split_end, mode))
        split_start = split_end

    masks = []
    doc_start = 0
    for doc_len in document_lens:
        doc_len = int(doc_len)
        doc_end = doc_start + doc_len
        positions = torch.arange(doc_len, device=device)
        allowed = positions[:, None] >= positions[None, :]

        for split_start, split_end, mode in split_ranges:
            if mode != "full":
                continue
            overlap_start = max(split_start, doc_start)
            overlap_end = min(split_end, doc_end)
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - doc_start
            local_end = overlap_end - doc_start
            allowed[local_start:local_end, local_start:local_end] = True

        mask = torch.zeros((doc_len, doc_len), dtype=torch.float32, device=device)
        mask = mask.masked_fill(~allowed, float("-inf"))
        masks.append(mask)
        doc_start = doc_end

    return masks


class BeingHConfig(PretrainedConfig):
    model_type = 'beingh'
    is_composition = True

    def __init__(
            self,
            llm_config=None,
            vit_config=None,
            connector_arch=None,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            action_chunk_length = 16,
            gen_action_type = "action_token", # action_token, prop_hidden, last_hidden
            layer_select_for_action = -1,
            action_token_num = 16,
            learnable_action_query = False,
            num_inference_timesteps = 4,
            prompt_template = "long",
            use_expert=True,
            use_flow_matching=False,

            attn_mode="causal",
            **kwargs):
        super().__init__(**kwargs)

        if not isinstance(llm_config, dict):
            self.llm_config = llm_config
        else:
            CustomConfig = LLM_MODEL_ARCH[llm_config['architectures'][0]][0]
            self.llm_config = CustomConfig(**llm_config)

        if not isinstance(vit_config, dict):
            self.vit_config = vit_config
        else:
            CustomViTConfig = VIT_MODEL_ARCH[vit_config['architectures'][0]][0]
            self.vit_config = CustomViTConfig(**vit_config)

        #self.tokenizer_class = tokenizer_class
        self.connector_arch = connector_arch

        self.template = template
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio

        self.action_chunk_length = action_chunk_length
        self.gen_action_type = gen_action_type

        self.layer_select_for_action = layer_select_for_action
        self.action_token_num = action_token_num
        self.learnable_action_query = learnable_action_query
        self.prompt_template = prompt_template

        self.use_expert = use_expert
        self.use_flow_matching = use_flow_matching
        self.attn_mode = attn_mode

        self.max_num_embodiments = 32
        self.num_timestep_buckets = 1000
        self.num_inference_timesteps = num_inference_timesteps

        # =====================================================
        # MPG Enhancement Parameters
        # =====================================================
        self.use_mpg = kwargs.get('use_mpg', False)
        self.mpg_num_projections = kwargs.get('mpg_num_projections', 32)
        self.mpg_lambda = kwargs.get('mpg_lambda', 0.0)
        self.mpg_use_stop_gradient = kwargs.get('mpg_use_stop_gradient', True)
        self.mpg_refinement_iters = kwargs.get('mpg_refinement_iters', 1)
        self.mpg_gate_temperature = kwargs.get('mpg_gate_temperature', 2.0)  # Calibrated for LayerNorm features

        # =====================================================
        # RTC checkpoint compatibility and inference prefix control.
        # =====================================================
        self.use_training_time_rtc = kwargs.get('use_training_time_rtc', False)
        self.use_inference_prefix_overwrite = kwargs.get('use_inference_prefix_overwrite', False)


class BeingH(PreTrainedModel):
    config_class = BeingHConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'beingh'
    _no_split_modules = ['InternVisionModel', 'InternLM2DecoderLayer', 'Qwen2DecoderLayer']

    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = False

    def __init__(self, language_model, vit_model, connector, config: BeingHConfig, use_flash_attn=True):
        super().__init__(config)

        self.select_layer = config.select_layer
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio

        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vit_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        self.config = config

        self.vit_model = vit_model
        self.language_model = language_model
        self.connector = connector
        self.use_expert = config.use_expert
        self.use_flow_matching = config.use_flow_matching

        self.hidden_size = config.llm_config.hidden_size
        self.action_hidden_size = config.llm_config.expert_config.hidden_size if \
                                self.use_expert else config.llm_config.hidden_size

        # --- New code: Initialize robot-related modules ---
        # Get dimension information from config
        self.action_chunk_length = config.action_chunk_length
        self.gen_action_type = config.gen_action_type

        self.layer_select_for_action = config.layer_select_for_action
        self.action_token_num = config.action_token_num
        self.max_num_embodiments = config.max_num_embodiments

        self.unified_state_dim = 200
        self.unified_action_dim = 200

        # Proprioception Encoder
        self.proprio_encoder_robot = SimpleMLP(
            input_dim=self.unified_state_dim,
            hidden_dim=self.action_hidden_size,
            output_dim=self.action_hidden_size,
        )

        if self.use_flow_matching:
            logger.info("Using Flow Matching for action generation.")

            self.action_encoder = ActionEncoder(
                action_dim=self.unified_action_dim,
                hidden_size=self.action_hidden_size,
            )
            self.action_decoder = SimpleMLP(
                input_dim=self.action_hidden_size,
                hidden_dim=self.action_hidden_size,
                output_dim=self.unified_action_dim,
            )
            self.num_timestep_buckets = config.num_timestep_buckets
            self.num_inference_timesteps = config.num_inference_timesteps

            # ============================================================================
            # MPG Enhancement Module
            # ============================================================================
            self.use_mpg = config.use_mpg
            if self.use_mpg:
                logger.info("Initializing MPG enhancement module.")
                # UNIVERSAL SOLUTION: Always create projection layers for consistent model structure
                action_dim_for_proj = self.action_hidden_size

                self.action_to_vlm_proj = nn.Linear(
                    action_dim_for_proj,  # Action encoder output
                    self.hidden_size,     # VLM dimension
                )
                self.vlm_to_action_proj = nn.Linear(
                    self.hidden_size,     # VLM dimension
                    action_dim_for_proj,  # Action dimension
                )

                if action_dim_for_proj != self.hidden_size:
                    logger.info(f"MPG: Created projection layers ({action_dim_for_proj} <-> {self.hidden_size})")
                else:
                    logger.info(f"MPG: Created projection layers (unified_dim={self.hidden_size})")

                # MPG module (operates in VLM dimension space)
                self.mpg = MPGEnhancement(
                    obs_feature_dim=self.hidden_size,
                    action_feature_dim=self.action_hidden_size,
                    num_projections=config.mpg_num_projections,
                    lambda_strength=config.mpg_lambda,
                    use_stop_gradient=config.mpg_use_stop_gradient,
                    gate_temperature=config.mpg_gate_temperature,
                )

                self.mpg_refinement_iters = config.mpg_refinement_iters
                self.last_mpg_gate = None
                self.last_mpg_transport_cost = None
                logger.info(f"MPG initialized: lambda={config.mpg_lambda}, projections={config.mpg_num_projections}, gate_temp={config.mpg_gate_temperature}")
            else:
                self.mpg = None
                self.action_to_vlm_proj = None
                self.vlm_to_action_proj = None

        else:
            logger.info("Using special action tokens for prediction.")

            self.action_decoder = SimpleMLP(
                input_dim=self.action_hidden_size,
                hidden_dim=self.action_hidden_size,
                output_dim=self.unified_action_dim,
            )
            # MPG not supported without flow matching
            self.use_mpg = False
            self.mpg = None
            self.action_to_vlm_proj = None
            self.vlm_to_action_proj = None

        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self._init_weights()

    def _init_weights(self):
        for name, k in self.named_parameters():
            if any(n in name for n in ["action_decoder", "proprio_encoder_robot", "action_encoder",
                                       "action_to_vlm_proj", "vlm_to_action_proj"]):
                if "weight" in name:
                    if len(k.shape)>1:
                        nn.init.xavier_uniform_(k)
                    else:
                        nn.init.normal_(k, mean=1.0, std=0.02)
                elif "bias" in name:
                    nn.init.zeros_(k)


    @torch.no_grad()
    def get_action(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: Union[List[int], torch.Tensor],
        packed_position_ids: torch.LongTensor,
        split_lens: List[int],
        attn_modes: List[str],
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_action_indexes: Optional[torch.Tensor] = None,
        padded_state: Optional[torch.Tensor] = None,
        packed_state_indexes: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,  # (B,)
        # RTC inference parameters
        prev_chunk: Optional[torch.Tensor] = None,  # Previous action chunk for prefix conditioning
        inference_delay: int = 0,  # Number of prefix actions for RTC
        **kwargs
    ):
        self.eval()
        device = packed_text_ids.device

        packed_text_embedding = self.language_model.get_input_embeddings()(packed_text_ids)
        packed_sequence = torch.zeros(size=(sequence_length, self.hidden_size), device=device, dtype=packed_text_embedding.dtype)

        packed_sequence[packed_text_indexes] = packed_text_embedding
        vit_embeds = self.extract_feature(packed_vit_tokens)
        vit_embeds = vit_embeds.reshape(-1, self.config.llm_config.hidden_size)
        packed_sequence[packed_vit_token_indexes] = vit_embeds.to(packed_sequence.dtype)

        if padded_state is not None:
            packed_state_embeds = self.proprio_encoder_robot(padded_state.to(packed_sequence.dtype))

            if self.use_expert:
                packed_sequence_gen = torch.zeros(size=(sequence_length, self.action_hidden_size), device=device, dtype=packed_text_embedding.dtype)
                packed_sequence_gen[packed_state_indexes] = packed_state_embeds
            else:
                packed_sequence[packed_state_indexes] = packed_state_embeds

        sample_lens_list = sample_lens if isinstance(sample_lens, list) else sample_lens.tolist()
        seqlen = sum(sample_lens_list)
        B = 1

        BLOCK_SIZE = 128
        padding_len = (BLOCK_SIZE - (seqlen % BLOCK_SIZE)) % BLOCK_SIZE

        if padding_len > 0:
            # Add a dummy sample for padding
            padded_sample_lens_list = sample_lens_list + [padding_len]
            # Assume causal attention for the padded part
            padded_attn_modes = attn_modes + ['causal']
            padded_seqlen = seqlen + padding_len
        else:
            padded_sample_lens_list = sample_lens_list
            padded_attn_modes = attn_modes
            padded_seqlen = seqlen

        if sum(sample_lens_list) == sequence_length:
            sample_lens_for_llm = sample_lens_list
        else:
            sample_lens_for_llm = [sequence_length]
        attention_mask_kind = str(
            getattr(self, "worldfoundry_attention_mask_kind", None)
            or os.environ.get("WORLDFOUNDRY_BEING_H05_ATTENTION_MASK_KIND", "dense")
        ).lower()
        if attention_mask_kind == "sparse":
            sparse_mask = create_sparse_mask(
                document_lens=sample_lens_for_llm,
                split_lens=split_lens,
                attn_modes=attn_modes,
                device=device,
            )
            attention_mask_for_llm = create_block_mask(
                sparse_mask,
                B=1,
                H=self.config.llm_config.num_attention_heads,
                Q_LEN=sequence_length,
                KV_LEN=sequence_length,
                device=device,
                BLOCK_SIZE=128,
                _compile=True,
            )
        elif attention_mask_kind == "dense":
            attention_mask_for_llm = create_dense_attention_masks(
                document_lens=sample_lens_for_llm,
                split_lens=split_lens,
                attn_modes=attn_modes,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported Being-H0.5 attention mask kind: {attention_mask_kind}")

        if self.use_flow_matching:
            num_steps = self.num_inference_timesteps
            dt = 1.0 / num_steps
            action_shape = (B, self.action_chunk_length, self.unified_action_dim)
            actions = torch.randn(action_shape, device=device, dtype=packed_text_embedding.dtype)

            # =====================================================================
            # RTC Prefix Locking Setup
            # =====================================================================
            use_rtc = (
                self.config.use_training_time_rtc and
                self.config.use_inference_prefix_overwrite and
                prev_chunk is not None and
                inference_delay > 0
            )

            if use_rtc:
                # Convert prev_chunk to model dtype
                prev_chunk = prev_chunk.to(dtype=actions.dtype)

                # Create prefix mask: True for prefix positions (to be locked)
                prefix_mask = (
                    torch.arange(self.action_chunk_length, device=device)[None, :] < inference_delay
                )  # (1, action_chunk_length)

                # Pad/truncate prev_chunk to match action_chunk_length
                if prev_chunk.shape[1] < self.action_chunk_length:
                    pad_width = self.action_chunk_length - prev_chunk.shape[1]
                    prev_chunk_padded = F.pad(prev_chunk, (0, 0, 0, pad_width), mode='replicate')
                else:
                    prev_chunk_padded = prev_chunk[:, :self.action_chunk_length]

                # Initialize actions: prefix from prev_chunk, suffix from noise
                actions = torch.where(
                    prefix_mask.unsqueeze(-1),  # (1, action_chunk_length, 1)
                    prev_chunk_padded,            # (B, Chunk, A_Dim)
                    actions                       # (B, Chunk, A_Dim) - random noise
                )

            base_packed_sequence = packed_sequence.clone()
            if self.use_expert:
                base_packed_sequence_gen = packed_sequence_gen.clone()

            # =====================================================================
            # MPG Inference: Two-Stage Approach
            # Stage 1: Baseline prediction (no MPG enhancement)
            # Stage 2: Iterative refinement with MPG (if enabled)
            # =====================================================================

            # Determine number of refinement iterations
            mpg_refinement_iters = getattr(self, 'mpg_refinement_iters', 0)
            use_mpg_inference = (
                self.use_mpg and
                self.mpg is not None and
                self.mpg.lambda_strength > 0 and
                mpg_refinement_iters > 0
            )

            # Total iterations: 1 baseline + N refinement
            total_iterations = 1 + (mpg_refinement_iters if use_mpg_inference else 0)
            predicted_action_emb = None  # Will be set after first iteration for MPG

            for iteration in range(total_iterations):
                # Reset actions for each iteration (start from noise)
                actions = torch.randn(action_shape, device=device, dtype=packed_text_embedding.dtype)

                # RTC: Re-apply prefix from prev_chunk for each iteration
                if use_rtc:
                    actions = torch.where(
                        prefix_mask.unsqueeze(-1),
                        prev_chunk_padded,
                        actions
                    )

                for t_step in range(num_steps):
                    t_continuous = t_step / float(num_steps)  # Time from 0 -> 1
                    t_discretized = int(t_continuous * self.num_timestep_buckets)

                    if use_rtc:
                        # RTC: overwrite prefix and use per-token timesteps (prefix=1.0)
                        actions = torch.where(
                            prefix_mask.unsqueeze(-1),
                            prev_chunk_padded,
                            actions
                        )
                        timesteps_full = torch.full(
                            (B, self.action_chunk_length),
                            t_continuous,
                            device=device,
                            dtype=actions.dtype
                        )
                        timesteps_full = torch.where(prefix_mask, 1.0, timesteps_full)
                        actions_flat = actions.reshape(B * self.action_chunk_length, -1)
                        timesteps_flat = (timesteps_full.reshape(-1) * self.num_timestep_buckets).long()
                        action_features = self.action_encoder(actions_flat, timesteps_flat)
                        action_features = action_features.reshape(B, self.action_chunk_length, -1)
                    else:
                        timesteps_tensor = torch.full(size=(B,), fill_value=t_discretized, device=device)
                        action_features = self.action_encoder(actions, timesteps_tensor)

                    action_features_flat = action_features.reshape(B * self.action_chunk_length, -1)

                    current_packed_sequence = base_packed_sequence.clone()
                    if self.use_expert:
                        current_packed_sequence_gen = base_packed_sequence_gen.clone()
                        current_packed_sequence_gen[packed_action_indexes] = action_features_flat.to(current_packed_sequence_gen.dtype)
                    else:
                        current_packed_sequence[packed_action_indexes] = action_features_flat.to(current_packed_sequence.dtype)

                    # =====================================================================
                    # MPG Per-Step Enhancement (for refinement iterations only)
                    # =====================================================================
                    if (use_mpg_inference and
                        iteration > 0 and
                        predicted_action_emb is not None):

                        # Extract state features (VLM dimension)
                        state_features = current_packed_sequence[packed_state_indexes]  # (N_state, hidden_size)

                        # Extract action features (action dimension)
                        if self.use_expert:
                            action_features_cur = current_packed_sequence_gen[packed_action_indexes]
                        else:
                            action_features_cur = current_packed_sequence[packed_action_indexes]

                        # Project action to VLM dimension
                        action_features_proj = self.action_to_vlm_proj(action_features_cur)

                        # Concatenate suffix (state + action) in VLM dimension
                        suffix_features = torch.cat([state_features, action_features_proj], dim=0)
                        suffix_features_batched = suffix_features.unsqueeze(0)

                        # Unified enhancement using predicted CLEAN action embeddings
                        enhanced_suffix = self.mpg(
                            suffix_features_batched,
                            predicted_action_emb,  # Clean predicted actions from previous iteration
                            return_gate=False
                        )
                        enhanced_suffix = enhanced_suffix.squeeze(0)

                        # Split enhanced suffix back
                        N_state = len(packed_state_indexes)
                        enhanced_state = enhanced_suffix[:N_state]
                        enhanced_action_proj = enhanced_suffix[N_state:]

                        # Inverse project action
                        enhanced_action = self.vlm_to_action_proj(enhanced_action_proj)

                        # Update packed sequences with enhanced features
                        current_packed_sequence[packed_state_indexes] = enhanced_state.to(current_packed_sequence.dtype)
                        if self.use_expert:
                            current_packed_sequence_gen[packed_action_indexes] = enhanced_action.to(current_packed_sequence_gen.dtype)
                        else:
                            current_packed_sequence[packed_action_indexes] = enhanced_action.to(current_packed_sequence.dtype)

                    # Prepare token indexes for LLM
                    if self.use_expert:
                        packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
                        packed_gen_token_indexes = torch.cat([packed_state_indexes, packed_action_indexes], dim=0)
                        current_packed_sequence_und = current_packed_sequence[packed_und_token_indexes]
                        current_packed_sequence_gen_slice = current_packed_sequence_gen[packed_gen_token_indexes]
                    else:
                        packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes, packed_state_indexes, packed_action_indexes], dim=0)
                        packed_gen_token_indexes = torch.tensor([], dtype=torch.long, device=device)
                        current_packed_sequence_und = current_packed_sequence[packed_und_token_indexes]
                        current_packed_sequence_gen_slice = torch.tensor([], dtype=packed_text_embedding.dtype, device=device).view(0, self.action_hidden_size)

                    extra_inputs = {
                        "packed_und_token_indexes": packed_und_token_indexes,
                        "packed_gen_token_indexes": packed_gen_token_indexes
                    }
                    hidden_states_und, hidden_states_gen = self.language_model.forward_train(
                        packed_sequence_und=current_packed_sequence_und, packed_sequence_gen=current_packed_sequence_gen_slice,
                        sample_lens=sample_lens_for_llm, attention_mask=attention_mask_for_llm,
                        packed_position_ids=packed_position_ids, **extra_inputs,
                    )

                    if self.use_expert:
                        last_hidden_state_act = hidden_states_gen[len(packed_state_indexes):]
                    else:
                        start_idx = len(packed_text_indexes) + len(packed_vit_token_indexes) + len(packed_state_indexes)
                        last_hidden_state_act = hidden_states_und[start_idx:]

                    pred_velocity = self.action_decoder(
                        last_hidden_state_act.reshape(B, self.action_chunk_length, -1)
                    )

                    actions = actions + dt * pred_velocity

                    # =====================================================================
                    # RTC Prefix Locking After Each Step
                    # =====================================================================
                    if use_rtc:
                        # Re-apply prefix from prev_chunk to keep it locked
                        actions = torch.where(
                            prefix_mask.unsqueeze(-1),
                            prev_chunk_padded,
                            actions
                        )

                # After each iteration, encode predicted actions as CLEAN embeddings for next iteration
                if use_mpg_inference and iteration < total_iterations - 1:
                    t_clean = torch.zeros(B, dtype=torch.long, device=device)
                    predicted_action_emb = self.action_encoder(actions, t_clean)
                    # Shape: (B, action_chunk_length, action_hidden_size)

            predicted_actions = actions
            predicted_actions = predicted_actions.reshape(B * self.action_chunk_length, -1)

        else:
            if self.use_expert:
                # packed_sequence_gen = torch.zeros(size=(sequence_length, self.action_hidden_size), device=device, dtype=packed_text_embedding.dtype)
                # packed_sequence_gen[packed_state_indexes] = packed_state_embeds

                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
                packed_sequence_und = packed_sequence[packed_und_token_indexes]

                packed_gen_token_indexes = torch.cat([packed_state_indexes, packed_action_indexes], dim=0)
                packed_sequence_gen = packed_sequence_gen[packed_gen_token_indexes]
            else:
                # packed_sequence[packed_state_indexes] = packed_state_embeds

                packed_und_token_indexes = torch.cat([
                    packed_text_indexes, packed_vit_token_indexes, packed_state_indexes, packed_action_indexes
                ], dim=0)
                packed_sequence_und = packed_sequence[packed_und_token_indexes]
                packed_gen_token_indexes = torch.tensor([], dtype=torch.long, device=device)
                packed_sequence_gen = torch.tensor([], dtype=packed_sequence_und.dtype, device=device).view(0, self.action_hidden_size)

            extra_inputs = {
                "packed_und_token_indexes": packed_und_token_indexes,
                "packed_gen_token_indexes": packed_gen_token_indexes,
            }

            hidden_states_und, hidden_states_gen = self.language_model.forward_train(
                packed_sequence_und=packed_sequence_und,
                packed_sequence_gen=packed_sequence_gen,
                sample_lens=sample_lens_list,
                attention_mask=attention_mask_for_llm,
                packed_position_ids=packed_position_ids,
                **extra_inputs,
            )

            if self.use_expert:
                start_index_for_action = len(packed_state_indexes) if packed_state_indexes is not None else 0
                last_hidden_state_act = hidden_states_gen[start_index_for_action:]
            else:
                len_text = len(packed_text_indexes)
                len_vit = len(packed_vit_token_indexes)
                len_state = len(packed_state_indexes)
                start_index_for_action = len_text + len_vit + len_state
                last_hidden_state_act = hidden_states_und[start_index_for_action:]

            # predicted_actions = self.action_decoder(last_hidden_state_act)

            predicted_actions = self.action_decoder(
                    last_hidden_state_act.reshape(B, self.action_chunk_length, -1)
                )
            predicted_actions = predicted_actions.reshape(B * self.action_chunk_length, -1)

        return {"action_pred": predicted_actions}

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vit_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vit_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.connector(vit_embeds)
        return vit_embeds

    @property
    def lm_head(self):
        # for models like InternVL, lm_head is a function of language_model
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a pretrained BeingVLA model with comprehensive checkpoint support.

        This method handles:
        1. Sharded safetensors checkpoints (model-00001-of-00004.safetensors)
        2. Single safetensors files (model.safetensors)
        3. PyTorch checkpoints (pytorch_model.bin, pytorch_model.pt)
        4. Original InternVL models with automatic format conversion
        """

        config = kwargs.pop('config', None)
        if config is None:
            # Load config manually to ensure compatibility
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs
            )

        CustomForCausalLM  = LLM_MODEL_ARCH[config.llm_config.architectures[0]][1]
        language_model = CustomForCausalLM(config.llm_config)

        CustomViTConfig, CustomViTModel = VIT_MODEL_ARCH[config.vit_config.architectures[0]]
        #config.vit_config = CustomViTConfig(**config.vit_config)
        vit_model = CustomViTModel(config.vit_config)
        config.connector_arch = "internvl_connector"
        connector = CONNECTOR_ARCH[config.connector_arch](
            llm_hidden_size=config.llm_config.hidden_size,
            vit_hidden_size=config.vit_config.hidden_size,
            downsample_ratio=config.downsample_ratio,
        )

        # Create the model instance with our custom init parameters
        model = cls(language_model, vit_model, connector, config)

        state_dict = None
        if os.path.isdir(pretrained_model_name_or_path):
            index_file = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
            if os.path.exists(index_file):
                # Load sharded safetensors model
                from safetensors.torch import load_file
                import json

                print(f"Loading sharded model from {pretrained_model_name_or_path}")
                with open(index_file, 'r') as f:
                    index = json.load(f)

                # Load all unique shard files
                shard_files = set(index['weight_map'].values())
                state_dict = {}

                for shard_file in sorted(shard_files):
                    shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                    print(f"Loading shard: {shard_file}")
                    shard_dict = load_file(shard_path)
                    state_dict.update(shard_dict)
            else:
                # Check for single checkpoint files
                for filename in ['pytorch_model.bin', 'model.safetensors', 'pytorch_model.pt']:
                    candidate = os.path.join(pretrained_model_name_or_path, filename)
                    if os.path.exists(candidate):
                        # Load state dict based on file format
                        if candidate.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            state_dict = load_file(candidate)
                        else:
                            from worldfoundry.core.model_loading.checkpoints.file import load_torch_checkpoint

                            state_dict = load_torch_checkpoint(
                                candidate,
                                map_location='cpu',
                                weights_only=True,
                            )
                        break

        if state_dict is not None:
            if not isinstance(state_dict, dict) or not all(
                isinstance(key, str) and torch.is_tensor(value)
                for key, value in state_dict.items()
            ):
                raise TypeError("Being-H0.5 checkpoint must contain only named tensors")
            model.load_state_dict(state_dict, strict=True)
            print(f"Loaded BeingH model from {pretrained_model_name_or_path}")
        else:
            raise FileNotFoundError(
                f"No supported Being-H0.5 checkpoint found at {pretrained_model_name_or_path}"
            )

        # Handle dtype conversion if specified
        torch_dtype = kwargs.get('torch_dtype', None)
        if torch_dtype is not None:
            model = model.to(torch_dtype)

        return model
