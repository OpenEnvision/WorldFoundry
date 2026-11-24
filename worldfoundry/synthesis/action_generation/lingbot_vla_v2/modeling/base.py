"""Shared checkpoint-compatible inference layers for LingBot-VLA v2."""

from typing import ClassVar

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.cache_utils import Cache
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
)
from transformers.utils import logging

from worldfoundry.base_models.llm_mllm_core.action.lingbot.modeling import AdaRMSNorm

from .qwen25_vl import Qwen2_5_VLForConditionalGeneration

try:  # noqa: SIM105 - The handler carries an explanatory comment.
    from dinov3.hub.backbones import (
        dinov3_vitb16,
        dinov3_vits16,  # noqa: F401 - Import checks optional dependency availability.
        dinov3_vits16plus,  # noqa: F401 - Import checks optional dependency availability.
    )
except BaseException:
    pass
from worldfoundry.base_models.llm_mllm_core.action.lingbot.modeling import (
    TaskTokenDepthHead,
)
from worldfoundry.base_models.llm_mllm_core.action.tensor_ops import (
    apply_rope,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from worldfoundry.base_models.llm_mllm_core.action.tensor_ops import (
    masked_eager_attention_forward as our_eager_attention_forward,
)

from .attention import (
    build_block_mask,
    flex_attention_forward,
    flex_attention_with_block_mask,
)
from .qwen2_expert import (
    FixQwen2RMSNorm,
    Qwen2ForCausalLM,
    Qwen2TokenMoeBlock,
)

logger = logging.get_logger(__name__)


class QwenvlWithExpertConfig(PretrainedConfig):
    model_type = "QwenvlWithExpertModel"
    sub_configs: ClassVar[dict] = {"qwenvl_config": AutoConfig, "qwen_expert_config": AutoConfig}

    def __init__(
        self,
        qwenvl_config: dict | None = None,
        qwen_expert_config: dict | None = None,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        vocab_size: int = 257152,
        use_lm_head: bool = False,
        attention_implementation: str = "eager",
        tokenizer_path: str | None = None,
        enable_expert_vision: bool = False,
        expert_vision_type: str | None = None,
        use_cache: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.tokenizer_path = tokenizer_path
        self.enable_expert_vision = enable_expert_vision
        self.expert_vision_type = expert_vision_type
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        if qwenvl_config is None:
            self.qwenvl_config = CONFIG_MAPPING["qwen2_5_vl"](
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151645,
                vision_start_token_id=151652,
                vision_end_token_id=151653,
                vision_token_id=151654,
                image_token_id=151655,
                video_token_id=151656,
                hidden_act="silu",
                hidden_size=2048,
                initializer_range=0.02,
                intermediate_size=11008,
                max_position_embeddings=128000,
                max_window_layers=70,
                model_type="qwen2_5_vl",
                num_attention_heads=16,
                num_hidden_layers=36,
                num_key_value_heads=2,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=32768,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                transformers_version="4.41.2",
                use_cache=True,
                use_sliding_window=False,
                vision_config={
                    "depth": 32,
                    "hidden_act": "silu",
                    "hidden_size": 1280,
                    "intermediate_size": 3420,
                    "num_heads": 16,
                    "in_chans": 3,
                    "out_hidden_size": 2048,
                    "patch_size": 14,
                    "spatial_merge_size": 2,
                    "spatial_patch_size": 14,
                    "window_size": 112,
                    "fullatt_block_indexes": [7, 15, 23, 31],
                    "tokens_per_second": 2,
                    "temporal_patch_size": 2,
                },
                rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
                vocab_size=151936,
            )
        elif isinstance(self.qwenvl_config, dict):
            if "model_type" not in qwen_expert_config:
                qwenvl_config["model_type"] = "qwen2_5_vl"

            cfg_cls = CONFIG_MAPPING[qwenvl_config["model_type"]]
            self.qwenvl_config = cfg_cls(**qwenvl_config)

        if qwen_expert_config is None:
            self.qwen_expert_config = CONFIG_MAPPING["qwen2"](
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151645,
                hidden_act="silu",
                hidden_size=expert_hidden_size,
                head_dim=128,
                initializer_range=0.02,
                intermediate_size=expert_intermediate_size,
                max_position_embeddings=32768,
                max_window_layers=21,
                model_type="qwen2",
                num_attention_heads=16,
                num_hidden_layers=36,
                num_key_value_heads=2,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=32768,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                transformers_version="4.43.1",
                use_cache=use_cache,
                use_sliding_window=False,
                vocab_size=151936,
            )
        elif isinstance(self.qwen_expert_config, dict):
            if "model_type" not in qwen_expert_config:
                qwen_expert_config["model_type"] = "qwen2"

            cfg_cls = CONFIG_MAPPING[qwenvl_config["model_type"]]
            self.qwen_expert_config = cfg_cls(**qwen_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )


class FixAdaRMSNorm(AdaRMSNorm):
    """Keep the fixed-condition path's float32 Linear inputs."""

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return super().forward(hidden_states, cond.float())


# HACK: show directly use this norm during initialization
# TODO: clear the logics
def replace_lnorm_with_adanorm(module, hidden_size, cond_dim, final_norm_adanorm):
    for name, child in module.named_children():
        if final_norm_adanorm:
            if isinstance(child, Qwen2RMSNorm):
                if "q_layernorm" not in name and "k_layernorm" not in name:
                    setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
            elif isinstance(child, FixQwen2RMSNorm):
                if "q_layernorm" not in name and "k_layernorm" not in name:
                    setattr(module, name, FixAdaRMSNorm(hidden_size, cond_dim))
            else:
                replace_lnorm_with_adanorm(child, hidden_size, cond_dim, final_norm_adanorm)
        else:
            if isinstance(child, Qwen2RMSNorm):
                if "q_layernorm" not in name and "k_layernorm" not in name:
                    setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
            else:
                replace_lnorm_with_adanorm(child, hidden_size, cond_dim, final_norm_adanorm)


class QwenvlWithExpertModel(PreTrainedModel):
    config_class = QwenvlWithExpertConfig

    def __init__(self, config: QwenvlWithExpertConfig, eval=False):
        super().__init__(config=config)
        self.config = config
        vlm_config = AutoConfig.from_pretrained(
            self.config.tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        vlm_config.vision_config.initializer_range = 0.02
        print(f"=====Vocab_size in Config is {self.config.vocab_size}=====")
        if (
            self.config.vocab_size != 0
            and self.config.vocab_size != 257152
            and vlm_config.vocab_size != self.config.vocab_size
        ):
            vlm_config.vocab_size = self.config.vocab_size
        print(f"====Vocabulary Size is {vlm_config.vocab_size}====")
        vlm_config._attn_implementation = "flash_attention_2"
        vlm_config.vision_config._attn_implementation = self.config.vit_attn_implementation
        self.qwenvl = Qwen2_5_VLForConditionalGeneration._from_config(vlm_config)
        if self.config.use_lm_head:
            self.qwenvl.tie_weights()
        self.config.qwen_expert_config._attn_implementation = "flash_attention_2"
        self.qwen_expert = Qwen2ForCausalLM._from_config(self.config.qwen_expert_config, eval=eval)

        if getattr(self.config, "adanorm_time", False):
            replace_lnorm_with_adanorm(
                self.qwen_expert,
                self.config.qwen_expert_config.hidden_size,
                self.config.qwen_expert_config.hidden_size,
                config.final_norm_adanorm,
            )
        if getattr(self.config, "use_moe", False):
            bias_update_speed = getattr(self.config, "bias_update_speed", 0.001)
            hidden_size = self.config.qwen_expert_config.hidden_size  # 768

            token_moe_layers = getattr(self.config, "token_moe_layers", None) or []

            if token_moe_layers:
                token_config = CONFIG_MAPPING["qwen2_moe"](
                    num_experts=getattr(self.config, "token_num_experts", 32),
                    num_experts_per_tok=getattr(self.config, "token_top_k", 1),
                    norm_topk_prob=True,
                    hidden_size=hidden_size,
                    moe_intermediate_size=getattr(self.config, "token_moe_intermediate_size", 256),
                    shared_expert_intermediate_size=getattr(self.config, "token_shared_intermediate_size", 256),
                    output_router_logits=False,
                )
                token_config.bias_update_speed = bias_update_speed
                token_config._moe_implementation = getattr(self.config, "_moe_implementation", None)
                token_config.router_activation = getattr(self.config, "router_activation", "softmax")
                token_config.routed_scaling_factor = getattr(self.config, "routed_scaling_factor", 1.0)
                token_config.use_shared_expert_gate = getattr(self.config, "use_shared_expert_gate", True)
                for idx in token_moe_layers:
                    self.qwen_expert.model.layers[idx].mlp = Qwen2TokenMoeBlock(token_config)
        # Precomputed grid_thw cache (populated on first call when precompute_grid_thw=True)
        self.rotary_pos_emb = None
        self.window_index = None
        self.cu_window_seqlens = None
        self.cu_seqlens = None

        # Remove unused embed_tokens
        del self.qwen_expert.model.embed_tokens
        if self.config.enable_expert_vision:
            if "dinov3_vitb16" in self.config.expert_vision_type:
                self.expert_visual = dinov3_vitb16(pretrained=False)
            self.expert_visual_mlp = nn.Sequential(
                nn.Linear(self.expert_visual.embed_dim, self.expert_visual.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.expert_visual.embed_dim * 2, self.config.qwen_expert_config.hidden_size),
            )
        self.attention_interface = self.get_attention_interface()

        # self.to_bfloat16_like_physical_intelligence()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        precompute_grid_thw: bool = False,
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            precompute_grid_thw (`bool`): If True, compute and cache rotary_pos_emb/window_index/cu_seqlens on first call.
        """
        if precompute_grid_thw and self.rotary_pos_emb is None:
            (self.rotary_pos_emb, self.window_index, self.cu_window_seqlens, self.cu_seqlens) = (
                self.qwenvl.visual.preprcess_grid_thw(grid_thw=image_grid_thw)
            )
        image_embeds = self.qwenvl.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            rotary_pos_emb=self.rotary_pos_emb,
            window_index=self.window_index,
            cu_window_seqlens=self.cu_window_seqlens,
            cu_seqlens=self.cu_seqlens,
        )
        split_sizes = (image_grid_thw.prod(-1) // self.qwenvl.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        image_embeds = torch.stack(image_embeds, dim=0)
        return image_embeds

    def embed_image(self, image: torch.Tensor, patch_size=14, temporal_patch_size=2, precompute_grid_thw=False):
        h = w = int(image.shape[1] ** 0.5)
        image_grid_thw = torch.tensor([[1, h, w]] * image.shape[0], device=image.device)
        image_embeds = self.get_image_features(
            image, image_grid_thw=image_grid_thw, precompute_grid_thw=precompute_grid_thw
        )
        return image_embeds
        # return torch.randn(72, 64, 2048).to(device=image.device, dtype=torch.bfloat16)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwenvl.model.embed_tokens(tokens)

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}

            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat(
                    [past_key_values[layer_idx]["value_states"], value_states],
                    dim=1,
                )
        return key_states, value_states, past_key_values

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        vlm_position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        ada_cond: list[torch.FloatTensor] | None = None,
    ):
        """
        Args:
            attention_mask (Optional[torch.Tensor], optional):
                Attention mask with shape (b, seq_len, seq_len). Defaults to None.
            position_ids (Optional[torch.LongTensor], optional):
                Position indices for applying RoPE. Defaults to None.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]], optional):
                Optional kv cache. Defaults to None.
            inputs_embeds (List[torch.FloatTensor], optional):
                Input embeddings. Defaults to None.
            use_cache (Optional[bool], optional):
                Whether to use kv cache. Defaults to None.
            fill_kv_cache (Optional[bool], optional):
                Whether to return kv tensors in this forward pass as cache. Defaults to None.

        Returns:
            outputs_embeds (torch.Tensor): Output embeddings.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]]):
                Optional kv cache.
        """
        models = [
            self.qwenvl.model,
            self.qwen_expert.model,
        ]  # Qwen2_5_VLTextModel, Qwen2Model (We have re-writeen their forward as follows:)

        # RMSNorm
        num_layers = self.qwenvl.config.num_hidden_layers  # 36
        action_num_layers = self.config.qwen_expert_config.num_hidden_layers  # 36
        assert action_num_layers == num_layers, (
            "Action expert and VLM must have the same number of layers "
            f"(got action={action_num_layers}, vlm={num_layers})."
        )

        router_logits_list = []
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                if i == 1:  # For action expert
                    query_state, key_state, value_state = models[i].layers[layer_idx](
                        hidden_states, compute_kqv=True, ada_cond=ada_cond
                    )
                else:  # For VLM
                    query_state, key_state, value_state = models[i].layers[layer_idx](hidden_states, compute_kqv=True)

                if query_state.dtype != torch.float32:
                    query_state, key_state, value_state = (
                        query_state.to(torch.float32),
                        key_state.to(torch.float32),
                        value_state.to(torch.float32),
                    )
                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length (img, lang, state, action), H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            key_states, value_states, past_key_values = self.handle_kv_cache(
                key_states,
                value_states,
                layer_idx,
                past_key_values=past_key_values,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
            )
            if self.config.attention_implementation == "flex_cached":
                if layer_idx == 0:
                    _full_len = query_states.shape[1]
                    _full_block_mask = build_block_mask(
                        attention_mask, self.qwenvl.config.num_attention_heads, _full_len, _full_len
                    )
                att_output = flex_attention_with_block_mask(
                    query_states, key_states, value_states, _full_block_mask, query_states.shape[1]
                )
            else:
                att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is not None:
                    end = start + hidden_states.shape[1]
                    if i == 1:
                        out_emb, _router_logits = models[i].layers[layer_idx](
                            hidden_states, att_output, start, end, output_atten=True, ada_cond=ada_cond
                        )
                        if _router_logits is not None:
                            router_logits_list.append(_router_logits)
                    else:
                        out_emb = models[i].layers[layer_idx](hidden_states, att_output, start, end, output_atten=True)
                    outputs_embeds.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                if self.config.final_norm_adanorm:
                    if i == 1:
                        out_emb, _ = models[i].norm(hidden_states, ada_cond)
                    else:
                        out_emb = models[i].norm(hidden_states)
                else:
                    out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values, router_logits_list

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            raise NotImplementedError("FA2 is not implemented (yet)")
        elif self.config.attention_implementation == "flex":
            print("=====Using Flex Attn=====")
            attention_interface = flex_attention_forward
        elif self.config.attention_implementation == "eager":
            print("=====Using Eager Attn=====")
            attention_interface = our_eager_attention_forward
        elif self.config.attention_implementation == "flex_cached":
            print("=====Using Flex Cached (prebuilt BlockMask) Attn=====")
            attention_interface = flex_attention_forward  # fallback
        elif self.config.attention_implementation == "xformer":
            # attention_interface = xformer_attention_forward
            raise NotImplementedError("Xformer attention is not implemented (yet)")
        else:
            raise ValueError(
                f"Invalid attention implementation: {self.config.attention_implementation}. "
                "Expected one of ['fa2', 'flex', 'flex_cached', 'eager', 'xformer']."
            )
        return attention_interface


class FlowMatching(nn.Module):
    def __init__(self, config, eval):
        super().__init__()
        self.config = config

        # qwenvl with action expert
        qwenvl_with_export_config = QwenvlWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            vocab_size=getattr(self.config, "vocab_size", 0),
            use_lm_head=getattr(self.config, "use_lm_head", False),
            attention_implementation=self.config.attention_implementation,
            tokenizer_path=self.config.tokenizer_path,
            enable_expert_vision=self.config.enable_expert_vision,
            expert_vision_type=self.config.expert_vision_type,
            use_cache=getattr(self.config, "use_cache", True),
            expert_hidden_size=getattr(self.config, "expert_hidden_size", 768),
            expert_intermediate_size=getattr(self.config, "expert_intermediate_size", 2752),
        )
        qwenvl_with_export_config.adanorm_time = getattr(config, "adanorm_time", False)
        qwenvl_with_export_config.final_norm_adanorm = getattr(config, "final_norm_adanorm", False)
        qwenvl_with_export_config.vit_attn_implementation = getattr(
            config, "vit_attn_implementation", "flash_attention_2"
        )
        if getattr(config, "use_moe", False):
            qwenvl_with_export_config.use_moe = config.use_moe
            qwenvl_with_export_config.bias_update_speed = getattr(config, "bias_update_speed", 0.001)
            qwenvl_with_export_config.token_moe_layers = getattr(config, "token_moe_layers", None)
            qwenvl_with_export_config.token_num_experts = getattr(config, "token_num_experts", 32)
            qwenvl_with_export_config.token_top_k = getattr(config, "token_top_k", 1)
            qwenvl_with_export_config.token_moe_intermediate_size = getattr(config, "token_moe_intermediate_size", 256)
            qwenvl_with_export_config.token_shared_intermediate_size = getattr(
                config, "token_shared_intermediate_size", 256
            )
        # Pass _moe_implementation through for EP/fused support
        qwenvl_with_export_config._moe_implementation = getattr(config, "_moe_implementation", None)
        self.qwenvl_with_expert = QwenvlWithExpertModel(qwenvl_with_export_config, eval)
        self.config.proj_width = qwenvl_with_export_config.qwen_expert_config.hidden_size
        self.config.initializer_range = getattr(qwenvl_with_export_config.qwen_expert_config, "initializer_range", None)
        # projection layers
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)
        self.config.align_params = getattr(self.config, "align_params", {})
        if self.config.align_params != {}:
            self.steps = 0
            self.use_depth_align = True
            self.init_depth_heads(self.config.align_params)
            self.use_future_video = self.config.align_params.get("use_future_video", False)
            if self.use_future_video:
                self.init_video_heads(self.config.align_params)
        else:
            self.use_depth_align = False
            self.use_future_video = False
            self.use_future_video_patch = False
            self.use_current_video_patch = False
            self.use_current_shared_task_proj = False
            self.use_future_video_cls = False
            self.use_shared_future_task_proj = False
            self.future_video_share_future_depth_query = False

    def init_depth_heads(self, config):
        self.llm_image_token_size = config["llm"]["image_token_size"]
        self.llm_image_input_size = config["llm"]["image_input_size"]
        self.depth_token_size = config["depth"]["token_size"]
        self.depth_input_size = config["depth"]["input_size"]
        self.align_type = config.get("mode", None)
        self.model_type = config["depth"]["model_type"]
        if self.align_type != "query":
            raise ValueError(f"Only query depth alignment is supported, got {self.align_type!r}.")
        if self.model_type != "MoRGBD":
            raise ValueError(f"Only MoRGBD depth distillation is supported, got {self.model_type!r}.")
        self.use_future_depth = (config.get("depth") or {}).get("use_future_depth", False)
        self.block_future_depth_to_action = (config.get("depth") or {}).get("block_future_depth_to_action", False)
        self.detach_future_depth_image_feats = bool((config.get("depth") or {}).get("detach_future_image_feats", False))
        self.use_future_video = bool(config.get("use_future_video", False))
        self.use_future_video_patch = False
        self.use_current_video_patch = False
        self.use_current_shared_task_proj = False
        self.use_future_video_cls = False
        self.use_shared_future_task_proj = False
        self.future_video_share_future_depth_query = False
        self.num_task_tokens = config["num_task_tokens"]
        assert config["depth"]["num_backbone_tokens"] % self.num_task_tokens == 0
        self.depth_align_embs = nn.Parameter(
            torch.randn(config["depth"]["num_backbone_tokens"], config["llm"]["dim_out"])
        )
        self.depth_align_embs.requires_grad = True

        self.depth_align_head = TaskTokenDepthHead(config["depth"], llm_hidden_size=config["llm"]["dim_out"]).to(
            dtype=torch.bfloat16
        )

        for p in self.depth_align_head.parameters():
            p.requires_grad = True

        if self.use_future_depth:
            self.future_depth_align_embs = nn.Parameter(
                torch.randn(config["depth"]["num_backbone_tokens"], config["llm"]["dim_out"])
            )
            self.future_depth_align_embs.requires_grad = True

            self.future_depth_align_head = TaskTokenDepthHead(
                config["depth"], llm_hidden_size=config["llm"]["dim_out"]
            ).to(dtype=torch.bfloat16)

            for p in self.future_depth_align_head.parameters():
                p.requires_grad = True

    def init_video_heads(self, config):
        if self.align_type != "query":
            raise ValueError("future-video alignment is only supported for query align mode.")

        video_config = dict(config.get("depth", {}))
        video_config.update(config.get("video", {}))
        required_keys = ("num_backbone_tokens", "dim_out", "num_layers", "num_heads", "dim_head", "ff_mult")
        missing = [key for key in required_keys if key not in video_config]
        if missing:
            raise ValueError(f"video align config missing required keys: {missing}")
        self.use_future_video_patch = bool(video_config.get("use_patch_loss", True))
        self.use_current_video_patch = bool(video_config.get("use_current_patch_loss", False))
        if self.use_current_video_patch and not self.use_future_video_patch:
            raise ValueError(
                "align_params.video.use_current_patch_loss=True requires align_params.video.use_patch_loss=True."
            )
        self.use_current_shared_task_proj = bool(
            video_config.get("use_current_shared_task_proj", self.use_current_video_patch)
        )
        if self.use_current_shared_task_proj and not self.use_current_video_patch:
            raise ValueError(
                "align_params.video.use_current_shared_task_proj=True requires "
                "align_params.video.use_current_patch_loss=True."
            )
        self.use_future_video_cls = bool(video_config.get("use_cls_loss", False))
        self.future_video_share_future_depth_query = bool(video_config.get("share_future_depth_query", False))
        self.use_shared_future_task_proj = bool(video_config.get("use_shared_future_task_proj", False))
        if self.use_shared_future_task_proj and not self.use_future_video_patch:
            raise ValueError(
                "align_params.video.use_shared_future_task_proj=True requires align_params.video.use_patch_loss=True."
            )
        if self.use_shared_future_task_proj and not self.future_video_share_future_depth_query:
            raise ValueError(
                "align_params.video.use_shared_future_task_proj=True requires "
                "align_params.video.share_future_depth_query=True."
            )
        if self.future_video_share_future_depth_query:
            if not self.use_future_depth:
                raise ValueError(
                    "align_params.video.share_future_depth_query=True requires "
                    "align_params.depth.use_future_depth=True."
                )
            if int(video_config["num_backbone_tokens"]) != int(config["depth"]["num_backbone_tokens"]):
                raise ValueError(
                    "future-video shared query requires video.num_backbone_tokens to match depth.num_backbone_tokens."
                )

        self.block_suffix_to_future_video = bool(video_config.get("block_suffix_to_future_video", False))
        self.future_video_context_mode = str(video_config.get("context_mode", "img_query")).lower()
        if self.future_video_context_mode not in ("img_query", "query_only"):
            raise ValueError(
                "future-video context_mode must be 'img_query' or 'query_only', "
                f"got {self.future_video_context_mode!r}."
            )
        if self.use_future_video_patch:
            if self.use_current_video_patch:
                self.current_video_align_embs = nn.Parameter(
                    torch.randn(video_config["num_backbone_tokens"], config["llm"]["dim_out"])
                )
                self.current_video_align_embs.requires_grad = True
                if self.use_current_shared_task_proj:
                    self.current_shared_task_proj = nn.Linear(
                        config["llm"]["dim_out"] * 2,
                        config["llm"]["dim_out"],
                    )
                    for p in self.current_shared_task_proj.parameters():
                        p.requires_grad = True
                self.current_video_align_head = TaskTokenDepthHead(
                    video_config, llm_hidden_size=config["llm"]["dim_out"]
                ).to(dtype=torch.bfloat16)
                for p in self.current_video_align_head.parameters():
                    p.requires_grad = True

            if not self.future_video_share_future_depth_query or self.use_shared_future_task_proj:
                self.future_video_align_embs = nn.Parameter(
                    torch.randn(video_config["num_backbone_tokens"], config["llm"]["dim_out"])
                )
                self.future_video_align_embs.requires_grad = True
            if self.use_shared_future_task_proj:
                self.future_shared_task_proj = nn.Linear(
                    config["llm"]["dim_out"] * 2,
                    config["llm"]["dim_out"],
                )
                for p in self.future_shared_task_proj.parameters():
                    p.requires_grad = True
            self.future_video_align_head = TaskTokenDepthHead(
                video_config, llm_hidden_size=config["llm"]["dim_out"]
            ).to(dtype=torch.bfloat16)
            for p in self.future_video_align_head.parameters():
                p.requires_grad = True

        if self.use_future_video_cls:
            self.future_video_cls_align_emb = nn.Embedding(1, config["llm"]["dim_out"])
            self.future_video_cls_head = nn.Sequential(
                nn.LayerNorm(config["llm"]["dim_out"]),
                nn.Linear(config["llm"]["dim_out"], video_config["dim_out"]),
            ).to(dtype=torch.bfloat16)
            for p in self.future_video_cls_head.parameters():
                p.requires_grad = True

    def _future_depth_token_count(self):
        return self.num_task_tokens if getattr(self, "use_future_depth", False) else 0

    def _future_video_own_token_count(self):
        if not getattr(self, "use_future_video", False):
            return 0
        count = 1 if getattr(self, "use_future_video_cls", False) else 0
        if getattr(self, "use_future_video_patch", True) and not getattr(
            self, "future_video_share_future_depth_query", False
        ):
            count += self.num_task_tokens
        return count

    def _future_video_own_span(self, hidden_states):
        own_count = self._future_video_own_token_count()
        future_depth_count = self._future_depth_token_count()
        end = hidden_states.shape[1] - future_depth_count
        start = end - own_count
        return start, end

    def _future_depth_task_tokens(self, hidden_states):
        if not getattr(self, "use_future_depth", False):
            raise ValueError("future-depth query tokens are not enabled.")
        return hidden_states[:, -self.num_task_tokens :, :]

    def _future_video_cls_task_tokens(self, hidden_states):
        if not getattr(self, "use_future_video_cls", False):
            return None
        start, _ = self._future_video_own_span(hidden_states)
        return hidden_states[:, start : start + 1, :]

    def _future_video_patch_task_tokens(self, hidden_states):
        if getattr(self, "future_video_share_future_depth_query", False):
            return self._future_depth_task_tokens(hidden_states)
        start, end = self._future_video_own_span(hidden_states)
        if getattr(self, "use_future_video_cls", False):
            start += 1
        return hidden_states[:, start:end, :]

    def _current_depth_task_tokens(self, hidden_states, num_images=3):
        chunk_size = self.llm_image_token_size * self.llm_image_token_size
        image_token_len = chunk_size + (2 if getattr(self.config, "qwen3vl_use_vision_boundaries", False) else 0)
        if getattr(self, "use_future_depth", False):
            start = num_images * image_token_len
            return hidden_states[:, start : start + self.num_task_tokens, :]
        end = hidden_states.shape[1] - self._future_video_own_token_count()
        start = end - self.num_task_tokens
        return hidden_states[:, start:end, :]

    def _future_video_query_span(self, prefix_len):
        if not getattr(self, "use_future_video", False):
            return prefix_len, prefix_len
        future_depth_count = self._future_depth_token_count()
        own_count = self._future_video_own_token_count()
        end = prefix_len - future_depth_count
        return end - own_count, end

    def _block_suffix_to_future_video_(self, att_2d_masks, suffix_row_start, prefix_len):
        start, end = self._future_video_query_span(prefix_len)
        if end <= start:
            return att_2d_masks
        att_2d_masks[:, suffix_row_start:, start:end] = False
        return att_2d_masks

    def _block_suffix_to_future_video_if_enabled_(
        self,
        att_2d_masks,
        suffix_row_start,
        prefix_len,
    ):
        if not getattr(self, "block_suffix_to_future_video", False):
            return att_2d_masks
        return self._block_suffix_to_future_video_(
            att_2d_masks,
            suffix_row_start=suffix_row_start,
            prefix_len=prefix_len,
        )

    @staticmethod
    def _fp32_linear(module, x):
        """Compute linear layer in fp32 regardless of module's current parameter dtype."""
        return F.linear(x.float(), module.weight.float(), module.bias.float() if module.bias is not None else None)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, vlm_causal, precompute_grid_thw=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsize = images.shape[0]
        device = images.device
        _dtype = images.dtype

        # embed image
        if images.ndim == 5:
            images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        elif images.ndim == 4:
            images = einops.rearrange(images, "b n l d -> (b n) l d")
        elif images.ndim == 3:  # For inference bs=1
            bsize = 1
        img_emb = self.qwenvl_with_expert.embed_image(images, precompute_grid_thw=precompute_grid_thw)
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b (n l) d", b=bsize)  # bsize = 24
        num_img_embs = img_emb.shape[1]
        if img_masks.ndim == 1:  # For inference bs=1
            img_masks = img_masks.unsqueeze(0)
        if self.use_depth_align and self.align_type == "query":
            align_masks = einops.repeat(img_masks, "b n -> b (n l)", l=self.num_task_tokens)
        img_masks = einops.repeat(img_masks, "b n -> b (n l)", l=num_patch)

        # embed language
        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens)
        num_lang_embs = lang_emb.shape[1]

        if self.use_depth_align and self.align_type == "query":

            def _get_align_tokens(tokens):
                tk_weights = tokens.view(self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1])
                tk_weights = tk_weights.mean(dim=1)
                return tk_weights

            align_embs = (
                _get_align_tokens(self.depth_align_embs).repeat(img_emb.size(0), 1, 1).to(img_emb.device, img_emb.dtype)
            )
            # align_masks = einops.rearrange(img_masks, "b (n l) -> b n l", n=3)
            # align_masks = align_masks[:, :, 0]
            # align_masks = einops.repeat(align_masks, "b n -> b (n l)", l=self.num_task_tokens)
            embs = torch.cat([img_emb, align_embs, align_embs, align_embs, lang_emb], dim=1)
            pad_masks = torch.cat([img_masks, align_masks, lang_masks], dim=1)
        else:
            # assemble embeddings
            embs = torch.cat([img_emb, lang_emb], dim=1)
            pad_masks = torch.cat([img_masks, lang_masks], dim=1)

        # (see `make_att_2d_masks` to understand why zeros means bidirection)
        if not vlm_causal:
            if self.use_depth_align and self.align_type == "query":
                att_masks = torch.zeros(
                    (img_emb.size(0), num_img_embs + 3 * self.num_task_tokens + num_lang_embs),
                    device=device,
                    dtype=torch.bool,
                )  # 1, bs_img*(768+48)
            else:
                att_masks = torch.zeros(
                    (img_emb.size(0), num_img_embs + num_lang_embs), device=device, dtype=torch.bool
                )  # 1, bs_img*(768+48)
        else:
            if self.use_depth_align and self.align_type == "query":
                att_masks = torch.ones(
                    (img_emb.size(0), num_img_embs + 3 * self.num_task_tokens + num_lang_embs),
                    device=device,
                    dtype=torch.bool,
                )  # 1, bs_img*(768+48)
            else:
                att_masks = torch.ones(
                    (img_emb.size(0), num_img_embs + num_lang_embs), device=device, dtype=torch.bool
                )  # 1, bs_img*(768+48)
        return embs, pad_masks, att_masks

    def embed_suffix(
        self, state, noisy_actions, timestep
    ):  # (torch.Size([state_bs, 32]), torch.Size([1, state_bs*50, 32]), torch.Size([1]))
        bsize = state.shape[0]  # state_bs = img_bs
        device = state.device
        dtype = state.dtype
        _fp32 = getattr(self.config, "action_fp32", False)
        # embed state
        state_emb = self._fp32_linear(self.state_proj, state) if _fp32 else self.state_proj(state)

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(  # 1, 1024
            timestep,  # torch.Size([1]))
            self.config.proj_width,  # 1024
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb_ori = time_emb

        # Fuse timestep + action information using an MLP
        action_emb = (
            self._fp32_linear(self.action_in_proj, noisy_actions) if _fp32 else self.action_in_proj(noisy_actions)
        )  # torch.Size([1, state_bs*50, 1024])
        time_emb = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])  # [1, 1024] -> [1, state_bs*50, 1024]
        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)  # [1, state_bs*50, 2048]

        action_time_emb = (
            self._fp32_linear(self.action_time_mlp_in, action_time_emb)
            if _fp32
            else self.action_time_mlp_in(action_time_emb)
        )
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = (
            self._fp32_linear(self.action_time_mlp_out, action_time_emb)
            if _fp32
            else self.action_time_mlp_out(action_time_emb)
        )  # [1, state_bs*50, 1024]
        action_time_dim = action_time_emb.shape[1]

        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        pad_masks = torch.ones((bsize, action_time_dim + 1), device=device, dtype=torch.bool)

        # Set attention masks for suffix tokens so that prefix tokens cannot attend to suffix tokens.
        # And state token cannot attend action tokens.
        # Action tokens use a bidirectional attention.
        att_masks = torch.zeros((bsize, action_time_dim + 1), device=device, dtype=torch.bool)
        att_masks[:, :2] = True

        return time_emb_ori, embs, pad_masks, att_masks

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, vlm_causal=False, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vlm_causal
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)  # bs, prefix_len, prefix_len
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values, _ = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        count = 0
        while time >= -dt / 2:
            count += 1
            expanded_time = time.expand(bsize)

            v_t = self.predict_velocity(state, prefix_pad_masks, past_key_values, x_t, expanded_time)

            # Euler step
            x_t += dt * v_t
            time += dt
        print(f"Denoise {count} steps")
        return x_t

    def predict_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """predict velocity at time t using the suffix model."""
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
        )  # bs, suffix_len, prefix_len+suffix_len

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _, _ = self.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if getattr(self.config, "action_fp32", False):
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            v_t = self.action_out_proj(suffix_out)
        return v_t


__all__ = ["AdaRMSNorm", "FixAdaRMSNorm", "FlowMatching", "replace_lnorm_with_adanorm"]
