"""RT-2-compatible inference baseline supported by VLANeXt checkpoints."""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from .encoder import ActionTransformerProjector
from .model import load_local_transformers_model


class LlamaProcessorWrapper:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor


class RT2LikeBaseline(nn.Module):
    def __init__(
        self,
        lmm_path="meta-llama/Llama-3.2-1B-Instruct",
        vision_encoder_path="google/siglip2-base-patch16-256",
        action_dim=7,
        num_actions=1,
        num_history=0,
        use_proprio_input_vlm=True,
        use_transformer_projector=True,
        projector_depth=2,
        projector_num_heads=4,
        num_bins=256,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_backbone_weights=True,
    ):
        super().__init__()

        print(f"Initializing RT-2 Baseline with attn_implementation: {attn_implementation}")

        self.lmm = load_local_transformers_model(
            LlamaForCausalLM,
            lmm_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
            load_weights=load_backbone_weights,
        )
        self.vision_encoder = load_local_transformers_model(
            SiglipVisionModel,
            vision_encoder_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
            load_weights=load_backbone_weights,
        )

        tokenizer = AutoTokenizer.from_pretrained(lmm_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        image_processor = SiglipImageProcessor.from_pretrained(
            vision_encoder_path,
            local_files_only=True,
        )
        self.processor = LlamaProcessorWrapper(tokenizer, image_processor)

        self.hidden_size = self.lmm.config.hidden_size
        self.vision_projector = nn.Sequential(
            nn.Linear(self.vision_encoder.config.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.action_dim = action_dim
        self.num_actions = num_actions
        self.num_history = num_history
        self.num_bins = num_bins
        self.use_proprio_input_vlm = use_proprio_input_vlm

        if self.use_proprio_input_vlm:
            if use_transformer_projector:
                self.action_projector = ActionTransformerProjector(
                    action_dim=action_dim,
                    hidden_size=self.hidden_size,
                    depth=projector_depth,
                    num_heads=projector_num_heads,
                )
            else:
                self.action_projector = nn.Linear(action_dim, self.hidden_size)
        else:
            self.action_projector = None

        vocab_limit = 128000
        self.action_token_start_idx = vocab_limit - num_bins
        print(
            f"RT-2 Mode: Using vocabulary indices [{self.action_token_start_idx}, {vocab_limit}) for {num_bins} action bins."
        )

        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def predict_action(
        self, input_ids, attention_mask, proprioception=None, proprio_attention_mask=None, pixel_values=None, **kwargs
    ):
        B = input_ids.shape[0]

        vision_outputs = self.vision_encoder(pixel_values, output_hidden_states=True)
        image_embeds = self.vision_projector(vision_outputs.last_hidden_state)

        if image_embeds.shape[0] != B:
            num_views = image_embeds.shape[0] // B
            image_embeds = image_embeds.view(B, num_views, -1, image_embeds.shape[-1])
            image_embeds = image_embeds.flatten(1, 2)

        text_embeds = self.lmm.model.embed_tokens(input_ids)

        embeds_list = [image_embeds]
        mask_list = [torch.ones(B, image_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)]

        if self.use_proprio_input_vlm and proprioception is not None:
            proprio_embeds = self.action_projector(
                proprioception.to(device=text_embeds.device, dtype=text_embeds.dtype)
            )
            embeds_list.append(proprio_embeds)
            if proprio_attention_mask is not None:
                mask_list.append(proprio_attention_mask.to(attention_mask.device))
            else:
                mask_list.append(
                    torch.ones(B, proprio_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                )

        embeds_list.append(text_embeds)
        mask_list.append(attention_mask)

        inputs_embeds = torch.cat(embeds_list, dim=1)
        attention_mask = torch.cat(mask_list, dim=1)

        class RT2ActionLogitsProcessor(LogitsProcessor):
            def __init__(self, start_idx, end_idx):
                self.start_idx = start_idx
                self.end_idx = end_idx

            def __call__(self, input_ids, scores):
                scores[:, : self.start_idx] = float("-inf")
                scores[:, self.end_idx :] = float("-inf")
                return scores

        logits_processor = LogitsProcessorList(
            [RT2ActionLogitsProcessor(self.action_token_start_idx, self.action_token_start_idx + self.num_bins)]
        )

        total_action_tokens = self.num_actions * self.action_dim

        pad_token_id = getattr(self.lmm.config, "pad_token_id", None)
        pad_token_id = pad_token_id if pad_token_id is not None else 0

        dummy_input_ids = torch.full(
            (B, inputs_embeds.shape[1]), pad_token_id, dtype=torch.long, device=inputs_embeds.device
        )

        outputs = self.lmm.generate(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=total_action_tokens,
            logits_processor=logits_processor,
            do_sample=False,
            use_cache=True,
            eos_token_id=[],
            pad_token_id=pad_token_id,
        )

        generated_ids = outputs[:, inputs_embeds.shape[1] :]

        bin_indices = generated_ids - self.action_token_start_idx

        action_flat = (bin_indices.float() / (self.num_bins - 1)) * 2 - 1
        action = action_flat.view(B, self.num_actions, self.action_dim).to(dtype=self.lmm.dtype)

        return action
