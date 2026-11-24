"""Inference-only X-VLA model.

Adapted from ``models/modeling_xvla.py`` in 2toINF/X-VLA at revision
``6bc2513f5f1cbec715cc668b414392a6cae5c671``.  The FastAPI server,
training forward/loss path, OpenCV, and JSON transport are intentionally
excluded; only direct in-process action generation remains.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from transformers import PreTrainedModel

from .action_spaces import build_action_space
from .configuration import XVLAConfig
from .modeling_florence2 import Florence2ForConditionalGeneration
from .transformer import SoftPromptedTransformer


class XVLA(PreTrainedModel):
    """Florence-2 plus a soft-prompted cross-embodiment action head."""

    config_class = XVLAConfig
    base_model_prefix = "xvla"
    supports_gradient_checkpointing = False
    # The checkpoint-backed Florence-2 tower dispatches SDPA itself and the
    # action transformer uses WorldFoundry's fused SDPA attention.  Declare
    # that capability on the outer wrapper so Transformers 5 does not reject
    # the implementation before constructing either supported component.
    _supports_sdpa = True
    # The released checkpoint stores Florence's shared embedding once and
    # omits the encoder alias.  Transformers 5 requires the outer wrapper to
    # expose the fully-qualified mapping so the omitted alias is restored by
    # weight tying instead of being treated as a randomly initialized tensor.
    _tied_weights_keys: ClassVar[dict] = {
        "vlm.language_model.model.encoder.embed_tokens.weight": "vlm.language_model.model.shared.weight",
    }

    def __init__(self, config: XVLAConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.num_actions = int(config.num_actions)
        self.use_proprio = bool(config.use_proprio)
        self.action_mode = str(config.action_mode).lower()
        if self.action_mode == "auto":
            self.action_space = build_action_space(
                self.action_mode,
                real_dim=config.real_action_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(self.action_mode)

        self.vlm = Florence2ForConditionalGeneration(config.florence_config).to(torch.float32)

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("X-VLA Florence-2 config must provide projection_dim")
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.action_space.dim_action,
            dim_propio=self.action_space.dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )
        # Required by the modern PreTrainedModel lifecycle: this gathers the
        # nested Florence tied-weight aliases before from_pretrained finalizes
        # missing keys.  It runs before checkpoint restoration and therefore
        # does not mutate loaded parameters.
        self.post_init()

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode text and all valid camera views with Florence-2."""

        batch, views = pixel_values.shape[:2]
        flat_mask = image_mask.reshape(-1).to(torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        if int(flat_mask.sum()) == 0:
            raise ValueError("X-VLA requires at least one valid image view")

        valid_features = self.vlm._encode_image(flat_images[flat_mask])
        tokens, width = valid_features.shape[1:]
        image_features = valid_features.new_zeros((batch * views, tokens, width))
        image_features[flat_mask] = valid_features
        image_features = image_features.view(batch, views, tokens, width)

        input_embeddings = self.vlm.get_input_embeddings()(input_ids)
        merged, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            input_embeddings,
        )
        encoded = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged,
        )[0]
        auxiliary = image_features[:, 1:].reshape(batch, -1, width)
        return {"vlm_features": encoded, "aux_visual_inputs": auxiliary}

    @torch.inference_mode()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """Generate a denoised action chunk using the released linear flow schedule."""

        self.eval()
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be positive")
        encoded = self.forward_vlm(input_ids, image_input, image_mask)
        batch = int(input_ids.shape[0])
        noise = torch.randn(
            batch,
            self.num_actions,
            self.action_space.dim_action,
            device=proprio.device,
            dtype=proprio.dtype,
        )
        action = torch.zeros_like(noise)
        for index in range(steps, 0, -1):
            time = torch.full(
                (batch,),
                index / steps,
                device=proprio.device,
                dtype=proprio.dtype,
            )
            noisy_action = noise * time[:, None, None] + action * (1 - time[:, None, None])
            model_proprio, noisy_action = self.action_space.preprocess(proprio, noisy_action)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=noisy_action,
                proprio=model_proprio,
                t=time,
                **encoded,
            )
        return self.action_space.postprocess(action)


__all__ = ["XVLA"]
