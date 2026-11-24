# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Inference-only Qwen3-VL OFT policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .action_head import build_action_head
from .base import StarVLAInferenceModel
from .config import add_discretized_state_to_instruction, merge_framework_config
from .images import to_pil_preserve
from .qwen3 import Qwen3VLInterface
from .utils import resize_images


@dataclass
class QwenOFTDefaultConfig:
    name: str = "QwenOFT"
    qwenvl: dict[str, Any] = field(
        default_factory=lambda: {
            "base_vlm": "Qwen/Qwen3-VL-4B-Instruct",
            "attn_implementation": "auto",
        }
    )
    action_model: dict[str, Any] = field(
        default_factory=lambda: {
            "action_model_type": "MLP",
            "action_dim": 7,
            "action_hidden_dim": 2560,
            "future_action_window_size": 7,
            "past_action_window_size": 0,
        }
    )


class QwenvlOFT(StarVLAInferenceModel):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenOFTDefaultConfig, config)
        self.qwen_vl_interface = Qwen3VLInterface(self.config)
        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = build_action_head(self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        self.action_token = "🔍"
        token_ids = self.qwen_vl_interface.processor.tokenizer(self.action_token, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise RuntimeError(
                f"StarVLA action marker must tokenize to one ID, got {token_ids}. "
                "Use the checkpoint-compatible Qwen tokenizer."
            )
        self.action_token_id = int(token_ids[0])

    @torch.inference_mode()
    def predict_action(self, examples: list[dict[str, Any]], **_: Any) -> dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("StarVLA predict_action requires at least one example.")

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [str(example["lang"]) for example in examples]
        has_state = [example.get("state") is not None for example in examples]
        if any(has_state) and not all(has_state):
            raise ValueError("StarVLA batches cannot mix examples with and without state.")
        if all(has_state):
            instructions = add_discretized_state_to_instruction(
                instructions, [example["state"] for example in examples]
            )

        datasets = getattr(self.config, "datasets", None)
        vla_data = getattr(datasets, "vla_data", None) if datasets is not None else None
        image_size = vla_data.get("obs_image_size", None) if vla_data is not None else None
        if image_size:
            batch_images = resize_images(batch_images, target_size=int(image_size))

        action_tokens = self.action_token * self.chunk_len
        suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + suffix for instruction in instructions]

        model_inputs = self.qwen_vl_interface.build_inputs(batch_images, instructions)
        last_hidden = self.qwen_vl_interface.encode_last_hidden(**model_inputs)
        action_queries = self._gather_action_token_embeddings(
            last_hidden, model_inputs["input_ids"], self.action_token_id
        )
        head_dtype = next(self.action_model.parameters()).dtype
        predicted = self.action_model.predict_action(action_queries.to(dtype=head_dtype))
        return {"normalized_actions": predicted.float().cpu().numpy()}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = last_hidden.shape
        if input_ids.shape[:2] != (batch_size, sequence_length):
            raise RuntimeError(
                f"StarVLA hidden/token shape mismatch: hidden={last_hidden.shape}, ids={input_ids.shape}."
            )
        mask = input_ids == action_token_id
        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            raise RuntimeError(
                f"StarVLA expected at least {self.chunk_len} action tokens per sample, got {counts.tolist()}."
            )

        positions = torch.arange(sequence_length, device=input_ids.device).expand(batch_size, -1)
        masked_positions = torch.where(mask, positions, torch.full_like(positions, -1))
        selected = masked_positions.topk(k=self.chunk_len, dim=-1).values.sort(dim=-1).values
        gather_index = selected.unsqueeze(-1).expand(-1, -1, hidden_dim)
        return last_hidden.gather(dim=1, index=gather_index)


__all__ = ["QwenvlOFT"]
