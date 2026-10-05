# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Qwen3-VL feature encoder used by StarVLA QwenOFT inference."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class Qwen3VLInterface(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        from worldfoundry.core.attention import resolve_transformers_attention_implementation
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        qwen_config = config.framework.get("qwenvl", {})
        model_id = str(qwen_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct"))
        requested_attention = str(qwen_config.get("attn_implementation", "auto"))
        probe_device = resolve_inference_device(
            str(qwen_config.get("device", "cuda")), allow_cpu_fallback=True
        )
        self.inference_dtype = resolve_inference_dtype(
            probe_device, str(qwen_config.get("torch_dtype", "auto"))
        )
        self.attention_implementation = resolve_transformers_attention_implementation(
            requested_attention, probe_device
        )

        model_kwargs = {
            "attn_implementation": self.attention_implementation,
            "torch_dtype": self.inference_dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        except (ImportError, TypeError):
            if self.attention_implementation == "eager":
                raise
            self.attention_implementation = "sdpa"
            model_kwargs["attn_implementation"] = "sdpa"
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)

        self.model = model
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.processor.tokenizer.padding_side = "left"
        self.config = config

        # Released framework code reads this compatibility field directly.
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def encode_last_hidden(self, **inputs: Any) -> torch.Tensor:
        """Return only the final multimodal hidden state, avoiding logits/KV/all-layer retention."""

        device_type = next(self.model.parameters()).device.type
        with torch.autocast(
            device_type=device_type,
            dtype=self.inference_dtype,
            enabled=device_type == "cuda" and self.inference_dtype in {torch.float16, torch.bfloat16},
        ):
            outputs = self.model.model(
                **inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        return outputs.last_hidden_state

    def build_inputs(self, images: list[list[Any]], instructions: list[str]):
        if len(images) != len(instructions):
            raise ValueError("StarVLA images and instructions must have equal batch size.")

        messages = []
        cot_prompt = ""
        datasets = getattr(self.config, "datasets", None)
        vla_data = getattr(datasets, "vla_data", None) if datasets is not None else None
        if vla_data is not None:
            cot_prompt = str(vla_data.get("CoT_prompt", ""))

        for sample_images, instruction in zip(images, instructions):
            content = [{"type": "image", "image": image} for image in sample_images]
            prompt = cot_prompt.replace("{instruction}", instruction) if cot_prompt else instruction
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return batch.to(self.model.device)


__all__ = ["Qwen3VLInterface"]
