"""Inference-only EventVLA architecture and strict checkpoint restoration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from worldfoundry.synthesis.action_generation.starvla.modeling.action_head import (
    L1RegressionActionHead,
)


class EventVLAVisionLanguageInterface(nn.Module):
    """Local-only Qwen3-VL wrapper with EventVLA's image-role prompt layout."""

    def __init__(
        self,
        base_vlm: str | Path,
        *,
        dtype: torch.dtype,
        attention_implementation: str,
    ) -> None:
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_kwargs = {
            "attn_implementation": attention_implementation,
            "local_files_only": True,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
        }
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(base_vlm),
            **model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(
            str(base_vlm),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.processor.tokenizer.padding_side = "left"
        self.attention_implementation = attention_implementation

    @staticmethod
    def _content(
        anchor_images: Sequence[Any],
        memory_images: Sequence[Any],
        *,
        use_image_role_text: bool,
        temporal_role_text: str,
        memory_role_text: str,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if use_image_role_text and anchor_images:
            content.append({"type": "text", "text": temporal_role_text})
        content.extend({"type": "image", "image": image} for image in anchor_images)
        if use_image_role_text and memory_images:
            content.append({"type": "text", "text": memory_role_text})
        content.extend({"type": "image", "image": image} for image in memory_images)
        return content

    def build_inputs(
        self,
        *,
        anchor_images: Sequence[Any],
        memory_images: Sequence[Any],
        instruction: str,
        use_image_role_text: bool,
        temporal_role_text: str,
        memory_role_text: str,
    ) -> Any:
        content = self._content(
            anchor_images,
            memory_images,
            use_image_role_text=use_image_role_text,
            temporal_role_text=temporal_role_text,
            memory_role_text=memory_role_text,
        )
        content.append({"type": "text", "text": instruction})
        messages = [[{"role": "user", "content": content}]]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def encode_last_hidden(self, inputs: Any) -> torch.Tensor:
        """Avoid logits, KV cache, and all-layer retention during feature extraction."""

        device_type = next(self.model.parameters()).device.type
        with torch.autocast(
            device_type=device_type,
            dtype=next(self.model.parameters()).dtype,
            enabled=device_type == "cuda",
        ):
            outputs = self.model.model(
                **inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        return outputs.last_hidden_state


class EventVLAPolicy(nn.Module):
    """Qwen3-VL action-token policy with raw keyframe-image event prediction."""

    def __init__(
        self,
        base_vlm: str | Path,
        *,
        action_dim: int,
        action_horizon: int,
        image_size: tuple[int, int],
        max_keyframe_images: int,
        use_image_role_text: bool,
        action_token: str,
        prompt_template: str,
        temporal_role_text: str,
        memory_role_text: str,
        dtype: torch.dtype,
        attention_implementation: str,
    ) -> None:
        super().__init__()
        self.qwen_vl_interface = EventVLAVisionLanguageInterface(
            base_vlm,
            dtype=dtype,
            attention_implementation=attention_implementation,
        )
        text_config = self.qwen_vl_interface.model.config.text_config
        hidden_dim = int(text_config.hidden_size)
        self.action_model = L1RegressionActionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            action_dim=action_dim,
            NUM_ACTIONS_CHUNK=action_horizon,
        )
        self.keyframe_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer(
            "_keyframe_annotations_observed",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.image_size = tuple(int(value) for value in image_size)
        self.max_keyframe_images = max(0, int(max_keyframe_images))
        self.use_image_role_text = bool(use_image_role_text)
        self.action_token = str(action_token)
        self.prompt_template = str(prompt_template)
        self.temporal_role_text = str(temporal_role_text)
        self.memory_role_text = str(memory_role_text)

        token_ids = self.qwen_vl_interface.processor.tokenizer(
            self.action_token,
            add_special_tokens=False,
        )["input_ids"]
        if len(token_ids) != 1:
            raise RuntimeError("EventVLA's action marker must map to exactly one token in the staged Qwen3 tokenizer")
        self.action_token_id = int(token_ids[0])

    def _prompt(self, instruction: str) -> str:
        action_tokens = self.action_token * self.action_horizon
        return self.prompt_template.format(
            instruction=instruction,
            action_horizon=self.action_horizon,
            action_tokens=action_tokens,
        )

    def _resize(self, images: Sequence[Any]) -> list[Any]:
        from PIL import Image

        from worldfoundry.core.media.processing.image_utils import load_pil_image

        resized = []
        for image in images:
            pil_image = load_pil_image(image, first_sequence_item=False)
            if pil_image.size != self.image_size:
                pil_image = pil_image.resize(self.image_size, Image.Resampling.BICUBIC)
            resized.append(pil_image)
        return resized

    def _gather_action_queries(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        mask = input_ids == self.action_token_id
        counts = mask.sum(dim=1)
        if bool((counts < self.action_horizon).any().item()):
            raise RuntimeError(
                "EventVLA prompt produced insufficient action tokens: "
                f"counts={counts.tolist()}, required={self.action_horizon}"
            )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)
        masked = torch.where(mask, positions, torch.full_like(positions, -1))
        selected = masked.topk(k=self.action_horizon, dim=-1).values.sort(dim=-1).values
        gather_index = selected.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        return hidden.gather(dim=1, index=gather_index)

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        anchor_images: Sequence[Any],
        memory_images: Sequence[Any],
        instruction: str,
    ) -> dict[str, torch.Tensor]:
        if not anchor_images:
            raise ValueError("EventVLA requires temporal observation images")
        anchors = self._resize(anchor_images)
        selected_memories = list(memory_images)[-self.max_keyframe_images :] if self.max_keyframe_images > 0 else []
        memories = self._resize(selected_memories)
        inputs = self.qwen_vl_interface.build_inputs(
            anchor_images=anchors,
            memory_images=memories,
            instruction=self._prompt(instruction),
            use_image_role_text=self.use_image_role_text,
            temporal_role_text=self.temporal_role_text,
            memory_role_text=self.memory_role_text,
        )
        hidden = self.qwen_vl_interface.encode_last_hidden(inputs)
        queries = self._gather_action_queries(hidden, inputs["input_ids"])

        keyframe_dtype = next(self.keyframe_head.parameters()).dtype
        action_dtype = next(self.action_model.parameters()).dtype
        logits = self.keyframe_head(queries.to(dtype=keyframe_dtype)).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        actions = self.action_model.predict_action(queries.to(dtype=action_dtype))
        return {
            "normalized_actions": actions,
            "chunk_keyframe_prob": probabilities,
        }


def load_eventvla_state_dict(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    path = Path(checkpoint).expanduser().resolve()
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        payload: Any = load_file(str(path), device="cpu")
    elif path.suffix == ".pt":
        payload = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    else:
        raise ValueError(f"unsupported EventVLA checkpoint format: {path.suffix}")
    if isinstance(payload, dict) and isinstance(payload.get("module"), dict):
        payload = payload["module"]
    elif isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload.items()
    ):
        raise TypeError(f"EventVLA checkpoint is not a tensor state dictionary: {path}")
    return payload


def restore_eventvla_policy(
    checkpoint: str | Path,
    *,
    base_vlm: str | Path,
    action_dim: int,
    action_horizon: int,
    image_size: tuple[int, int],
    max_keyframe_images: int,
    use_image_role_text: bool,
    action_token: str,
    prompt_template: str,
    temporal_role_text: str,
    memory_role_text: str,
    device: torch.device,
    dtype: torch.dtype,
    attention_implementation: str,
) -> EventVLAPolicy:
    """Construct the released architecture and restore every policy tensor."""

    policy = EventVLAPolicy(
        base_vlm,
        action_dim=action_dim,
        action_horizon=action_horizon,
        image_size=image_size,
        max_keyframe_images=max_keyframe_images,
        use_image_role_text=use_image_role_text,
        action_token=action_token,
        prompt_template=prompt_template,
        temporal_role_text=temporal_role_text,
        memory_role_text=memory_role_text,
        dtype=dtype,
        attention_implementation=attention_implementation,
    )
    state_dict = load_eventvla_state_dict(checkpoint)
    try:
        policy.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        policy.load_state_dict(state_dict, strict=True)
    return policy.to(device=device, dtype=dtype).requires_grad_(False).eval()


__all__ = [
    "EventVLAPolicy",
    "EventVLAVisionLanguageInterface",
    "load_eventvla_state_dict",
    "restore_eventvla_policy",
]
