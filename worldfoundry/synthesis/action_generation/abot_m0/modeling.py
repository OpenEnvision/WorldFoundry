"""In-tree ABot-M0 model assembly and multimodal preprocessing."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from worldfoundry.core.attention import scaled_dot_product_attention

from .action_head import ActionHeadConfig, FlowmatchingActionHead


def _transformers_no_init_weights() -> Any:
    """Return the no-init context across Transformers 4 and 5."""

    try:
        from transformers.initialization import no_init_weights

        return no_init_weights()
    except ImportError:
        from transformers.modeling_utils import no_init_weights

        return no_init_weights()


def _as_rgb_pil(value: Any) -> Any:
    import numpy as np
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, bytes)):
        return Image.open(value).convert("RGB")
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.dtype.kind == "f" and array.max(initial=0.0) <= 1.0:
        array = (array * 255.0).clip(0, 255).astype("uint8")
    if array.dtype != np.uint8:
        array = array.clip(0, 255).astype("uint8")
    return Image.fromarray(array).convert("RGB")


def resize_images(images: Sequence[Sequence[Any]], size: Sequence[int] | int) -> list[list[Any]]:
    from PIL import Image

    if isinstance(size, int):
        height = width = size
    else:
        if len(size) != 2:
            raise ValueError(f"image_size must have two values, got {size!r}")
        height, width = int(size[0]), int(size[1])
    return [
        [_as_rgb_pil(image).resize((width, height), Image.Resampling.BICUBIC) for image in sample]
        for sample in images
    ]


class Qwen3VLInterface(nn.Module):
    """Checkpoint-compatible Qwen3-VL wrapper with local-only assets."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: torch.dtype,
        attention_backend: str = "auto",
        processor_path: str | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
        from worldfoundry.core.attention import resolve_transformers_attention_implementation

        attention = resolve_transformers_attention_implementation(attention_backend)
        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        # The released task .pt contains the complete Qwen state dict. Build
        # only its architecture instead of loading another 4B weight set that
        # would immediately be overwritten by the task checkpoint.
        with _transformers_no_init_weights():
            self.model = Qwen3VLForConditionalGeneration._from_config(
                config,
                dtype=dtype,
                attn_implementation=attention,
            )
        self.processor = AutoProcessor.from_pretrained(
            processor_path or model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None and not hasattr(self.model.config, "hidden_size"):
            self.model.config.hidden_size = text_config.hidden_size

    @property
    def input_device(self) -> torch.device:
        embeddings = self.model.get_input_embeddings()
        return next(embeddings.parameters()).device

    def build_inputs(
        self,
        images: Sequence[Sequence[Any]],
        instructions: Sequence[str],
        *,
        max_length: int | None = None,
    ) -> Mapping[str, Any]:
        if len(images) != len(instructions):
            raise ValueError("images and instructions must have the same batch size")
        messages = []
        for sample_images, instruction in zip(images, instructions, strict=True):
            content = [{"type": "image", "image": image} for image in sample_images]
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])
        options: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True if max_length is None else "max_length",
        }
        if max_length is not None:
            options.update(max_length=int(max_length), truncation=True)
        inputs = self.processor.apply_chat_template(messages, **options)
        return inputs.to(self.input_device)

    def forward(self, **inputs: Any) -> Any:
        return self.model(**inputs)


class CrossAttention(nn.Module):
    """VGGT-to-language fusion retaining the released parameter names."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        nhead: int = 8,
        dropout: float = 0.0,
        kv_dim: int = 2048,
    ) -> None:
        super().__init__()
        if d_hidden % nhead:
            raise ValueError("d_hidden must be divisible by nhead")
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.nhead = nhead
        self.head_dim = d_hidden // nhead
        self.q_proj = nn.Linear(d_model, d_hidden)
        self.k_proj = nn.Linear(kv_dim, d_hidden)
        self.v_proj = nn.Linear(kv_dim, d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image_feature: torch.Tensor, spatial_feature: torch.Tensor) -> torch.Tensor:
        batch, query_length, _ = image_feature.shape
        key_length = spatial_feature.shape[1]
        query = self.q_proj(image_feature).view(batch, query_length, self.nhead, self.head_dim).transpose(1, 2)
        key = self.k_proj(spatial_feature).view(batch, key_length, self.nhead, self.head_dim).transpose(1, 2)
        value = self.v_proj(spatial_feature).view(batch, key_length, self.nhead, self.head_dim).transpose(1, 2)
        attention = scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_attn.p if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(batch, query_length, self.d_hidden)
        return self.norm(image_feature + self.dropout_out(self.out_proj(attention)))


def preprocess_spatial_images(images: Sequence[Sequence[Any]], target_size: int) -> torch.Tensor:
    """Create the `[B, views, 3, H, W]` tensor consumed by in-tree VGGT."""

    import numpy as np

    batches: list[torch.Tensor] = []
    for sample in images:
        views = []
        for value in sample:
            image = _as_rgb_pil(value)
            width, height = image.size
            resized_height = max(14, round(height * (target_size / width) / 14) * 14)
            image = image.resize((target_size, resized_height))
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            if resized_height > target_size:
                start = (resized_height - target_size) // 2
                tensor = tensor[:, start : start + target_size]
            elif resized_height < target_size:
                top = (target_size - resized_height) // 2
                bottom = target_size - resized_height - top
                tensor = torch.nn.functional.pad(tensor, (0, 0, top, bottom), value=1.0)
            views.append(tensor)
        batches.append(torch.stack(views))
    return torch.stack(batches)


class ABotM0Model(nn.Module):
    """Qwen3-VL + optional spatial encoder + AML action expert."""

    def __init__(
        self,
        *,
        qwen_path: str,
        processor_path: str | None,
        action_config: ActionHeadConfig,
        dtype: torch.dtype,
        attention_backend: str,
        use_vggt: bool,
    ) -> None:
        super().__init__()
        self.qwen_vl_interface = Qwen3VLInterface(
            qwen_path,
            processor_path=processor_path,
            dtype=dtype,
            attention_backend=attention_backend,
        )
        hidden_size = int(
            getattr(
                self.qwen_vl_interface.model.config,
                "hidden_size",
                self.qwen_vl_interface.model.config.text_config.hidden_size,
            )
        )
        diffusion = {**dict(action_config.diffusion_model_cfg or {}), "cross_attention_dim": hidden_size}
        action_config = ActionHeadConfig(**{**action_config.__dict__, "diffusion_model_cfg": diffusion})
        self.action_model = FlowmatchingActionHead(action_config)
        self.use_vggt = bool(use_vggt)
        if self.use_vggt:
            from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.models.vggt import VGGT

            self.spatial_model = VGGT()
            self.spatial_projector = nn.Linear(2048, hidden_size)
            self.fuser = CrossAttention(hidden_size, hidden_size, kv_dim=hidden_size)
        else:
            self.spatial_model = None
            self.spatial_projector = None
            self.fuser = None
        self.action_config = action_config

    @staticmethod
    def _autocast(device: torch.device, dtype: torch.dtype) -> Any:
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        images: Sequence[Sequence[Any]],
        instructions: Sequence[str],
        state: torch.Tensor | None,
        image_size: Sequence[int] | int,
        seed: int,
        num_inference_steps: int | None = None,
        tokenizer_max_length: int | None = None,
    ) -> torch.Tensor:
        resized = resize_images(images, image_size)
        inputs = self.qwen_vl_interface.build_inputs(
            resized,
            instructions,
            max_length=tokenizer_max_length,
        )
        qwen_dtype = next(self.qwen_vl_interface.model.parameters()).dtype
        with self._autocast(self.qwen_vl_interface.input_device, qwen_dtype):
            outputs = self.qwen_vl_interface(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
            if self.use_vggt:
                assert self.spatial_model is not None
                assert self.spatial_projector is not None
                assert self.fuser is not None
                spatial_device = next(self.spatial_model.parameters()).device
                spatial_dtype = next(self.spatial_model.parameters()).dtype
                spatial_input = preprocess_spatial_images(resized, int(image_size[0] if not isinstance(image_size, int) else image_size))
                spatial_input = spatial_input.to(device=spatial_device, dtype=spatial_dtype)
                tokens, patch_start = self.spatial_model.aggregator(spatial_input)
                spatial = tokens[-1][:, 0, patch_start:, :]
                spatial = self.spatial_projector(spatial)
                hidden = self.fuser(hidden.to(spatial.device), spatial)
        action_device = next(self.action_model.parameters()).device
        action_dtype = next(self.action_model.parameters()).dtype
        hidden = hidden.to(device=action_device, dtype=action_dtype)
        if state is not None:
            state = state.to(device=action_device, dtype=action_dtype)
        generator = torch.Generator(device=action_device)
        generator.manual_seed(int(seed))
        return self.action_model.predict_action(
            hidden,
            state,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )


__all__ = ["ABotM0Model", "CrossAttention", "Qwen3VLInterface", "resize_images"]
