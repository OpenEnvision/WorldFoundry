from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from transformers import (
    AutoProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.quantize.nf4 import qwen_load_kwargs

from worldfoundry.base_models.diffusion_model.kandinsky6_types import TextEmbeds

_PROMPT_TEMPLATE = "\n".join([
    "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
    "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
    "Describe the location of the video, main characters or objects and their action.",
    "Describe the dynamism of the video and presented actions.",
    "Name the visual style of the video: whether it is a professional footage, user generated content, "
    "some kind of animation, video game or scren content.",
    "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
    "Pay attention to the order of key actions shown in the scene.<|im_end|>",
    "<|im_start|>user\n{}<|im_end|>",
])
# Number of tokens to crop from the front (system prompt + user header)
_CROP_START = 129


def clip_text_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Select CLIP text weights from either a full CLIP save or a ``CLIPTextModel`` file.

    A dual-encoder checkpoint prefixes text keys with ``text_model.``. The Diffusers
    ``text_encoder_2`` file stores the text tower directly (``embeddings``, ``encoder``,
    ``final_layer_norm``). ``position_ids`` is a generated buffer and is not loaded.
    """
    prefix = "text_model."
    if any(key.startswith(prefix) for key in state):
        selected = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix) and not key.endswith("position_ids")
        }
    else:
        selected = {key: value for key, value in state.items() if not key.endswith("position_ids")}
    return selected


def clip_tokenizer_dir(clip_path: str | Path) -> Path:
    """Directory that holds the CLIP tokenizer next to ``text_encoder_2``."""
    root = Path(clip_path)
    if (root / "tokenizer.json").is_file():
        return root
    sibling = root.parent / "tokenizer_2"
    if (sibling / "tokenizer.json").is_file():
        return sibling
    return root


def qwen_processor_dir(qwen_path: str | Path) -> Path:
    """Directory that holds the Qwen processor.

    The Diffusers snapshot keeps ``preprocessor_config.json`` inside ``text_encoder``.
    When that file is absent, the sibling ``tokenizer`` directory is the processor.
    """
    root = Path(qwen_path)
    if (root / "preprocessor_config.json").is_file() or (root / "processor_config.json").is_file():
        return root
    sibling = root.parent / "tokenizer"
    if (sibling / "preprocessor_config.json").is_file() or (sibling / "processor_config.json").is_file():
        return sibling
    return root


def _load_clip_text_model(clip_path: str) -> CLIPTextModel:
    """Load CLIP text tower weights into ``CLIPTextModel``.

    A full CLIP checkpoint is reduced to its ``text_model.*`` keys. A Diffusers
    ``CLIPTextModel`` file is loaded as stored.
    """
    root = Path(clip_path)
    model = CLIPTextModel(CLIPTextConfig.from_pretrained(root))

    safetensors_path = root / "model.safetensors"
    bin_path = root / "pytorch_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path))
    elif bin_path.is_file():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No CLIP weights under {root}")

    text_state = clip_text_state(state)
    if not text_state:
        raise RuntimeError(f"No CLIP text weights in {root}")
    model.load_state_dict(text_state, strict=True)
    return model


class Kandinsky6TextEmbedder:
    """Qwen2.5-VL-7B (dense text tokens) + CLIP ViT-L/14 (pooled embed)."""

    def __init__(
        self,
        qwen_path: str,
        clip_path: str,
        max_length: int = 1024,
        device: str | torch.device = "cuda",
        quantized_qwen: bool = False,
        text_token_padding: bool = False,
    ):
        self.max_length = max_length
        self.device = torch.device(device)
        self.quantized_qwen = quantized_qwen
        # K5 parity: pad to max_length (static shapes for compile/AOTI) + attn mask.
        self.text_token_padding = text_token_padding

        qwen_kwargs = qwen_load_kwargs(self.device, quantized=quantized_qwen)
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_path, **qwen_kwargs
        ).eval()
        self.processor = AutoProcessor.from_pretrained(qwen_processor_dir(qwen_path))

        self.clip = _load_clip_text_model(clip_path).to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_dir(clip_path))

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        type_of_content: str = "video",
    ) -> tuple[TextEmbeds, Tensor, Tensor | None]:
        """Encode texts → (TextEmbeds, text_cu_seqlens, attention_mask|None).

        Returns:
            text_embeds["text_embeds"]:
                packed ``(S_text, 3584)`` when ``text_token_padding=False``;
                padded ``(bs, max_length, 3584)`` for batched input when
                ``text_token_padding=True``.
            text_embeds["pooled_embed"]: (bs, 768)
            text_cu_seqlens: (bs+1,) int32
            attention_mask: ``(bs, max_length)`` bool (True=valid) for batched
                input, or ``(max_length,)`` for one input; None only for the
                historical single-input packed path.
        """
        del type_of_content  # reserved for image/i2v templates (K5)
        full_texts = [_PROMPT_TEMPLATE.format(t) for t in texts]

        inputs = self.processor(
            text=full_texts,
            images=None,
            videos=None,
            max_length=self.max_length + _CROP_START,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        ).to(self.qwen.device)

        qwen_out = self.qwen(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        )
        # (bs, max_length+crop, 3584) → crop system prompt
        embeds = qwen_out["hidden_states"][-1][:, _CROP_START:]  # (bs, max_length, 3584)
        attn = inputs["attention_mask"][:, _CROP_START:]  # (bs, max_length)

        if self.text_token_padding:
            # Keep the historical unbatched shape for export and callers that
            # pass one prompt; preserve B for actual batched inference.
            if embeds.shape[0] == 1:
                text_embeds_out = embeds[0]
                attention_mask: Tensor | None = attn[0].to(dtype=torch.bool)
            else:
                text_embeds_out = embeds
                attention_mask = attn.to(dtype=torch.bool)
            cu_seqlens = torch.arange(
                len(texts) + 1, dtype=torch.int32, device=embeds.device
            ) * embeds.shape[1]
        else:
            token_counts = attn.sum(dim=1).to(torch.int32)
            if len(texts) == 1:
                text_embeds_out = embeds[attn.bool()]
                attention_mask = None
                cu_seqlens = torch.zeros(2, dtype=torch.int32, device=embeds.device)
                cu_seqlens[1] = token_counts[0]
            else:
                # Native batch-dim attention needs a rectangular tensor. Trim
                # only the unused right padding and mask the remainder.
                max_tokens = int(token_counts.max().item())
                text_embeds_out = embeds[:, :max_tokens]
                attention_mask = attn[:, :max_tokens].to(dtype=torch.bool)
                cu_seqlens = torch.arange(
                    len(texts) + 1, dtype=torch.int32, device=embeds.device
                ) * max_tokens

        # CLIP pooled
        clip_inputs = self.tokenizer(
            texts,
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.clip.device)
        pooled_embed = self.clip(**clip_inputs)["pooler_output"]  # (bs, 768)

        embeds_dict: TextEmbeds = {
            "text_embeds": text_embeds_out.to(self.device),
            "pooled_embed": pooled_embed.to(self.device),
        }
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        return embeds_dict, cu_seqlens.to(self.device), attention_mask

    def to(self, device, non_blocking: bool = False) -> Kandinsky6TextEmbedder:
        self.device = torch.device(device)
        # bitsandbytes NF4 weights are device-bound; moving breaks the quantized modules.
        if not self.quantized_qwen:
            self.qwen = self.qwen.to(self.device, non_blocking=non_blocking)
        self.clip = self.clip.to(self.device, non_blocking=non_blocking)
        return self
