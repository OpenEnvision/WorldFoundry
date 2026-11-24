"""Optional local Qwen-VL backend for VideoVerse in-tree judging."""

from __future__ import annotations

from pathlib import Path


def local_vlm_generate(*, prompt: str, video_path: Path, model_path: Path) -> str:
    """Run a local Qwen2.5-VL style model on one video prompt pair."""
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        from worldfoundry.base_models.llm_mllm_core.mllm.qwen.qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "Local VideoVerse judge requires torch, transformers, and the in-tree Qwen-VL utilities."
        ) from exc

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        use_cache=True,
    )
    processor = AutoProcessor.from_pretrained(str(model_path), use_fast=True)
    model.eval()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "max_pixels": 360 * 420,
                    "fps": 8,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return str(output_text[0])
