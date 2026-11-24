"""Reusable binary video-question inference backends for PAI-Bench-G."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from .io import read_video


class QwenVideoJudge:
    """Lazily load the shared in-tree Qwen2.5-VL inference components once."""

    def __init__(self, model_path: str | Path, *, max_frames: int = 32) -> None:
        self.model_path = str(model_path)
        self.max_frames = max_frames
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._processor
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("local Qwen judging requires torch and transformers") from exc
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            use_cache=True,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        return self._model, self._processor

    def generate(self, *, prompt: str, video_path: Path, system_prompt: str | None = None) -> str:
        from worldfoundry.base_models.llm_mllm_core.mllm.qwen.qwen_vl_utils import process_vision_info

        model, processor = self._load()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "max_pixels": 360 * 420,
                        "fps": 2,
                        "max_frames": self.max_frames,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)
        generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)]
        return str(processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


class OpenAICompatibleVideoJudge:
    """Send uniformly sampled video frames to an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        max_frames: int = 16,
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.max_frames = max_frames
        self.timeout = timeout

    def generate(self, *, prompt: str, video_path: Path, system_prompt: str | None = None) -> str:
        import cv2
        import requests

        frames = read_video(video_path, rgb=True)
        indices = (
            range(len(frames))
            if len(frames) <= self.max_frames
            else [round(index) for index in __import__("numpy").linspace(0, len(frames) - 1, self.max_frames)]
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index in indices:
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frames[int(index)], cv2.COLOR_RGB2BGR))
            if not ok:
                continue
            value = base64.b64encode(encoded.tobytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{value}"}})
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 32,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])


def build_judge(
    backend: str,
    *,
    model: str | Path,
    base_url: str | None = None,
    max_frames: int = 16,
) -> QwenVideoJudge | OpenAICompatibleVideoJudge:
    if backend == "local-qwen":
        return QwenVideoJudge(model, max_frames=max_frames)
    if backend == "openai-compatible":
        return OpenAICompatibleVideoJudge(
            str(model),
            base_url=base_url or "http://localhost:8000/v1",
            max_frames=max_frames,
        )
    raise ValueError(f"unsupported judge backend: {backend}")
