"""Client for OpenAI Sora 2 video generation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..api_video_client import OpenAiVideoSynthesis


class Sora2Synthesis(OpenAiVideoSynthesis):
    """Sora2 生成合成类，提供统一的接口用于音视频生成。

    负责 API 调用和模型推理相关的工作。
    """

    DEFAULT_ENDPOINT = "https://api.openai.com/v1"

    MODEL = "sora-2"

    def generate_t2av(
        self,
        input_prompt: str,
        size: str = "1280x720",
        duration: int = 8,
    ):
        """文本到视频生成（T2V）。"""
        return self.client.videos.create(
            model=self.MODEL,
            prompt=input_prompt,
            size=_normalize_size(size),
            seconds=str(duration),
        )

    def generate_i2av(
        self,
        encoded_image: Tuple[str, bytes, str],
        input_prompt: str,
        size: str = "1280x720",
        duration: int = 8,
    ):
        """图像到视频生成（I2V）。

        Args:
            encoded_image: 图像数据元组 (filename, bytes, mime_type)。
            input_prompt: 输入提示词。
            size: 视频尺寸。
            duration: 视频时长（秒）。
        """
        return self.client.videos.create(
            model=self.MODEL,
            prompt=input_prompt,
            size=_normalize_size(size),
            seconds=str(duration),
            input_reference=tuple(encoded_image),
        )

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        size: str = "1280x720",
        duration: int = 8,
        **kwargs
    ) -> Dict[str, Any]:
        """按输入自动选择 T2V 或 I2V 并提交生成任务。

        Raises:
            ValueError: 任务类型不支持，或 i2av 缺少 ``encoded_image``。
        """
        prompt = processed_data.get("prompt", "")
        encoded_image = processed_data.get("encoded_image", None)

        if task_type == "auto":
            task_type = "i2av" if encoded_image is not None else "t2av"

        if task_type == "i2av":
            if encoded_image is None:
                raise ValueError("i2av 任务需要提供 encoded_image 参数")
            response = self.generate_i2av(
                encoded_image=encoded_image,
                input_prompt=prompt,
                size=size,
                duration=duration,
                **kwargs
            )
        elif task_type == "t2av":
            response = self.generate_t2av(
                input_prompt=prompt,
                size=size,
                duration=duration,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response
        }


def _normalize_size(size: str) -> str:
    """Accept ``1280*720`` as well as the ``1280x720`` the API expects."""
    return size.replace('*', 'x')


__all__ = ["Sora2Synthesis"]
