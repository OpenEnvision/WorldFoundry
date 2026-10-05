"""Client for the Wan2.5 audio-video generation API (via the DashScope SDK)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..api_video_client import CredentialedSynthesis
from ._dashscope import RUNTIME_STATUS

__all__ = ["RUNTIME_STATUS", "Wan2p5Synthesis"]

#: DashScope rejects a null seed, so unseeded requests get this fixed value.
DEFAULT_SEED = 12345


def _dashscope_video_synthesis(endpoint: str):
    """Load DashScope only for external-service calls.

    Args:
        endpoint: DashScope-compatible API base URL.
    """
    import dashscope
    from dashscope import VideoSynthesis

    dashscope.base_http_api_url = endpoint
    return VideoSynthesis


class Wan2p5Synthesis(CredentialedSynthesis):
    """Wan2.5 生成合成类，提供统一的接口用于音视频生成。

    负责 API 调用和模型推理相关的工作。
    """

    DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1"

    RUNTIME_STATUS = RUNTIME_STATUS
    IN_TREE_BACKEND = False
    BACKEND_STAGE = "external_service"
    EXTERNAL_SERVICE = True

    def generate_t2av(
        self,
        input_prompt: str,
        size: str = '832*480',
        duration: int = 10,
        negative_prompt: str = "",
        audio: bool = True,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        """文本到音视频生成。

        Args:
            input_prompt: 输入提示词。
            size: 视频尺寸，格式为 'width*height'。
            duration: 视频时长（秒）。
            negative_prompt: 负面提示词。
            audio: 是否生成音频。
            prompt_extend: 是否扩展提示词。
            watermark: 是否添加水印。
            seed: 随机种子；为 None 时使用 ``DEFAULT_SEED``。
            kwargs: 其他透传参数。
        """
        return _dashscope_video_synthesis(self.endpoint).call(
            api_key=self.api_key,
            model='wan2.5-t2v-preview',
            prompt=input_prompt,
            size=size,
            duration=duration,
            negative_prompt=negative_prompt,
            audio=audio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed if seed is not None else DEFAULT_SEED,
            **kwargs
        )

    def generate_i2av(
        self,
        encoded_image: str,
        input_prompt: str,
        resolution: str = '480P',
        duration: int = 10,
        negative_prompt: str = "",
        audio: bool = True,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        """图像到音视频生成。

        Args:
            encoded_image: 编码后的图像（base64 格式）。
            input_prompt: 输入提示词。
            resolution: 视频分辨率。
            duration: 视频时长（秒）。
            negative_prompt: 负面提示词。
            audio: 是否生成音频。
            prompt_extend: 是否扩展提示词。
            watermark: 是否添加水印。
            seed: 随机种子；为 None 时使用 ``DEFAULT_SEED``。
            kwargs: 其他透传参数。
        """
        return _dashscope_video_synthesis(self.endpoint).call(
            api_key=self.api_key,
            model='wan2.5-i2v-preview',
            prompt=input_prompt,
            img_url=encoded_image,
            resolution=resolution,
            duration=duration,
            negative_prompt=negative_prompt,
            audio=audio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed if seed is not None else DEFAULT_SEED,
            **kwargs
        )

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        size: str = '832*480',
        resolution: str = '480P',
        duration: int = 10,
        negative_prompt: str = "",
        audio: bool = True,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成预测结果。

        Args:
            processed_data: 处理后的数据（来自 operator），含 ``prompt``，
                可选 ``encoded_image``。
            task_type: "auto" 自动判断，"t2av" 文本到视频，"i2av" 图像到视频。
            size: t2av 任务的视频尺寸。
            resolution: i2av 任务的分辨率。

        Returns:
            含 ``task_type``、``prompt`` 与 ``response`` 的字典。

        Raises:
            ValueError: 任务类型不支持，或 i2av 缺少 ``encoded_image``。
        """
        prompt = processed_data.get("prompt", "")
        encoded_image = processed_data.get("encoded_image", None)

        if task_type == "auto":
            task_type = "i2av" if encoded_image is not None else "t2av"

        common = dict(
            input_prompt=prompt,
            duration=duration,
            negative_prompt=negative_prompt,
            audio=audio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            **kwargs
        )
        if task_type == "i2av":
            if encoded_image is None:
                raise ValueError("i2av 任务需要提供 encoded_image 参数")
            response = self.generate_i2av(
                encoded_image=encoded_image,
                resolution=resolution,
                **common
            )
        elif task_type == "t2av":
            response = self.generate_t2av(size=size, **common)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response
        }
