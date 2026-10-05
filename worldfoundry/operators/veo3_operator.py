"""Module for the Veo3 operator implementation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ._media import pil_to_png_data_url
from .base_operator import BaseOperator


def _image_to_data_url(image_input: Image.Image) -> Tuple[str, str]:
    """Compatibility helper for explicit callers; Veo generation uses PIL images."""
    mime_type = "image/png"
    return pil_to_png_data_url(image_input), mime_type


class Veo3Operator(BaseOperator):
    """
    Veo3 数据处理 Operator
    
    负责输入类型检查并把 PIL 图像直接交给 Gemini SDK
    不涉及模型推理和API调用
    """
    
    def __init__(
        self,
        operation_types: list = None
    ):
        """
        初始化 Veo3Operator
        
        Args:
            operation_types: 操作类型列表
        """
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(Veo3Operator, self).__init__(operation_types)
        
        # 初始化交互模板   
        self.interaction_template = ["text_prompt", "image_prompt", "multimodal_prompt"]
        self.interaction_template_init()
    
    def process_image(self, image_input: Image.Image) -> Tuple[str, str]:
        """Return a data URL for legacy explicit callers; inference skips this."""
        return _image_to_data_url(image_input)

    def build_user_content(
        self,
        prompt: str,
        *,
        aspect_ratio: str,
        resolution: str,
        duration_seconds: int,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        person_generation: Optional[str] = None,
        reference_images: Optional[List[Image.Image]] = None,
        image: Optional[Image.Image] = None,   # 可为 None（T2V）
        last_frame: Optional[Image.Image] = None,
        enhance_prompt: Optional[bool] = None,
        generate_audio: Optional[bool] = None,
        fps: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retain the historical payload for explicit callers of this helper.

        Normal Veo inference uses native SDK fields and never invokes this method.
        """
        metadata = {
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration_seconds": duration_seconds,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "person_generation": person_generation,
            "enhance_prompt": enhance_prompt,
            "generate_audio": generate_audio,
            "fps": fps,
        }
        content: List[Dict[str, Any]] = [{
            "type": "text",
            "text": json.dumps({"instruction": prompt, "metadata": metadata}, ensure_ascii=False),
        }]

        def to_data_url(value: Image.Image) -> Dict[str, Any]:
            url, _ = _image_to_data_url(value)
            return {"type": "image_url", "image_url": {"url": url}}

        if image is not None:
            content.append(to_data_url(image))
            if reference_images:
                content.extend(to_data_url(item) for item in reference_images)
            if last_frame is not None:
                content.append(to_data_url(last_frame))
        return content

    def get_interaction(self, prompt: str):
        """Process and append the interaction to the current sequence."""
        if self.check_interaction(prompt):
            self.current_interaction.append(prompt)

    def check_interaction(self, prompt: str) -> bool:
        """Validate the given interaction sequence or parameters."""
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt)}")
        return True
    
    def process_interaction(self, **kwargs) -> Dict[str, Any]:
        """Process the recorded interactions and return the generated actions."""
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return {
            "processed_prompt": now_interaction
        }
        
    
    def process_perception(
        self,
        prompt: str,
        *,
        images: Optional[Image.Image] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration_seconds: int = 8,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        last_frame: Optional[Image.Image] = None,
        reference_images: Optional[List[Image.Image]] = None,
        person_generation: Optional[str] = None,
        enhance_prompt: Optional[bool] = None,
        generate_audio: Optional[bool] = None,
        fps: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理交互输入，生成模型所需的输入格式
        
        Args:
            prompt: 文本提示词
            images: 主图像（PIL.Image，可选）
            aspect_ratio: 宽高比
            resolution: 分辨率
            duration_seconds: 视频时长（秒）
            negative_prompt: 负面提示词
            seed: 随机种子
            last_frame: 最后一帧图像（PIL.Image，可选）
            reference_images: 参考图像列表（PIL.Image，可选）
            person_generation: 人物生成设置（可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含处理后的输入数据：
                - prompt: 文本提示词
                - images: 主图像（如果有）
                - reference_images: 参考图像列表（如果有）
        """
        # 类型检查
        if images is not None and not isinstance(images, Image.Image):
            raise TypeError(f"images must be PIL.Image, got {type(images)}")
        if last_frame is not None and not isinstance(last_frame, Image.Image):
            raise TypeError(f"last_frame must be PIL.Image, got {type(last_frame)}")
        if reference_images:
            for img in reference_images:
                if not isinstance(img, Image.Image):
                    raise TypeError(f"reference_images must be List[PIL.Image], got {type(img)}")
        
        return {
            "images": images,
            "reference_images": reference_images,
            "last_frame": last_frame,
            "veo_config": {
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "duration_seconds": duration_seconds,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "person_generation": person_generation,
                "enhance_prompt": enhance_prompt,
                "generate_audio": generate_audio,
                "fps": fps,
            },
        }
