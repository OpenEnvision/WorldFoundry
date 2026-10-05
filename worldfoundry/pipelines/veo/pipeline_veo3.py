"""Veo3 visual generation pipeline module."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PIL import Image

from ...operators.veo3_operator import Veo3Operator
from ...synthesis.visual_generation.veo.veo3_synthesis import Veo3Synthesis
from ..pipeline_utils import PipelineABC

_DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com"
_API_KEY_ENV = ("VEO3_API_KEY", "VEO_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
_ENDPOINT_ENV = ("VEO3_ENDPOINT", "VEO_ENDPOINT", "GOOGLE_GEMINI_BASE_URL")
_PLACEHOLDER_KEYS = {"your_api_key", "your api key"}


def _resolve_api_key(api_key: Optional[str]) -> str:
    """Resolve the Veo3 API key from explicit input or supported environment variables.

    Args:
        api_key: Optional API key passed by the caller.
    """
    if api_key and api_key.strip() and api_key.strip() not in _PLACEHOLDER_KEYS:
        return api_key.strip()
    for env_name in _API_KEY_ENV:
        # Attempt to retrieve from environment variables as a fallback resolution
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return env_value.strip()
    raise ValueError("Veo3 API key is required. Pass api_key or set VEO3_API_KEY/VEO_API_KEY/GEMINI_API_KEY/GOOGLE_API_KEY.")


def _resolve_endpoint(endpoint: Optional[str]) -> str:
    """Resolve the Veo3 endpoint from explicit input, env, or the Gemini default.

    Args:
        endpoint: Optional endpoint URL passed by the caller.
    """
    if endpoint and endpoint.strip():
        return endpoint.strip()
    for env_name in _ENDPOINT_ENV:
        # Attempt to retrieve from environment variables as a fallback resolution
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return env_value.strip()
    return _DEFAULT_ENDPOINT


class Veo3Pipeline(PipelineABC):
    """
    使用 Gemini 原生视频生成 API 实现的 Veo3 管线。
    通过检测是否携带图像自动区分 T2V / I2V。
    """

    def __init__(
        self,
        operator: Optional[Veo3Operator] = None,
        synthesis_model: Optional[Veo3Synthesis] = None,
        endpoint: str = _DEFAULT_ENDPOINT,
        api_key: str = "your_api_key",
    ):
        """
        初始化 Veo3Pipeline
        
        Args:
            operator: Veo3 operator 实例（如果为None则自动创建）
            synthesis_model: Veo3 synthesis 模型实例（如果为None则自动创建）
            api_key: API密钥
            endpoint: API基础URL
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.operator = operator
        self.synthesis_model = synthesis_model

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> 'Veo3Pipeline':
        """Build the API-only Veo3 pipeline without loading a checkpoint.

        Args:
            model_path: Optional dict of API adapter options.
            required_components: Optional dict merged into adapter options.
            device: Accepted for the shared loader signature.
            model_id: Accepted for the shared loader signature.
            **kwargs: Additional API adapter options.
        """
        del device, model_id
        options: Dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            raise ValueError("Veo3 is API-only; pass API options as a dict instead of a checkpoint path.")
        options.update(required_components or {})
        options.update(kwargs)
        return cls.api_init(
            endpoint=options.get("endpoint"),
            api_key=options.get("api_key"),
            model=options.get("model"),
            logger=options.get("logger"),
        )

    @classmethod
    def api_init(
        cls,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        logger=None,
        **kwargs
    ) -> 'Veo3Pipeline':
        """
        从配置加载完整的 pipeline
        
        Args:
            api_key: API密钥
            endpoint: API基础URL
            logger: 日志记录器
            **kwargs: 额外参数
            
        Returns:
            Veo3Pipeline: 初始化的 pipeline 实例
        """
        endpoint = _resolve_endpoint(endpoint)
        api_key = _resolve_api_key(api_key)
        if logger:
            logger.info(f"Loading Veo3 pipeline with endpoint: {endpoint}")
        
        # 加载 synthesis 模型
        if logger:
            logger.info("Loading Veo3 synthesis model...")
        synthesis_model = Veo3Synthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
            **kwargs
        )
        
        if logger:
            logger.info("Initializing Veo3 operator...")
        operator = Veo3Operator()
        
        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            api_key=api_key,
            endpoint=endpoint
        )
        
        if logger:
            logger.info("Veo3 pipeline loaded successfully")
        
        return pipeline

    def process(
        self,
        prompt: str,
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
        处理输入，通过 operator 预处理后传给 synthesis 模型
        
        Args:
            prompt: 文本提示词
            images: 主图像（可选）
            aspect_ratio: 宽高比
            resolution: 分辨率
            duration_seconds: 视频时长（秒）
            negative_prompt: 负面提示词
            seed: 随机种子
            last_frame: 最后一帧图像（可选）
            reference_images: 参考图像列表（可选）
            person_generation: 人物生成设置（可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含处理后的数据
        """
        if self.operator is None:
            raise ValueError("Operator is not initialized")

        processed_data: Dict[str, Any] = {}

        self.operator.get_interaction(prompt)
        processed_interaction = self.operator.process_interaction()
        processed_data['prompt'] = processed_interaction['processed_prompt']
        
        processed_perception = self.operator.process_perception(
            prompt=prompt,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            seed=seed,
            last_frame=last_frame,
            reference_images=reference_images,
            person_generation=person_generation,
            enhance_prompt=enhance_prompt,
            generate_audio=generate_audio,
            fps=fps,
            **kwargs
        )

        processed_data.update(processed_perception)

        return processed_data

    def __call__(
        self,
        prompt: str,
        images: Optional[Image.Image] = None,
        task_type: str = "auto",
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
        自动根据是否提供 image 选择 T2V 或 I2V
        
        Args:
            prompt: 文本提示词
            images: 主图像（可选），如果提供则自动使用 i2av
            task_type: 任务类型，"auto" 自动判断，"t2av" 文本到视频，"i2av" 图像到视频
            aspect_ratio: 宽高比
            resolution: 分辨率
            duration_seconds: 视频时长（秒）
            negative_prompt: 负面提示词
            seed: 随机种子
            last_frame: 最后一帧图像（可选）
            reference_images: 参考图像列表（可选）
            person_generation: 人物生成设置（可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含生成结果：
                - task_type: 任务类型
                - result: 生成结果
        """
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        _resolve_api_key(self.api_key)
        _resolve_endpoint(self.endpoint)
        
        # 使用 operator 预处理输入
        processed_data = self.process(
            prompt=prompt,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            seed=seed,
            last_frame=last_frame,
            reference_images=reference_images,
            person_generation=person_generation,
            enhance_prompt=enhance_prompt,
            generate_audio=generate_audio,
            fps=fps,
            **kwargs
        )
        
        # 使用 synthesis 模型的 predict 方法进行推理
        response = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            **kwargs
        )

        return response

    def get_operator(self) -> Optional[Veo3Operator]:
        """获取 operator 实例"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[Veo3Synthesis]:
        """获取 synthesis 模型实例"""
        return self.synthesis_model
