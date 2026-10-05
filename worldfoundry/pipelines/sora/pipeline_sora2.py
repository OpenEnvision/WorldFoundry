"""Sora2 visual generation pipeline module."""

import logging
import os
import time
from PIL import Image
from typing import Optional, Dict, Any

from ..pipeline_utils import PipelineABC
from ...operators.sora2_operator import Sora2Operator
from ...synthesis.visual_generation.sora.sora2_synthesis import Sora2Synthesis


_DEFAULT_ENDPOINT = "https://api.openai.com/v1"
_API_KEY_ENV = ("SORA2_API_KEY", "SORA_API_KEY", "OPENAI_API_KEY")
_ENDPOINT_ENV = ("SORA2_ENDPOINT", "SORA_ENDPOINT", "OPENAI_BASE_URL")
_PLACEHOLDER_KEYS = {"your_api_key", "your api key"}

logger = logging.getLogger(__name__)


def _resolve_api_key(api_key: Optional[str]) -> str:
    """Resolve the Sora2 API key from explicit input or supported environment variables.

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
    raise ValueError("Sora2 API key is required. Pass api_key or set SORA2_API_KEY/SORA_API_KEY/OPENAI_API_KEY.")


def _resolve_endpoint(endpoint: Optional[str]) -> str:
    """Resolve the Sora2 API endpoint from explicit input, env, or default.

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



class Sora2Pipeline(PipelineABC):
    """Pipeline implementation for Sora2 visual generation."""
    def __init__(
        self, 
        operator: Optional[Sora2Operator] = None,
        synthesis_model: Optional[Sora2Synthesis] = None,
        endpoint: str = "https://api.openai.com/v1", 
        api_key: str = "your_api_key"):
        """
        初始化 Sora2Pipeline
        Args:
            operator: Sora2 operator 实例（如果为None则自动创建）
            synthesis_model: Sora2 synthesis 模型实例（如果为None则自动创建）
            endpoint: API基础URL
            api_key: API密钥
        """
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.endpoint = endpoint
        self.api_key = api_key

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> 'Sora2Pipeline':
        """Build the API-only Sora2 pipeline without loading a checkpoint.

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
            raise ValueError("Sora2 is API-only; pass API options as a dict instead of a checkpoint path.")
        options.update(required_components or {})
        options.update(kwargs)
        return cls.api_init(
            endpoint=options.get("endpoint"),
            api_key=options.get("api_key"),
            logger=options.get("logger"),
        )

    @classmethod
    def api_init(
        cls,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        logger=None,
        **kwargs
    ) -> 'Sora2Pipeline':
        """
        从配置加载完整的 pipeline
        
        Args:
            endpoint: API基础URL
            api_key: API密钥
            logger: 日志记录器
            **kwargs: 额外参数
        """
        endpoint = _resolve_endpoint(endpoint)
        api_key = _resolve_api_key(api_key)
        if logger:
            logger.info(f"Loading Sora2 pipeline with endpoint: {endpoint}")
        
        # 加载 synthesis 模型
        if logger:
            logger.info("Loading Sora2 synthesis model...")
        synthesis_model = Sora2Synthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
            **kwargs
        )
        
        if logger:
            logger.info("Initializing Sora2 operator...")
        operator = Sora2Operator()
        
        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            endpoint=endpoint,
            api_key=api_key
        )
        
        if logger:
            logger.info("Sora2 pipeline loaded successfully")
        
        return pipeline

    def process(
        self,
        prompt: str,
        images: Optional[Image.Image] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理输入，通过 operator 预处理后传给 synthesis 模型
        """
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        processed_data: Dict[str, Any] = {}

        # 视文本为交互输入通过process_interaction处理
        self.operator.get_interaction(prompt)
        processed_interaction = self.operator.process_interaction()
        processed_data['prompt'] = processed_interaction['processed_prompt']

        # 视图片为感知输入通过process_perception处理
        processed_perception = self.operator.process_perception(
            images=images,
            **kwargs
        )
        processed_data['encoded_image'] = processed_perception['encoded_image']
        processed_data['images'] = processed_perception['images']
        
        return processed_data

    def __call__(
        self,
        prompt: str,
        images: Optional[Image.Image] = None,
        size: str = "1280x720",
        duration: int = 4,
        task_type: str = "auto",
        wait: bool = True,
        poll_interval: int = 2,
        max_retries: int = 600,
        **kwargs
    ) -> Dict[str, Any]:
        """
        自动根据是否提供 images 选择 T2V 或 I2V
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
            **kwargs
        )
        
        # 使用 synthesis 模型的 predict 方法进行推理
        result = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            size=size,
            duration=duration,
            **kwargs
        )

        if wait:
            final_video = self._poll_video_status(
                video=result["response"],
                poll_interval=poll_interval,
                max_retries=max_retries
            )
            result["response"] = final_video

        return result

    def _poll_video_status(self, video: Any, poll_interval: int = 2, max_retries: int = 600) -> Any:
        """
        轮询视频生成状态，直到完成或失败，返回最终视频对象
        """
        if self.synthesis_model is None or not hasattr(self.synthesis_model, "client"):
            raise ValueError("Synthesis model client is not initialized")

        openai_client = self.synthesis_model.client

        completed_statuses = ("completed", "succeeded", "success", "done")
        retry_count = 0
        last_status = None

        while (
            video.status.lower() not in [s.lower() for s in completed_statuses]
            and video.status.lower() not in ["failed", "error"]
        ):
            if retry_count >= max_retries:
                raise TimeoutError(f"Sora2 video polling reached the maximum number of retries ({max_retries}).")

            video = openai_client.videos.retrieve(video.id)

            progress_raw = getattr(video, "progress", 0)
            progress = progress_raw if progress_raw is not None else 0
            status_lower = video.status.lower()

            if video.status != last_status:
                logger.info("Sora2 video %s: %s (progress %.1f%%)", video.id, video.status, progress)
                last_status = video.status

            if status_lower in [s.lower() for s in completed_statuses]:
                break

            time.sleep(poll_interval)
            retry_count += 1

        if video.status.lower() in ["failed", "error"]:
            error_msg = getattr(getattr(video, "error", None), "message", "Unknown error")
            raise RuntimeError(f"Failed to generate Sora2 video: {error_msg}")

        return video

    def get_operator(self) -> Optional[Sora2Operator]:
        """获取 operator 实例"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[Sora2Synthesis]:
        """获取 synthesis 模型实例"""
        return self.synthesis_model
