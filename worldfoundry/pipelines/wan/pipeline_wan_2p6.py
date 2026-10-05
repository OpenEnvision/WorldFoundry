"""Wan 2P6 visual generation pipeline module."""

from ..pipeline_utils import PipelineABC
import logging
from typing import Optional, Dict, Any, List, Union

from PIL import Image

from ...operators.wan_2p6_operator import Wan2p6Operator
from ...synthesis.visual_generation.wan.wan_2p6_synthesis import Wan2p6Synthesis
from ..api_runtime import load_api_pipeline_from_pretrained, resolve_api_key
from ._hosted_polling import poll_wan_task_status


_API_KEY_ENV = ("DASHSCOPE_API_KEY", "WAN_API_KEY", "ALIYUN_API_KEY")

logger = logging.getLogger(__name__)


class Wan2p6Pipeline(PipelineABC):
    """
    Wan2.6 API Pipeline。
    """

    def __init__(
        self,
        operator: Optional[Wan2p6Operator] = None,
        synthesis_model: Optional[Wan2p6Synthesis] = None,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
    ):
        """Initialize the pipeline and configure runtime components."""
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Wan2.6")
        self.endpoint = endpoint
        self.api_key = api_key
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
    ) -> "Wan2p6Pipeline":
        """Build the API-only pipeline through the unified loader contract."""
        return load_api_pipeline_from_pretrained(
            cls,
            model_path=model_path,
            required_components=required_components,
            device=device,
            model_id=model_id,
            default_endpoint="https://dashscope.aliyuncs.com/api/v1",
            service_name="Wan 2.6",
            **kwargs,
        )

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ) -> "Wan2p6Pipeline":
        """Initialize API client credentials and runtime endpoints."""
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Wan2.6")
        synthesis_model = Wan2p6Synthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
            **kwargs,
        )
        operator = Wan2p6Operator()
        return cls(
            operator=operator,
            synthesis_model=synthesis_model,
            endpoint=endpoint,
            api_key=api_key,
        )

    def process(
        self,
        prompt: str,
        images: Optional[Union[Image.Image, str]] = None,
        reference_urls: Optional[List[str]] = None,
        audio_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if self.operator is None:
            raise ValueError("Operator is not initialized")

        processed_data: Dict[str, Any] = {}

        self.operator.get_interaction(prompt)
        processed_interaction = self.operator.process_interaction()
        processed_data["prompt"] = processed_interaction["processed_prompt"]

        processed_perception = self.operator.process_perception(
            images=images,
            reference_urls=reference_urls,
            audio_url=audio_url,
            **kwargs,
        )
        processed_data["encoded_image"] = processed_perception["encoded_image"]
        processed_data["images"] = processed_perception["images"]
        processed_data["reference_urls"] = processed_perception["reference_urls"]
        processed_data["audio_url"] = processed_perception["audio_url"]

        return processed_data

    def _extract_task_id(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract the task identifier from a service response."""
        return response.get("output", {}).get("task_id")

    def _extract_task_status(self, response: Dict[str, Any]) -> str:
        """Query and extract the task status from a service response."""
        return response.get("output", {}).get("task_status", "")

    def _extract_video_url(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract video url for Wan2p6Pipeline."""
        return response.get("output", {}).get("video_url")

    def _poll_task_status(
        self,
        task_id: str,
        poll_interval: int = 10,
        max_retries: int = 120,
    ) -> Dict[str, Any]:
        """Poll task status for Wan2p6Pipeline."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        return poll_wan_task_status(
            get_task=self.synthesis_model.get_task,
            extract_status=self._extract_task_status,
            task_id=task_id,
            service_label="Wan2.6",
            logger=logger,
            poll_interval=poll_interval,
            max_retries=max_retries,
        )

    def __call__(
        self,
        prompt: str,
        images: Optional[Union[Image.Image, str]] = None,
        reference_urls: Optional[List[str]] = None,
        audio_url: Optional[str] = None,
        task_type: str = "auto",
        model: Optional[str] = None,
        size: str = "1280*720",
        resolution: str = "720P",
        duration: int = 5,
        negative_prompt: str = "",
        audio: Optional[bool] = None,
        prompt_extend: bool = True,
        shot_type: Optional[str] = None,
        watermark: bool = False,
        seed: Optional[int] = None,
        wait: bool = True,
        poll_interval: int = 10,
        max_retries: int = 120,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        if self.operator is None:
            raise ValueError("Operator is not initialized")

        processed_data = self.process(
            prompt=prompt,
            images=images,
            reference_urls=reference_urls,
            audio_url=audio_url,
            **kwargs,
        )

        result = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            model=model,
            size=size,
            resolution=resolution,
            duration=duration,
            negative_prompt=negative_prompt,
            audio=audio,
            prompt_extend=prompt_extend,
            shot_type=shot_type,
            watermark=watermark,
            seed=seed,
            **kwargs,
        )

        response = result["response"]
        task_id = self._extract_task_id(response)
        result["task_id"] = task_id

        if wait and task_id:
            response = self._poll_task_status(
                task_id=task_id,
                poll_interval=poll_interval,
                max_retries=max_retries,
            )
            result["response"] = response

        result["task_status"] = self._extract_task_status(result["response"])
        result["video_url"] = self._extract_video_url(result["response"])

        if output_path and result["video_url"]:
            saved_path = self.synthesis_model.download_video(
                result["video_url"],
                output_path,
            )
            result["output_path"] = saved_path

        return result

    def get_operator(self) -> Optional[Wan2p6Operator]:
        """Get operator for Wan2p6Pipeline."""
        return self.operator

    def get_synthesis_model(self) -> Optional[Wan2p6Synthesis]:
        """Get synthesis model for Wan2p6Pipeline."""
        return self.synthesis_model
