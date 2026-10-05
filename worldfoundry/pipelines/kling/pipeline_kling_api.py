"""Kling Api visual generation pipeline module."""

from ..pipeline_utils import PipelineABC
import logging
import time
from typing import Optional, Dict, Any, Union

from PIL import Image

from ...operators.kling_api_operator import KlingApiOperator
from ...synthesis.visual_generation.kling.kling_api_synthesis import KlingApiSynthesis
from ..api_runtime import load_api_pipeline_from_pretrained, resolve_api_key


_API_KEY_ENV = ("KLING_API_KEY", "KLING_ACCESS_TOKEN")

logger = logging.getLogger(__name__)


class KlingApiPipeline(PipelineABC):
    """
    Kling API Pipeline。
    """

    def __init__(
        self,
        operator: Optional[KlingApiOperator] = None,
        synthesis_model: Optional[KlingApiSynthesis] = None,
        endpoint: str = "https://api.klingapi.com",
        api_key: str = "your_api_key",
    ):
        """Initialize the pipeline and configure runtime components."""
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Kling")
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
    ) -> "KlingApiPipeline":
        """Build the API-only pipeline through the unified loader contract."""
        return load_api_pipeline_from_pretrained(
            cls,
            model_path=model_path,
            required_components=required_components,
            device=device,
            model_id=model_id,
            default_endpoint="https://api.klingapi.com",
            service_name="Kling API",
            **kwargs,
        )

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://api.klingapi.com",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ) -> "KlingApiPipeline":
        """Initialize API client credentials and runtime endpoints."""
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Kling")
        synthesis_model = KlingApiSynthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
            **kwargs,
        )
        operator = KlingApiOperator()
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
        image_field: Optional[str] = None,
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
            image_field=image_field,
            **kwargs,
        )
        processed_data["image_payload"] = processed_perception["image_payload"]
        processed_data["images"] = processed_perception["images"]
        return processed_data

    def _extract_task_id(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract the task identifier from a service response."""
        return (
            response.get("task_id")
            or response.get("id")
            or response.get("data", {}).get("task_id")
        )

    def _extract_status(self, response: Dict[str, Any]) -> str:
        """Extract status for KlingApiPipeline."""
        return (
            response.get("status")
            or response.get("task_status")
            or response.get("data", {}).get("status")
            or response.get("data", {}).get("task_status")
            or ""
        )

    def _extract_video_url(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract video url for KlingApiPipeline."""
        candidates = [
            response.get("video_url"),
            response.get("url"),
            response.get("download_url"),
            response.get("data", {}).get("video_url"),
            response.get("data", {}).get("url"),
            response.get("data", {}).get("download_url"),
        ]

        task_result = response.get("task_result") or response.get("data", {}).get("task_result")
        if isinstance(task_result, dict):
            candidates.extend([
                task_result.get("video_url"),
                task_result.get("url"),
                task_result.get("download_url"),
            ])
            videos = task_result.get("videos")
            if isinstance(videos, list) and videos:
                first_item = videos[0]
                if isinstance(first_item, str):
                    candidates.append(first_item)
                elif isinstance(first_item, dict):
                    candidates.extend([
                        first_item.get("url"),
                        first_item.get("video_url"),
                        first_item.get("download_url"),
                    ])

        for candidate in candidates:
            if candidate:
                return candidate
        return None

    def _poll_task_status(
        self,
        task_id: str,
        poll_interval: int = 10,
        max_retries: int = 120,
    ) -> Dict[str, Any]:
        """Poll task status for KlingApiPipeline."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        last_status = None
        completed_markers = ("success", "succeed", "completed", "finish", "done")
        failed_markers = ("fail", "error", "cancel")

        for _ in range(max_retries):
            task_info = self.synthesis_model.get_task(task_id)
            status = self._extract_status(task_info).lower()
            video_url = self._extract_video_url(task_info)

            if status and status != last_status:
                logger.info("Kling task %s: %s", task_id, status)
                last_status = status

            if video_url and not any(marker in status for marker in failed_markers):
                return task_info
            if any(marker in status for marker in completed_markers):
                return task_info
            if any(marker in status for marker in failed_markers):
                return task_info

            time.sleep(poll_interval)

        raise TimeoutError(f"Kling task {task_id} polling timed out.")

    def __call__(
        self,
        prompt: str,
        images: Optional[Union[Image.Image, str]] = None,
        image_field: Optional[str] = None,
        task_type: str = "auto",
        model: str = "kling-v2.6-pro",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        mode: str = "professional",
        negative_prompt: Optional[str] = None,
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
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
            image_field=image_field,
            **kwargs,
        )

        result = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            model=model,
            duration=duration,
            aspect_ratio=aspect_ratio,
            mode=mode,
            negative_prompt=negative_prompt,
            callback_url=callback_url,
            external_task_id=external_task_id,
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

        result["task_status"] = self._extract_status(result["response"])
        result["video_url"] = self._extract_video_url(result["response"])

        if output_path and result["video_url"]:
            saved_path = self.synthesis_model.download_video(
                result["video_url"],
                output_path,
            )
            result["output_path"] = saved_path

        return result

    def get_operator(self) -> Optional[KlingApiOperator]:
        """Get operator for KlingApiPipeline."""
        return self.operator

    def get_synthesis_model(self) -> Optional[KlingApiSynthesis]:
        """Get synthesis model for KlingApiPipeline."""
        return self.synthesis_model
