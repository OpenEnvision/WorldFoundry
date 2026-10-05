"""Hailuo 2P3 visual generation pipeline module."""

from ..pipeline_utils import PipelineABC
import logging
import time
from typing import Optional, Dict, Any, Union

from PIL import Image

from ...operators.hailuo_2p3_operator import Hailuo2p3Operator
from ...synthesis.visual_generation.minimax.hailuo_2p3_synthesis import Hailuo2p3Synthesis
from ..api_runtime import load_api_pipeline_from_pretrained, resolve_api_key


_API_KEY_ENV = ("MINIMAX_API_KEY", "HAILUO_API_KEY")

logger = logging.getLogger(__name__)


class Hailuo2p3Pipeline(PipelineABC):
    """
    MiniMax Hailuo 2.3 API Pipeline。
    """

    def __init__(
        self,
        operator: Optional[Hailuo2p3Operator] = None,
        synthesis_model: Optional[Hailuo2p3Synthesis] = None,
        endpoint: str = "https://api.minimax.io/v1",
        api_key: str = "your_api_key",
    ):
        """Initialize the pipeline and configure runtime components."""
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "MiniMax Hailuo")
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
    ) -> "Hailuo2p3Pipeline":
        """Build the API-only pipeline through the unified loader contract."""
        return load_api_pipeline_from_pretrained(
            cls,
            model_path=model_path,
            required_components=required_components,
            device=device,
            model_id=model_id,
            default_endpoint="https://api.minimax.io/v1",
            service_name="MiniMax Hailuo 2.3",
            **kwargs,
        )

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://api.minimax.io/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ) -> "Hailuo2p3Pipeline":
        """Initialize API client credentials and runtime endpoints."""
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "MiniMax Hailuo")
        synthesis_model = Hailuo2p3Synthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
            **kwargs,
        )
        operator = Hailuo2p3Operator()
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
            **kwargs,
        )
        processed_data["first_frame_image"] = processed_perception["first_frame_image"]
        processed_data["images"] = processed_perception["images"]

        return processed_data

    def _extract_task_id(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract the task identifier from a service response."""
        return response.get("task_id")

    def _extract_status(self, response: Dict[str, Any]) -> str:
        """Extract status for Hailuo2p3Pipeline."""
        return response.get("status", "")

    def _extract_file_id(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract file id for Hailuo2p3Pipeline."""
        return response.get("file_id") or response.get("file", {}).get("file_id")

    def _extract_download_url(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract download url for Hailuo2p3Pipeline."""
        return response.get("file", {}).get("download_url") or response.get("download_url")

    def _poll_task_status(
        self,
        task_id: str,
        poll_interval: int = 10,
        max_retries: int = 120,
    ) -> Dict[str, Any]:
        """Poll task status for Hailuo2p3Pipeline."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        completed_statuses = {"SUCCESS"}
        failed_statuses = {"FAIL"}
        last_status = None

        for _ in range(max_retries):
            task_info = self.synthesis_model.query_task(task_id)
            task_status = self._extract_status(task_info).upper()

            if task_status and task_status != last_status:
                logger.info("Hailuo task %s: %s", task_id, task_status)
                last_status = task_status

            if task_status in completed_statuses:
                return task_info
            if task_status in failed_statuses:
                return task_info

            time.sleep(poll_interval)

        raise TimeoutError(f"Hailuo task {task_id} polling timed out.")

    def __call__(
        self,
        prompt: str,
        images: Optional[Union[Image.Image, str]] = None,
        task_type: str = "auto",
        model: str = "MiniMax-Hailuo-2.3",
        resolution: str = "768P",
        duration: int = 6,
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
            **kwargs,
        )

        result = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            model=model,
            resolution=resolution,
            duration=duration,
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
        file_id = self._extract_file_id(result["response"])
        result["file_id"] = file_id

        if wait and file_id:
            file_response = self.synthesis_model.retrieve_file(file_id)
            result["file_response"] = file_response
            result["video_url"] = self._extract_download_url(file_response)
        else:
            result["video_url"] = None

        if output_path and result["video_url"]:
            saved_path = self.synthesis_model.download_video(
                result["video_url"],
                output_path,
            )
            result["output_path"] = saved_path

        return result

    def get_operator(self) -> Optional[Hailuo2p3Operator]:
        """Get operator for Hailuo2p3Pipeline."""
        return self.operator

    def get_synthesis_model(self) -> Optional[Hailuo2p3Synthesis]:
        """Get synthesis model for Hailuo2p3Pipeline."""
        return self.synthesis_model
