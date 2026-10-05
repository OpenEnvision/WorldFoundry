"""Runway Gen4P5 visual generation pipeline module."""

import logging
import os
import time
from typing import Optional, Dict, Any, Union

from PIL import Image

from ..pipeline_utils import PipelineABC
from ...operators.runway_gen4p5_operator import RunwayGen4p5Operator
from ...synthesis.visual_generation.runway.runway_gen4p5_synthesis import RunwayGen4p5Synthesis


_DEFAULT_ENDPOINT = "https://api.dev.runwayml.com/v1"
_PLACEHOLDER_KEYS = {"your_api_key", "your api key"}

logger = logging.getLogger(__name__)


def _resolve_api_key(api_key: Optional[str]) -> str:
    """Resolve the Runway API key from explicit input or RUNWAY_API_KEY.

    Args:
        api_key: Optional API key passed by the caller.
    """
    if api_key and api_key.strip() and api_key.strip() not in _PLACEHOLDER_KEYS:
        return api_key.strip()
    # Attempt to retrieve from environment variables as a fallback resolution
    env_value = os.getenv("RUNWAY_API_KEY")
    if env_value and env_value.strip():
        return env_value.strip()
    raise ValueError("Runway API key is required. Pass api_key or set RUNWAY_API_KEY.")


def _resolve_endpoint(endpoint: Optional[str]) -> str:
    """Resolve the Runway API endpoint from explicit input, env, or default.

    Args:
        endpoint: Optional endpoint URL passed by the caller.
    """
    if endpoint and endpoint.strip():
        return endpoint.strip()
    # Attempt to retrieve from environment variables as a fallback resolution
    env_value = os.getenv("RUNWAY_ENDPOINT")
    if env_value and env_value.strip():
        return env_value.strip()
    return _DEFAULT_ENDPOINT


class RunwayGen4p5Pipeline(PipelineABC):
    """
    Runway Gen-4.5 API Pipeline。
    """

    def __init__(
        self,
        operator: Optional[RunwayGen4p5Operator] = None,
        synthesis_model: Optional[RunwayGen4p5Synthesis] = None,
        endpoint: str = "https://api.dev.runwayml.com/v1",
        api_key: str = "your_api_key",
    ):
        """Initialize the pipeline and configure runtime components."""
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
    ) -> "RunwayGen4p5Pipeline":
        """Build the API-only Runway Gen-4.5 pipeline without loading a checkpoint.

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
            raise ValueError("Runway Gen-4.5 is API-only; pass API options as a dict instead of a checkpoint path.")
        options.update(required_components or {})
        options.update(kwargs)
        return cls.api_init(
            endpoint=options.get("endpoint"),
            api_key=options.get("api_key"),
            runway_version=options.get("runway_version", "2024-11-06"),
            logger=options.get("logger"),
        )

    @classmethod
    def api_init(
        cls,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        runway_version: str = "2024-11-06",
        logger=None,
        **kwargs
    ) -> "RunwayGen4p5Pipeline":
        """Initialize API client credentials and runtime endpoints."""
        endpoint = _resolve_endpoint(endpoint)
        api_key = _resolve_api_key(api_key)
        synthesis_model = RunwayGen4p5Synthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            runway_version=runway_version,
            logger=logger,
            **kwargs,
        )
        operator = RunwayGen4p5Operator()
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
        last_frame: Optional[Union[Image.Image, str]] = None,
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
            last_frame=last_frame,
            **kwargs,
        )
        processed_data["prompt_image"] = processed_perception["prompt_image"]
        processed_data["images"] = processed_perception["images"]
        processed_data["last_frame"] = processed_perception["last_frame"]

        return processed_data

    def _extract_task_id(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract the task identifier from a service response."""
        return response.get("id")

    def _extract_task_status(self, response: Dict[str, Any]) -> str:
        """Query and extract the task status from a service response."""
        return response.get("status", "")

    def _extract_video_url(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract video url for RunwayGen4p5Pipeline."""
        output = response.get("output")
        if isinstance(output, list) and output:
            first_item = output[0]
            if isinstance(first_item, str):
                return first_item
            if isinstance(first_item, dict):
                return first_item.get("url")
        return None

    def _poll_task_status(
        self,
        task_id: str,
        poll_interval: int = 5,
        max_retries: int = 180,
    ) -> Dict[str, Any]:
        """Poll task status for RunwayGen4p5Pipeline."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        completed_statuses = {"SUCCEEDED"}
        failed_statuses = {"FAILED", "CANCELLED"}
        last_status = None

        for _ in range(max_retries):
            task_info = self.synthesis_model.get_task(task_id)
            task_status = self._extract_task_status(task_info).upper()

            if task_status and task_status != last_status:
                logger.info("Runway task %s: %s", task_id, task_status)
                last_status = task_status

            if task_status in completed_statuses:
                return task_info
            if task_status in failed_statuses:
                return task_info

            time.sleep(poll_interval)

        raise TimeoutError(f"Runway task {task_id} polling timed out.")

    def __call__(
        self,
        prompt: str,
        images: Optional[Union[Image.Image, str]] = None,
        last_frame: Optional[Union[Image.Image, str]] = None,
        task_type: str = "auto",
        model: str = "gen4.5",
        ratio: str = "1280:720",
        duration: int = 5,
        seed: Optional[int] = None,
        public_figure_threshold: Optional[str] = None,
        wait: bool = True,
        poll_interval: int = 5,
        max_retries: int = 180,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        _resolve_api_key(self.api_key)

        processed_data = self.process(
            prompt=prompt,
            images=images,
            last_frame=last_frame,
            **kwargs,
        )

        result = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            model=model,
            ratio=ratio,
            duration=duration,
            seed=seed,
            public_figure_threshold=public_figure_threshold,
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

    def get_operator(self) -> Optional[RunwayGen4p5Operator]:
        """Get operator for RunwayGen4p5Pipeline."""
        return self.operator

    def get_synthesis_model(self) -> Optional[RunwayGen4p5Synthesis]:
        """Get synthesis model for RunwayGen4p5Pipeline."""
        return self.synthesis_model
