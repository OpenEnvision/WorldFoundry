"""Luma Ray2 visual generation pipeline module."""

from ..pipeline_utils import PipelineABC
import logging
import time
from typing import Optional, Dict, Any

from ...operators.luma_ray2_operator import LumaRay2Operator
from ...synthesis.visual_generation.luma.luma_ray2_synthesis import LumaRay2Synthesis
from ..api_runtime import load_api_pipeline_from_pretrained, resolve_api_key


_API_KEY_ENV = ("LUMA_API_KEY", "LUMALABS_API_KEY")

logger = logging.getLogger(__name__)


class LumaRay2Pipeline(PipelineABC):
    """
    Luma Ray-2 API Pipeline。
    """

    def __init__(
        self,
        operator: Optional[LumaRay2Operator] = None,
        synthesis_model: Optional[LumaRay2Synthesis] = None,
        endpoint: str = "https://api.lumalabs.ai/dream-machine/v1",
        api_key: str = "your_api_key",
    ):
        """Initialize the pipeline and configure runtime components."""
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Luma")
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
    ) -> "LumaRay2Pipeline":
        """Build the API-only pipeline through the unified loader contract."""
        return load_api_pipeline_from_pretrained(
            cls,
            model_path=model_path,
            required_components=required_components,
            device=device,
            model_id=model_id,
            default_endpoint="https://api.lumalabs.ai/dream-machine/v1",
            service_name="Luma Ray-2",
            **kwargs,
        )

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://api.lumalabs.ai/dream-machine/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ) -> "LumaRay2Pipeline":
        """Initialize API client credentials and runtime endpoints."""
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Luma")
        synthesis_model = LumaRay2Synthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
            **kwargs,
        )
        operator = LumaRay2Operator()
        return cls(
            operator=operator,
            synthesis_model=synthesis_model,
            endpoint=endpoint,
            api_key=api_key,
        )

    def process(
        self,
        prompt: str,
        images: Optional[str] = None,
        last_frame: Optional[str] = None,
        start_generation_id: Optional[str] = None,
        end_generation_id: Optional[str] = None,
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
            start_generation_id=start_generation_id,
            end_generation_id=end_generation_id,
            **kwargs,
        )
        processed_data["keyframes"] = processed_perception["keyframes"]
        processed_data["images"] = processed_perception["images"]
        processed_data["last_frame"] = processed_perception["last_frame"]
        processed_data["start_generation_id"] = processed_perception["start_generation_id"]
        processed_data["end_generation_id"] = processed_perception["end_generation_id"]

        return processed_data

    def _extract_generation_id(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract the generation identifier from the API response."""
        return response.get("id")

    def _extract_state(self, response: Dict[str, Any]) -> str:
        """Extract state information from the pipeline runtime metadata."""
        return response.get("state", "")

    def _extract_video_url(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract video url for LumaRay2Pipeline."""
        return response.get("assets", {}).get("video")

    def _poll_generation_status(
        self,
        generation_id: str,
        poll_interval: int = 5,
        max_retries: int = 180,
    ) -> Dict[str, Any]:
        """Poll generation status for LumaRay2Pipeline."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        completed_statuses = {"completed"}
        failed_statuses = {"failed"}
        last_state = None

        for _ in range(max_retries):
            generation_info = self.synthesis_model.get_generation(generation_id)
            state = self._extract_state(generation_info).lower()

            if state and state != last_state:
                logger.info("Luma generation %s: %s", generation_id, state)
                last_state = state

            if state in completed_statuses:
                return generation_info
            if state in failed_statuses:
                return generation_info

            time.sleep(poll_interval)

        raise TimeoutError(f"Luma generation {generation_id} polling timed out.")

    def __call__(
        self,
        prompt: str,
        images: Optional[str] = None,
        last_frame: Optional[str] = None,
        start_generation_id: Optional[str] = None,
        end_generation_id: Optional[str] = None,
        task_type: str = "auto",
        model: str = "ray-2",
        resolution: str = "720p",
        duration: str = "5s",
        aspect_ratio: Optional[str] = "16:9",
        loop: bool = False,
        concepts: Optional[Any] = None,
        callback_url: Optional[str] = None,
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

        processed_data = self.process(
            prompt=prompt,
            images=images,
            last_frame=last_frame,
            start_generation_id=start_generation_id,
            end_generation_id=end_generation_id,
            **kwargs,
        )

        result = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            model=model,
            resolution=resolution,
            duration=duration,
            aspect_ratio=aspect_ratio,
            loop=loop,
            concepts=concepts,
            callback_url=callback_url,
            **kwargs,
        )

        response = result["response"]
        generation_id = self._extract_generation_id(response)
        result["generation_id"] = generation_id

        if wait and generation_id:
            response = self._poll_generation_status(
                generation_id=generation_id,
                poll_interval=poll_interval,
                max_retries=max_retries,
            )
            result["response"] = response

        result["task_status"] = self._extract_state(result["response"])
        result["video_url"] = self._extract_video_url(result["response"])

        if output_path and result["video_url"]:
            saved_path = self.synthesis_model.download_video(
                result["video_url"],
                output_path,
            )
            result["output_path"] = saved_path

        return result

    def get_operator(self) -> Optional[LumaRay2Operator]:
        """Get operator for LumaRay2Pipeline."""
        return self.operator

    def get_synthesis_model(self) -> Optional[LumaRay2Synthesis]:
        """Get synthesis model for LumaRay2Pipeline."""
        return self.synthesis_model
